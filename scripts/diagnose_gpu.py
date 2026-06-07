from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grpo_spad import format_prompt, load_config, load_model


def _optional_import_status(module_name: str) -> str:
    try:
        __import__(module_name)
    except Exception as exc:
        return f"missing ({type(exc).__name__}: {exc})"
    return "ok"


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose GPU placement and generation speed.")
    parser.add_argument("--config", default="configs/formal/spad_rl_only.yaml")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--benchmark-runs", type=int, default=3)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument(
        "--question",
        default="If Tom has 3 apples and buys 4 more, how many apples does he have?",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.max_completion_length = args.max_new_tokens

    print(f"[CONFIG] {args.config}")
    print(f"[TORCH] version={torch.__version__} cuda_runtime={torch.version.cuda} cuda_available={torch.cuda.is_available()}")
    print(f"[OPTIONAL] causal_conv1d={_optional_import_status('causal_conv1d')}")
    print(f"[OPTIONAL] fla={_optional_import_status('fla')}")
    print(f"[CONFIG] load_in_4bit={cfg.load_in_4bit} dtype={cfg.dtype} max_completion_length={cfg.max_completion_length}")
    if torch.cuda.is_available():
        print(f"[CUDA] device_count={torch.cuda.device_count()} current={torch.cuda.current_device()} name={torch.cuda.get_device_name(0)}")

    model, tokenizer = load_model(cfg)
    model.eval()

    print(f"[MODEL] device={getattr(model, 'device', None)}")
    print(f"[MODEL] hf_device_map={getattr(model, 'hf_device_map', None)}")
    for name, param in model.named_parameters():
        print(f"[PARAM] first={name} device={param.device} dtype={param.dtype} requires_grad={param.requires_grad}")
        break

    prompt = format_prompt(tokenizer, args.question, "gsm8k")
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    print(f"[INPUT] tokens={input_ids.shape[1]} device={input_ids.device}")

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": not args.greedy,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if not args.greedy:
        generation_kwargs.update({"temperature": 0.7, "top_p": 0.9})

    total_runs = max(args.warmup_runs, 0) + max(args.benchmark_runs, 1)
    benchmark_speeds = []
    completion = ""
    for run_idx in range(total_runs):
        phase = "warmup" if run_idx < args.warmup_runs else "bench"
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.time()
        with torch.inference_mode():
            output = model.generate(input_ids, **generation_kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - start

        completion_ids = output[0, input_ids.shape[1]:]
        speed = completion_ids.shape[0] / max(elapsed, 1e-9)
        if phase == "bench":
            benchmark_speeds.append(speed)
        completion = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        print(
            f"[GENERATE:{phase}:{run_idx + 1}] "
            f"new_tokens={completion_ids.shape[0]} seconds={elapsed:.3f} tok_per_s={speed:.3f}"
        )

    if benchmark_speeds:
        mean_speed = sum(benchmark_speeds) / len(benchmark_speeds)
        print(f"[GENERATE:bench:mean] tok_per_s={mean_speed:.3f}")
    print("[COMPLETION]")
    print(completion)

    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        reserved_gb = torch.cuda.max_memory_reserved() / 1024**3
        print(f"[CUDA] peak_allocated_gb={peak_gb:.3f} peak_reserved_gb={reserved_gb:.3f}")


if __name__ == "__main__":
    main()
