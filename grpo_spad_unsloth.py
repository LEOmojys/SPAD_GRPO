"""
Unsloth-backed entry point for baseline / SPAD / C-SPAD GRPO.

Import order matters: Unsloth must be imported before TRL/Transformers so its
patches can be applied.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Optional


def _set_compile_cache_env() -> None:
    cache_root = Path.cwd() / ".cache"
    os.environ["TRITON_CACHE_DIR"] = str((cache_root / "triton").resolve())
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str((cache_root / "torchinductor").resolve())
    os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "0"


def _import_unsloth():
    _set_compile_cache_env()
    try:
        from unsloth import FastLanguageModel
    except Exception as exc:  # pragma: no cover - optional dependency path.
        raise RuntimeError(
            "Unsloth is not installed or failed to import. Install it in a compatible "
            "environment first, then rerun this script. See ENVIRONMENT.md for notes."
        ) from exc
    _set_compile_cache_env()
    return FastLanguageModel


def _dtype_from_config(cfg: Any):
    import torch

    if cfg.dtype == "bfloat16":
        return torch.bfloat16
    if cfg.dtype == "float16":
        return torch.float16
    return None


def _set_mixed_precision_env(dtype_name: str) -> None:
    if dtype_name == "bfloat16":
        os.environ["ACCELERATE_MIXED_PRECISION"] = "bf16"
    elif dtype_name == "float16":
        os.environ["ACCELERATE_MIXED_PRECISION"] = "fp16"


def _patch_text_config_aliases(model: Any) -> None:
    alias_names = (
        "head_dim",
        "hidden_size",
        "layer_types",
        "max_position_embeddings",
        "num_attention_heads",
        "num_hidden_layers",
        "num_key_value_heads",
        "vocab_size",
    )
    configs = []
    for candidate in (model, getattr(model, "base_model", None), getattr(model, "model", None)):
        config = getattr(candidate, "config", None)
        if config is not None and config not in configs:
            configs.append(config)

    for config in configs:
        text_config = getattr(config, "text_config", None)
        if text_config is None:
            continue
        config_cls = config.__class__
        for name in alias_names:
            if not hasattr(config_cls, name):

                def getter(self, alias_name=name):
                    if alias_name in self.__dict__:
                        return self.__dict__[alias_name]
                    nested_text_config = getattr(self, "text_config", None)
                    if nested_text_config is not None and hasattr(nested_text_config, alias_name):
                        return getattr(nested_text_config, alias_name)
                    raise AttributeError(alias_name)

                def setter(self, value, alias_name=name):
                    self.__dict__[alias_name] = value

                setattr(config_cls, name, property(getter, setter))
            if not hasattr(config, name) and hasattr(text_config, name):
                setattr(config, name, getattr(text_config, name))
        if not hasattr(config, "sliding_window"):
            setattr(config, "sliding_window", None)


def _get_model_configs(model: Any) -> list[Any]:
    configs = []
    for candidate in (model, getattr(model, "base_model", None), getattr(model, "model", None)):
        config = getattr(candidate, "config", None)
        if config is not None and config not in configs:
            configs.append(config)
    return configs


def _has_linear_attention_layer_types(model: Any) -> bool:
    for config in _get_model_configs(model):
        text_config = getattr(config, "text_config", None)
        for candidate in (config, text_config):
            if candidate is None:
                continue
            layer_types = getattr(candidate, "layer_types", None)
            if layer_types and any(layer_type == "linear_attention" for layer_type in layer_types):
                return True
    return False


def _clear_generation_max_length(model: Any) -> None:
    seen = set()
    stack = [model]
    while stack:
        candidate = stack.pop()
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        generation_config = getattr(candidate, "generation_config", None)
        if generation_config is not None and getattr(generation_config, "max_length", None) is not None:
            generation_config.max_length = None
        for attr_name in ("base_model", "model", "module"):
            stack.append(getattr(candidate, attr_name, None))


def load_unsloth_model(
    cfg: Any,
    *,
    fast_inference: bool = False,
    gpu_memory_utilization: float = 0.6,
    for_eval: bool = False,
):
    _set_mixed_precision_env(cfg.dtype)
    FastLanguageModel = _import_unsloth()
    kwargs = {
        "model_name": cfg.model_name,
        "max_seq_length": cfg.max_seq_length + cfg.max_completion_length,
        "dtype": _dtype_from_config(cfg),
        "load_in_4bit": cfg.load_in_4bit,
        "trust_remote_code": True,
    }
    if fast_inference:
        kwargs.update(
            {
                "fast_inference": True,
                "gpu_memory_utilization": gpu_memory_utilization,
                "max_lora_rank": cfg.lora_r,
            }
        )

    model, tokenizer = FastLanguageModel.from_pretrained(**kwargs)
    _patch_text_config_aliases(model)
    _clear_generation_max_length(model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_r,
        target_modules=cfg.lora_target_modules,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias=cfg.lora_bias,
        use_gradient_checkpointing="unsloth",
        random_state=cfg.seed,
    )
    _patch_text_config_aliases(model)
    _clear_generation_max_length(model)
    if for_eval and hasattr(FastLanguageModel, "for_inference"):
        FastLanguageModel.for_inference(model)
    elif hasattr(FastLanguageModel, "for_training"):
        FastLanguageModel.for_training(model)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] Loaded {cfg.model_name} with Unsloth. Trainable params: {trainable:,}")
    return model, tokenizer


def load_unsloth_checkpoint_for_eval(
    cfg: Any,
    checkpoint_dir: str,
    *,
    fast_inference: bool = False,
    gpu_memory_utilization: float = 0.6,
):
    from peft import PeftModel

    _set_mixed_precision_env(cfg.dtype)
    FastLanguageModel = _import_unsloth()
    kwargs = {
        "model_name": cfg.model_name,
        "max_seq_length": cfg.max_seq_length + cfg.max_completion_length,
        "dtype": _dtype_from_config(cfg),
        "load_in_4bit": cfg.load_in_4bit,
        "trust_remote_code": True,
    }
    if fast_inference:
        kwargs.update(
            {
                "fast_inference": True,
                "gpu_memory_utilization": gpu_memory_utilization,
                "max_lora_rank": cfg.lora_r,
            }
        )
    model, tokenizer = FastLanguageModel.from_pretrained(**kwargs)
    model = PeftModel.from_pretrained(model, checkpoint_dir, is_trainable=False)
    _patch_text_config_aliases(model)
    _clear_generation_max_length(model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(FastLanguageModel, "for_inference"):
        FastLanguageModel.for_inference(model)
    model.eval()
    print(f"[MODEL] Loaded Unsloth eval checkpoint {checkpoint_dir}")
    return model, tokenizer


def train_unsloth(
    cfg: Any,
    mode: str,
    output_dir: Optional[str] = None,
    *,
    fast_inference: bool = False,
    use_transformers_paged: bool = False,
    cache_implementation: Optional[str] = None,
    generation_batch_size: Optional[int] = None,
    use_vllm: bool = False,
    vllm_mode: str = "server",
    vllm_model_impl: str = "vllm",
    vllm_server_base_url: Optional[str] = None,
    vllm_server_host: str = "0.0.0.0",
    vllm_server_port: int = 8000,
    vllm_server_timeout: float = 240.0,
    gpu_memory_utilization: float = 0.6,
    vllm_tensor_parallel_size: int = 1,
    vllm_enable_sleep_mode: bool = False,
):
    _set_mixed_precision_env(cfg.dtype)
    output_dir = output_dir or f"{cfg.output_dir}_unsloth"
    if use_vllm and importlib.util.find_spec("vllm") is None:
        raise RuntimeError(
            "Requested --use-vllm, but vLLM is not installed in this environment. "
            "On Windows, use --use-transformers-paged for the local fallback, or "
            "install a compatible vLLM build before enabling --use-vllm."
        )
    _import_unsloth()
    os.makedirs(output_dir, exist_ok=True)

    from grpo_spad import build_dataset, quick_eval, sft_warmup
    from grpo_spad_trl import (
        SPADGRPOTrainer,
        build_trl_dataset,
        make_answer_reward_func,
        make_format_reward_func,
        make_trl_config,
    )

    sft_samples, rl_samples, eval_samples = build_dataset(cfg)
    model, tokenizer = load_unsloth_model(
        cfg,
        fast_inference=fast_inference,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    print(
        "[OPT] "
        f"max_prompt={cfg.max_seq_length}, max_completion={cfg.max_completion_length}, "
        f"lora_dropout={cfg.lora_dropout}, num_generations={cfg.num_generations}, "
        f"per_device_train_batch_size={cfg.per_device_train_batch_size}, "
        f"gradient_accumulation_steps={cfg.grad_accum_steps}, grpo_steps={cfg.grpo_steps}"
    )
    if use_transformers_paged and _has_linear_attention_layer_types(model):
        print(
            "[GEN] Transformers paged generation is incompatible with this model's "
            "linear_attention layers; falling back to default generation."
        )
        use_transformers_paged = False

    sft_warmup(model, tokenizer, sft_samples, cfg)

    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}
    base_model = getattr(model, "base_model", None)
    if base_model is not None and not hasattr(base_model, "warnings_issued"):
        base_model.warnings_issued = model.warnings_issued

    train_dataset = build_trl_dataset(tokenizer, rl_samples)
    eval_dataset = build_trl_dataset(tokenizer, eval_samples)
    trl_args = make_trl_config(
        cfg,
        output_dir,
        use_transformers_paged=use_transformers_paged,
        cache_implementation=cache_implementation,
        generation_batch_size=generation_batch_size,
        use_vllm=use_vllm,
        vllm_mode=vllm_mode,
        vllm_model_impl=vllm_model_impl,
        vllm_server_base_url=vllm_server_base_url,
        vllm_server_host=vllm_server_host,
        vllm_server_port=vllm_server_port,
        vllm_server_timeout=vllm_server_timeout,
        vllm_gpu_memory_utilization=gpu_memory_utilization,
        vllm_tensor_parallel_size=vllm_tensor_parallel_size,
        vllm_enable_sleep_mode=vllm_enable_sleep_mode,
    )
    if use_vllm:
        print(f"[GEN] Using vLLM generation mode={vllm_mode}.")
    elif use_transformers_paged:
        print("[GEN] Using Transformers paged generation.")
    elif cache_implementation:
        print(f"[GEN] Using Transformers generation cache_implementation={cache_implementation}.")
    else:
        print("[GEN] Using TRL default Transformers generation.")

    trainer = SPADGRPOTrainer(
        model=model,
        reward_funcs=[make_answer_reward_func(cfg), make_format_reward_func(cfg)],
        args=trl_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        spad_cfg=cfg,
        mode=mode,
    )
    trainer.train()
    trainer.save_model(Path(output_dir) / f"{mode}_checkpoint_final")

    eval_result = None
    if not getattr(cfg, "skip_final_eval", False):
        eval_result = quick_eval(
            trainer.model,
            tokenizer,
            eval_samples,
            cfg,
            cfg.eval_episodes,
            Path(output_dir) / f"eval_details_{mode}_unsloth.jsonl",
        )
    metrics = {
        "mode": mode,
        "backend": "unsloth",
        "final_eval": [] if eval_result is None else [eval_result],
        "trainer_log_history": trainer.state.log_history,
    }
    with open(Path(output_dir) / f"metrics_{mode}_unsloth.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    if eval_result is not None:
        print(f"[EVAL] {eval_result}")
    else:
        print("[EVAL] skipped final eval; run --eval-only for formal metrics.")
    print(f"[DONE] saved to {output_dir}")


def eval_unsloth_checkpoint(
    cfg: Any,
    mode: str,
    output_dir: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    *,
    fast_inference: bool = False,
    gpu_memory_utilization: float = 0.6,
):
    from grpo_spad import build_dataset, quick_eval

    output_dir = output_dir or f"{cfg.output_dir}_unsloth"
    checkpoint_dir = checkpoint_dir or str(Path(output_dir) / f"{mode}_checkpoint_final")
    os.makedirs(output_dir, exist_ok=True)

    _, _, eval_samples = build_dataset(cfg)
    model, tokenizer = load_unsloth_checkpoint_for_eval(
        cfg,
        checkpoint_dir,
        fast_inference=fast_inference,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    eval_result = quick_eval(
        model,
        tokenizer,
        eval_samples,
        cfg,
        cfg.eval_episodes,
        Path(output_dir) / f"eval_details_{mode}_unsloth.jsonl",
    )

    metrics_path = Path(output_dir) / f"metrics_{mode}_unsloth_eval.json"
    metrics = {
        "mode": mode,
        "backend": "unsloth",
        "checkpoint_dir": checkpoint_dir,
        "final_eval": [eval_result],
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"[EVAL] {eval_result}")
    print(f"[DONE] metrics saved to {metrics_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/formal/spad_rl_only.yaml")
    parser.add_argument("--mode", choices=["baseline", "spad", "c_spad"], default="spad")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument(
        "--fast-inference",
        action="store_true",
        help="Enable Unsloth fast inference at model load time. Usually requires a compatible vLLM setup.",
    )
    parser.add_argument(
        "--use-transformers-paged",
        action="store_true",
        help="Use TRL's Transformers paged generation path when vLLM is not enabled.",
    )
    parser.add_argument("--cache-implementation", default=None)
    parser.add_argument("--generation-batch-size", type=int, default=None)
    parser.add_argument("--use-vllm", action="store_true")
    parser.add_argument("--vllm-mode", choices=["server", "colocate"], default="server")
    parser.add_argument("--vllm-model-impl", choices=["vllm", "transformers"], default="vllm")
    parser.add_argument("--vllm-server-base-url", default=None)
    parser.add_argument("--vllm-server-host", default="0.0.0.0")
    parser.add_argument("--vllm-server-port", type=int, default=8000)
    parser.add_argument("--vllm-server-timeout", type=float, default=240.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--vllm-enable-sleep-mode", action="store_true")
    precision_group = parser.add_mutually_exclusive_group()
    precision_group.add_argument("--bf16", action="store_true", help="Override config dtype and run bfloat16 mixed precision.")
    precision_group.add_argument("--fp16", action="store_true", help="Override config dtype and run float16 mixed precision.")
    parser.add_argument("--eval-episodes", type=int, default=None, help="Override logging.eval_episodes for eval-only smoke/full eval.")
    parser.add_argument("--eval-pass-at-k", type=int, default=None, help="Override logging.eval_pass_at_k for eval-only smoke/full eval.")
    parser.add_argument("--eval-batch-size", type=int, default=None, help="Override logging.eval_batch_size for batched evaluation generation.")
    parser.add_argument("--max-completion-length", type=int, default=None, help="Override model.max_completion_length.")
    parser.add_argument("--num-generations", type=int, default=None, help="Override training.num_generations.")
    parser.add_argument("--per-device-train-batch-size", type=int, default=None, help="Override training.per_device_train_batch_size.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None, help="Override training.gradient_accumulation_steps.")
    parser.add_argument("--grpo-steps", type=int, default=None, help="Override training.grpo_steps.")
    parser.add_argument("--logging-steps", type=int, default=None, help="Override logging.logging_steps.")
    parser.add_argument("--save-steps", type=int, default=None, help="Override logging.save_steps. Use 0 to disable intermediate checkpoints.")
    parser.add_argument("--lora-dropout", type=float, default=None, help="Override lora.dropout. Use 0 for Unsloth fast LoRA patching.")
    parser.add_argument("--skip-final-eval", action="store_true", help="Skip automatic evaluation after training.")
    parser.add_argument(
        "--no-process-metrics",
        action="store_true",
        help="Skip heuristic process metric logging during baseline training. SPAD/C-SPAD still compute process signals.",
    )
    args = parser.parse_args()

    if args.bf16:
        _set_mixed_precision_env("bfloat16")
    elif args.fp16:
        _set_mixed_precision_env("float16")

    _import_unsloth()

    from grpo_spad import load_config

    cfg = load_config(args.config)
    if args.bf16:
        cfg.dtype = "bfloat16"
    elif args.fp16:
        cfg.dtype = "float16"
    if args.eval_episodes is not None:
        cfg.eval_episodes = args.eval_episodes
    if args.eval_pass_at_k is not None:
        cfg.eval_pass_at_k = args.eval_pass_at_k
    if args.eval_batch_size is not None:
        cfg.eval_batch_size = args.eval_batch_size
    if args.max_completion_length is not None:
        cfg.max_completion_length = args.max_completion_length
    if args.num_generations is not None:
        cfg.num_generations = args.num_generations
    if args.per_device_train_batch_size is not None:
        cfg.per_device_train_batch_size = args.per_device_train_batch_size
    if args.gradient_accumulation_steps is not None:
        cfg.grad_accum_steps = args.gradient_accumulation_steps
    if args.grpo_steps is not None:
        cfg.grpo_steps = args.grpo_steps
    if args.logging_steps is not None:
        cfg.log_steps = args.logging_steps
    if args.save_steps is not None:
        cfg.save_steps = args.save_steps
    if args.lora_dropout is not None:
        cfg.lora_dropout = args.lora_dropout
    if args.skip_final_eval:
        cfg.skip_final_eval = True
    if args.no_process_metrics:
        cfg.log_process_metrics = False

    if args.eval_only:
        eval_unsloth_checkpoint(
            cfg,
            args.mode,
            args.output_dir,
            args.checkpoint_dir,
            fast_inference=args.fast_inference,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
    else:
        train_unsloth(
            cfg,
            args.mode,
            args.output_dir,
            fast_inference=args.fast_inference,
            use_transformers_paged=args.use_transformers_paged,
            cache_implementation=args.cache_implementation,
            generation_batch_size=args.generation_batch_size,
            use_vllm=args.use_vllm,
            vllm_mode=args.vllm_mode,
            vllm_model_impl=args.vllm_model_impl,
            vllm_server_base_url=args.vllm_server_base_url,
            vllm_server_host=args.vllm_server_host,
            vllm_server_port=args.vllm_server_port,
            vllm_server_timeout=args.vllm_server_timeout,
            gpu_memory_utilization=args.gpu_memory_utilization,
            vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
            vllm_enable_sleep_mode=args.vllm_enable_sleep_mode,
        )


if __name__ == "__main__":
    main()
