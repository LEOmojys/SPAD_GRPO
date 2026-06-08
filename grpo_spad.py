"""
CoT GRPO training with SPAD and C-SPAD.

Modes:
  baseline: standard answer-reward GRPO
  spad:     statistical process advantage decomposition
  c_spad:   SPAD + process consistency reward and weighting
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from tqdm import tqdm

from process_signals import (
    MathSample,
    analyze_process,
    extract_final_answer,
    format_cot_solution,
    load_gsm8k_json,
    normalize_answer,
    process_metrics_dict,
    split_steps,
    verify_answer,
)


@dataclass
class SPADConfig:
    model_name: str = "./Models/qwen3_5_2b"
    max_seq_length: int = 512
    load_in_4bit: bool = True
    dtype: str = "bfloat16"

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "in_proj_qkv", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    data_file: str = "gsm8k_easy.json"
    sft_file: str = "gsm8k_easy.json"
    rl_file: str = "gsm8k_easy.json"
    eval_file: str = "gsm8k_easy.json"
    max_samples: int = 500
    sft_samples: int = 200
    rl_samples: int = 250
    eval_samples: int = 50

    sft_lr: float = 2e-4
    sft_steps: int = 100
    sft_batch_size: int = 2
    grpo_lr: float = 5e-5
    grpo_steps: int = 100
    group_size: int = 4
    num_generations: Optional[int] = None
    per_device_train_batch_size: Optional[int] = None
    grad_accum_steps: int = 4
    steps_per_generation: int = 1
    max_completion_length: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    epsilon_clip: float = 0.2
    epsilon_high: Optional[float] = None
    beta_kl: float = 0.01
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"
    warmup_steps: float = 0.0
    loss_type: str = "grpo"
    scale_rewards: str = "group"
    mask_truncated_completions: bool = False
    top_entropy_quantile: float = 1.0
    drop_zero_advantage_groups: bool = False
    entropy_advantage_weighting: bool = False
    entropy_weight_power: float = 1.0
    entropy_weight_min: float = 0.5
    entropy_weight_max: float = 2.0
    seed: int = 42

    answer_reward: float = 1.0
    wrong_reward: float = -0.5
    format_reward: float = 0.05
    consistency_reward_weight: float = 0.2
    repetition_penalty_weight: float = 0.1
    length_penalty_weight: float = 0.02
    process_tau: float = 1.0
    positive_score_threshold: float = 1.2
    negative_score_threshold: float = 0.8
    min_step_weight: float = 0.0
    consistency_weight: float = 0.8
    virtual_reward: float = 1.0
    zero_advantage_fix: bool = True
    positive_length_tiebreak: bool = True
    normalize_token_advantage: bool = False
    max_token_advantage_scale: float = 4.0

    eval_episodes: int = 50
    eval_pass_at_k: int = 8
    eval_batch_size: int = 1
    eval_sample_temperature: float = 1.0
    eval_sample_top_p: float = 0.9
    skip_final_eval: bool = False
    log_process_metrics: bool = True
    log_steps: int = 10
    save_steps: int = 100
    output_dir: str = "./outputs_spad"
    run_name: str = "spad_grpo_gsm8k"


@dataclass
class Trajectory:
    prompt: str
    completion: str
    reward: float
    adjusted_reward: float
    answer_correct: bool
    final_answer: str
    token_count: int
    step_advantages: List[float] = field(default_factory=list)
    process_metrics: Dict[str, float] = field(default_factory=dict)
    process_scores: List[float] = field(default_factory=list)


_AUTOCAST_DTYPE = torch.bfloat16


def torch_dtype_from_name(dtype_name: str) -> torch.dtype:
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def set_autocast_dtype(dtype_name: str) -> None:
    global _AUTOCAST_DTYPE
    dtype = torch_dtype_from_name(dtype_name)
    _AUTOCAST_DTYPE = torch.bfloat16 if dtype == torch.float32 else dtype


def load_config(path: str) -> SPADConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = SPADConfig()
    model = raw.get("model", {})
    cfg.model_name = model.get("name", cfg.model_name)
    cfg.max_seq_length = model.get("max_seq_length", cfg.max_seq_length)
    cfg.load_in_4bit = model.get("load_in_4bit", cfg.load_in_4bit)
    cfg.dtype = model.get("dtype", cfg.dtype)
    cfg.max_completion_length = model.get("max_completion_length", cfg.max_completion_length)
    cfg.temperature = model.get("temperature", cfg.temperature)
    cfg.top_p = model.get("top_p", cfg.top_p)

    lora = raw.get("lora", {})
    cfg.lora_r = lora.get("r", cfg.lora_r)
    cfg.lora_alpha = lora.get("alpha", cfg.lora_alpha)
    cfg.lora_dropout = lora.get("dropout", cfg.lora_dropout)
    cfg.lora_bias = lora.get("bias", cfg.lora_bias)
    cfg.lora_target_modules = lora.get("target_modules", cfg.lora_target_modules)

    data = raw.get("data", {})
    cfg.data_file = data.get("file", cfg.data_file)
    cfg.sft_file = data.get("sft_file", data.get("file", cfg.sft_file))
    cfg.rl_file = data.get("rl_file", data.get("file", cfg.rl_file))
    cfg.eval_file = data.get("eval_file", data.get("file", cfg.eval_file))
    cfg.max_samples = data.get("max_samples", cfg.max_samples)
    cfg.sft_samples = data.get("sft_samples", cfg.sft_samples)
    cfg.rl_samples = data.get("rl_samples", cfg.rl_samples)
    cfg.eval_samples = data.get("eval_samples", cfg.eval_samples)

    train = raw.get("training", {})
    cfg.sft_lr = train.get("sft_learning_rate", cfg.sft_lr)
    cfg.sft_steps = train.get("sft_steps", cfg.sft_steps)
    cfg.sft_batch_size = train.get("sft_batch_size", cfg.sft_batch_size)
    cfg.grpo_lr = train.get("grpo_learning_rate", cfg.grpo_lr)
    cfg.grpo_steps = train.get("grpo_steps", cfg.grpo_steps)
    cfg.group_size = train.get("group_size", cfg.group_size)
    cfg.num_generations = train.get("num_generations", cfg.num_generations)
    cfg.per_device_train_batch_size = train.get("per_device_train_batch_size", cfg.per_device_train_batch_size)
    if cfg.num_generations is None:
        cfg.num_generations = cfg.group_size
    if cfg.per_device_train_batch_size is None:
        cfg.per_device_train_batch_size = cfg.group_size
    cfg.grad_accum_steps = train.get("gradient_accumulation_steps", cfg.grad_accum_steps)
    cfg.steps_per_generation = train.get("steps_per_generation", cfg.steps_per_generation)
    cfg.epsilon_clip = train.get("epsilon_clip", cfg.epsilon_clip)
    cfg.epsilon_high = train.get("epsilon_high", cfg.epsilon_high)
    cfg.beta_kl = train.get("beta_kl", cfg.beta_kl)
    cfg.weight_decay = train.get("weight_decay", cfg.weight_decay)
    cfg.max_grad_norm = train.get("max_grad_norm", cfg.max_grad_norm)
    cfg.lr_scheduler_type = train.get("lr_scheduler_type", cfg.lr_scheduler_type)
    cfg.warmup_steps = train.get("warmup_steps", cfg.warmup_steps)
    cfg.loss_type = train.get("loss_type", cfg.loss_type)
    cfg.scale_rewards = train.get("scale_rewards", cfg.scale_rewards)
    cfg.mask_truncated_completions = train.get("mask_truncated_completions", cfg.mask_truncated_completions)
    cfg.top_entropy_quantile = train.get("top_entropy_quantile", cfg.top_entropy_quantile)
    cfg.drop_zero_advantage_groups = train.get("drop_zero_advantage_groups", cfg.drop_zero_advantage_groups)
    cfg.entropy_advantage_weighting = train.get("entropy_advantage_weighting", cfg.entropy_advantage_weighting)
    cfg.entropy_weight_power = train.get("entropy_weight_power", cfg.entropy_weight_power)
    cfg.entropy_weight_min = train.get("entropy_weight_min", cfg.entropy_weight_min)
    cfg.entropy_weight_max = train.get("entropy_weight_max", cfg.entropy_weight_max)
    cfg.seed = train.get("seed", cfg.seed)

    spad = raw.get("spad", {})
    cfg.answer_reward = spad.get("answer_reward", cfg.answer_reward)
    cfg.wrong_reward = spad.get("wrong_reward", cfg.wrong_reward)
    cfg.format_reward = spad.get("format_reward", cfg.format_reward)
    cfg.consistency_reward_weight = spad.get("consistency_reward_weight", cfg.consistency_reward_weight)
    cfg.repetition_penalty_weight = spad.get("repetition_penalty_weight", cfg.repetition_penalty_weight)
    cfg.length_penalty_weight = spad.get("length_penalty_weight", cfg.length_penalty_weight)
    cfg.process_tau = spad.get("process_tau", cfg.process_tau)
    cfg.positive_score_threshold = spad.get("positive_score_threshold", cfg.positive_score_threshold)
    cfg.negative_score_threshold = spad.get("negative_score_threshold", cfg.negative_score_threshold)
    cfg.min_step_weight = spad.get("min_step_weight", cfg.min_step_weight)
    cfg.consistency_weight = spad.get("consistency_weight", cfg.consistency_weight)
    cfg.virtual_reward = spad.get("virtual_reward", cfg.virtual_reward)
    cfg.zero_advantage_fix = spad.get("zero_advantage_fix", cfg.zero_advantage_fix)
    cfg.positive_length_tiebreak = spad.get("positive_length_tiebreak", cfg.positive_length_tiebreak)
    cfg.normalize_token_advantage = spad.get("normalize_token_advantage", cfg.normalize_token_advantage)
    cfg.max_token_advantage_scale = spad.get("max_token_advantage_scale", cfg.max_token_advantage_scale)

    log = raw.get("logging", {})
    cfg.eval_episodes = log.get("eval_episodes", cfg.eval_episodes)
    cfg.eval_pass_at_k = log.get("eval_pass_at_k", cfg.eval_pass_at_k)
    cfg.eval_batch_size = log.get("eval_batch_size", cfg.eval_batch_size)
    cfg.eval_sample_temperature = log.get("eval_sample_temperature", cfg.eval_sample_temperature)
    cfg.eval_sample_top_p = log.get("eval_sample_top_p", cfg.eval_sample_top_p)
    cfg.skip_final_eval = log.get("skip_final_eval", cfg.skip_final_eval)
    cfg.log_process_metrics = log.get("log_process_metrics", cfg.log_process_metrics)
    cfg.log_steps = log.get("logging_steps", cfg.log_steps)
    cfg.save_steps = log.get("save_steps", cfg.save_steps)
    cfg.output_dir = log.get("output_dir", cfg.output_dir)
    cfg.run_name = log.get("run_name", cfg.run_name)
    return cfg


def autocast_context():
    if torch.cuda.is_available():
        return torch.amp.autocast("cuda", dtype=_AUTOCAST_DTYPE)
    return contextlib.nullcontext()


def load_model(cfg: SPADConfig):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    torch_dtype = torch_dtype_from_name(cfg.dtype)
    set_autocast_dtype(cfg.dtype)
    kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
        "attn_implementation": "eager",
    }
    if cfg.load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **kwargs)
    if not cfg.load_in_4bit and torch.cuda.is_available():
        model = model.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if cfg.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=cfg.lora_target_modules,
        lora_dropout=cfg.lora_dropout,
        bias=cfg.lora_bias,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    if not cfg.load_in_4bit and torch.cuda.is_available():
        model = model.to("cuda")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] Loaded {cfg.model_name}. Trainable params: {trainable:,}")
    return model, tokenizer


def format_prompt(tokenizer, question: str, dataset_type: str = "gsm8k") -> str:
    if dataset_type == "agieval_mc":
        system = (
            "Solve the multiple-choice math problem step by step.\n"
            "Use concise, numbered steps.\n"
            "End with exactly one line: Final Answer: \\boxed{A}.\n"
            "Use only the correct option letter inside the box.\n"
            "Do not use tools or JSON."
        )
    elif dataset_type == "bbh_arithmetic":
        system = (
            "Evaluate the arithmetic expression step by step.\n"
            "Use concise, numbered steps.\n"
            "End with exactly one line: Final Answer: \\boxed{number}.\n"
            "Do not use tools or JSON."
        )
    else:
        system = (
            "Solve the math problem step by step.\n"
            "Use concise, numbered steps.\n"
            "End with exactly one line: Final Answer: \\boxed{answer}.\n"
            "Do not use tools or JSON."
        )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Question: {question}"},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_dataset(cfg: SPADConfig) -> Tuple[List[MathSample], List[MathSample], List[MathSample]]:
    if cfg.sft_file == cfg.rl_file == cfg.eval_file:
        samples = load_gsm8k_json(cfg.data_file, max_samples=cfg.max_samples)
        sft_end = min(cfg.sft_samples, len(samples))
        rl_end = min(sft_end + cfg.rl_samples, len(samples))
        eval_end = min(rl_end + cfg.eval_samples, len(samples))
        sft_samples = samples[:sft_end]
        rl_samples = samples[sft_end:rl_end]
        eval_samples = samples[rl_end:eval_end]
        if not eval_samples:
            eval_samples = samples[-min(cfg.eval_samples, len(samples)):]
        return sft_samples, rl_samples, eval_samples

    if cfg.sft_file == cfg.rl_file:
        train_samples = load_gsm8k_json(cfg.sft_file, max_samples=cfg.max_samples)
        sft_end = min(cfg.sft_samples, len(train_samples))
        rl_end = min(sft_end + cfg.rl_samples, len(train_samples))
        sft_samples = train_samples[:sft_end]
        rl_samples = train_samples[sft_end:rl_end]
    else:
        sft_samples = load_gsm8k_json(cfg.sft_file, max_samples=cfg.sft_samples)
        rl_samples = load_gsm8k_json(cfg.rl_file, max_samples=cfg.rl_samples)
    eval_samples = load_gsm8k_json(cfg.eval_file, max_samples=cfg.eval_samples)
    return sft_samples, rl_samples, eval_samples


def sft_warmup(model, tokenizer, train_samples: List[MathSample], cfg: SPADConfig):
    if cfg.sft_steps <= 0 or not train_samples:
        print("[SFT] Skipped.")
        return
    print("[SFT] Starting CoT warmup...")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.sft_lr, weight_decay=cfg.weight_decay)
    model.train()
    for step in range(cfg.sft_steps):
        total_loss = 0.0
        for _ in range(cfg.sft_batch_size):
            sample = random.choice(train_samples)
            prompt = format_prompt(tokenizer, sample.question, sample.dataset_type)
            completion = format_cot_solution(sample.raw_answer, sample.answer)
            prompt_ids = tokenizer(text=prompt, return_tensors="pt", add_special_tokens=False).input_ids
            comp_ids = tokenizer(text=completion, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=cfg.max_completion_length).input_ids
            prompt_budget = max(1, cfg.max_seq_length - comp_ids.shape[1])
            if prompt_ids.shape[1] > prompt_budget:
                prompt_ids = prompt_ids[:, -prompt_budget:]
            input_ids = torch.cat([prompt_ids, comp_ids], dim=1)
            labels = input_ids.clone()
            labels[:, :prompt_ids.shape[1]] = -100
            input_ids = input_ids.to(model.device)
            labels = labels.to(model.device)
            attn_mask = torch.ones_like(input_ids)
            with autocast_context():
                loss = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels).loss
            (loss / cfg.sft_batch_size).backward()
            total_loss += float(loss.detach().item())
        opt.step()
        opt.zero_grad()
        if step % 10 == 0:
            print(f"[SFT] step={step}/{cfg.sft_steps} loss={total_loss/cfg.sft_batch_size:.4f}")
    print("[SFT] Done.")


def rollout_group(model, tokenizer, sample: MathSample, cfg: SPADConfig, mode: str) -> List[Trajectory]:
    prompt = format_prompt(tokenizer, sample.question, sample.dataset_type)
    prompt_ids = tokenizer(text=prompt, return_tensors="pt", truncation=True, max_length=cfg.max_seq_length).input_ids.to(model.device)
    group: List[Trajectory] = []
    model.eval()
    for _ in range(cfg.group_size):
        with torch.no_grad():
            output = model.generate(
                prompt_ids,
                max_new_tokens=cfg.max_completion_length,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        completion_ids = output[0, prompt_ids.shape[1]:]
        completion = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        group.append(score_completion(tokenizer, prompt, completion, sample, cfg, mode))
    model.train()
    return group


def score_completion(tokenizer, prompt: str, completion: str, sample: MathSample, cfg: SPADConfig, mode: str) -> Trajectory:
    final_answer = extract_final_answer(completion)
    correct = verify_answer(final_answer, sample.answer, dataset_type=sample.dataset_type)
    reward = cfg.answer_reward if correct else cfg.wrong_reward
    if "final answer" in completion.lower() or "\\boxed" in completion:
        reward += cfg.format_reward
    analysis = analyze_process(sample.question, completion, sample.answer, sample.dataset_type)
    metrics = process_metrics_dict(analysis)
    token_count = len(tokenizer(text=completion, add_special_tokens=False).input_ids)
    adjusted_reward = reward
    if mode != "baseline":
        adjusted_reward -= cfg.length_penalty_weight * max(0, token_count - cfg.max_completion_length * 0.75)
    if mode == "c_spad":
        adjusted_reward += cfg.consistency_reward_weight * analysis.step_consistency_rate
        adjusted_reward += cfg.consistency_reward_weight * analysis.final_process_consistency
        adjusted_reward -= cfg.repetition_penalty_weight * analysis.repetition_rate
    process_scores = [
        s.score + (cfg.consistency_weight * s.consistency if mode == "c_spad" else 0.0)
        for s in analysis.steps
    ]
    return Trajectory(
        prompt=prompt,
        completion=completion,
        reward=reward,
        adjusted_reward=adjusted_reward,
        answer_correct=correct,
        final_answer=final_answer,
        token_count=token_count,
        process_metrics=metrics,
        process_scores=process_scores,
    )


def compute_group_advantages(group: List[Trajectory], cfg: SPADConfig, mode: str):
    rewards = np.array([t.adjusted_reward for t in group], dtype=np.float32)
    std = float(rewards.std())
    if std < 1e-6 and cfg.zero_advantage_fix:
        if np.all(rewards <= 0):
            mean = float(np.mean(np.append(rewards, cfg.virtual_reward)))
            advantages = rewards - mean
        elif np.all(rewards > 0) and cfg.positive_length_tiebreak:
            lengths = np.array([t.token_count for t in group], dtype=np.float32)
            len_std = float(lengths.std())
            if len_std < 1e-6:
                advantages = np.zeros_like(rewards)
            else:
                advantages = (lengths.mean() - lengths) / (len_std + 1e-6)
        else:
            advantages = np.zeros_like(rewards)
    else:
        advantages = (rewards - rewards.mean()) / (std + 1e-6)

    for traj, adv in zip(group, advantages):
        if mode == "baseline":
            traj.step_advantages = [float(adv)]
        else:
            traj.step_advantages = decompose_advantage(float(adv), traj.process_scores, cfg)


def decompose_advantage(advantage: float, scores: List[float], cfg: SPADConfig) -> List[float]:
    if not scores:
        return [advantage]
    arr = np.array(scores, dtype=np.float32)
    tau = max(cfg.process_tau, 1e-3)

    if advantage > 0:
        gated = np.maximum(arr - cfg.positive_score_threshold, 0.0)
        if gated.sum() <= 1e-8:
            gated = np.zeros_like(arr)
            gated[int(arr.argmax())] = 1.0
        logits = np.log(gated + 1e-8) / tau
    elif advantage < 0:
        gated = np.maximum(cfg.negative_score_threshold - arr, 0.0)
        if gated.sum() <= 1e-8:
            gated = np.zeros_like(arr)
            gated[int(arr.argmin())] = 1.0
        logits = np.log(gated + 1e-8) / tau
    else:
        return [0.0 for _ in scores]

    logits = logits - logits.max()
    weights = np.exp(logits)
    if cfg.min_step_weight > 0 and len(weights) > 1:
        weights = weights + cfg.min_step_weight
    weights = weights / (weights.sum() + 1e-10)
    return [float(advantage * w) for w in weights]


def _swap_lora_weights(model, target_state: Dict[str, torch.Tensor]):
    prev = {}
    for name, param in model.named_parameters():
        if "lora" in name and name in target_state:
            prev[name] = param.data.clone()
            param.data.copy_(target_state[name])
    return prev


def _restore_lora_weights(model, saved_state: Dict[str, torch.Tensor]):
    for name, param in model.named_parameters():
        if "lora" in name and name in saved_state:
            param.data.copy_(saved_state[name])


def tokenize_step_advantages(tokenizer, completion: str, step_advantages: List[float], max_completion_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
    comp_ids = tokenizer(text=completion, add_special_tokens=False, truncation=True, max_length=max_completion_length).input_ids
    if not comp_ids:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.float32)
    if len(step_advantages) == 1:
        return torch.tensor(comp_ids, dtype=torch.long), torch.full((len(comp_ids),), step_advantages[0], dtype=torch.float32)

    steps = split_steps(completion)
    advs: List[float] = []
    for step, adv in zip(steps, step_advantages):
        n = len(tokenizer(text=step, add_special_tokens=False).input_ids)
        advs.extend([adv] * max(n, 1))
    if len(advs) < len(comp_ids):
        advs.extend([step_advantages[-1]] * (len(comp_ids) - len(advs)))
    advs = advs[:len(comp_ids)]
    return torch.tensor(comp_ids, dtype=torch.long), torch.tensor(advs, dtype=torch.float32)


def compute_grpo_loss(model, tokenizer, group: List[Trajectory], ema_state: Dict[str, torch.Tensor], cfg: SPADConfig):
    device = model.device
    total_loss_value = 0.0
    count = 0
    for traj in group:
        ctx_ids = tokenizer(text=traj.prompt, return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)
        if ctx_ids.shape[0] > cfg.max_seq_length:
            ctx_ids = ctx_ids[-cfg.max_seq_length:]
        comp_ids, advs = tokenize_step_advantages(tokenizer, traj.completion, traj.step_advantages, cfg.max_completion_length)
        if comp_ids.numel() == 0:
            continue
        input_ids = torch.cat([ctx_ids, comp_ids], dim=0).unsqueeze(0).to(device)
        attn_mask = torch.ones_like(input_ids)
        comp_len = comp_ids.shape[0]
        adv_tensor = advs.to(device)
        if input_ids.shape[1] > cfg.max_seq_length + cfg.max_completion_length:
            overflow = input_ids.shape[1] - (cfg.max_seq_length + cfg.max_completion_length)
            input_ids = input_ids[:, overflow:]
            attn_mask = attn_mask[:, overflow:]

        with autocast_context():
            logits = model(input_ids=input_ids, attention_mask=attn_mask).logits
        log_probs = torch.log_softmax(logits[0, -comp_len-1:-1, :], dim=-1)
        target_ids = input_ids[0, -comp_len:].unsqueeze(-1)
        token_lp = torch.gather(log_probs, dim=1, index=target_ids).squeeze(-1)
        del logits, log_probs

        saved = _swap_lora_weights(model, ema_state)
        with torch.no_grad():
            with autocast_context():
                ref_logits = model(input_ids=input_ids, attention_mask=attn_mask).logits
        _restore_lora_weights(model, saved)
        ref_log_probs = torch.log_softmax(ref_logits[0, -comp_len-1:-1, :], dim=-1)
        ref_token_lp = torch.gather(ref_log_probs, dim=1, index=target_ids).squeeze(-1)
        del ref_logits, ref_log_probs

        ratios = torch.exp(token_lp - ref_token_lp)
        clipped = torch.clamp(ratios, 1 - cfg.epsilon_clip, 1 + cfg.epsilon_clip)
        pg_loss = -torch.min(ratios * adv_tensor, clipped * adv_tensor)
        kl_loss = cfg.beta_kl * (token_lp - ref_token_lp)
        loss = (pg_loss + kl_loss).mean()
        (loss / cfg.grad_accum_steps).backward()
        total_loss_value += float(loss.detach().item())
        count += 1
    if count == 0:
        return 0.0
    return total_loss_value / count


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> List[float]:
    if total <= 0:
        return [0.0, 0.0]
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denom
    return [max(0.0, center - margin), min(1.0, center + margin)]


def has_final_answer_format(text: str) -> bool:
    return bool(re.search(r"(?im)^\s*Final Answer\s*:\s*\\boxed\{[^{}\n]+\}\s*\.?\s*$", str(text)))


def mean_interval(values: List[float], z: float = 1.959963984540054) -> List[float]:
    if not values:
        return [0.0, 0.0]
    if len(values) == 1:
        return [max(0.0, values[0]), min(1.0, values[0])]
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    stderr = float(arr.std(ddof=1) / math.sqrt(len(arr)))
    return [max(0.0, mean - z * stderr), min(1.0, mean + z * stderr)]


def majority_answer(answers: List[str]) -> Tuple[str, int, int, float]:
    valid_answers = [normalize_answer(answer) for answer in answers if normalize_answer(answer)]
    if not valid_answers:
        return "", 0, 0, 0.0
    counts: Dict[str, int] = {}
    first_seen: Dict[str, int] = {}
    for idx, answer in enumerate(valid_answers):
        counts[answer] = counts.get(answer, 0) + 1
        first_seen.setdefault(answer, idx)
    winner = max(counts, key=lambda answer: (counts[answer], -first_seen[answer]))
    return winner, counts[winner], len(counts), counts[winner] / len(valid_answers)


def _decode_completion(model, tokenizer, prompt_ids: torch.Tensor, cfg: SPADConfig, *, sample: bool) -> Dict[str, Any]:
    generation_kwargs: Dict[str, Any] = {
        "max_new_tokens": cfg.max_completion_length,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if sample:
        generation_kwargs.update(
            {
                "do_sample": True,
                "temperature": cfg.eval_sample_temperature,
                "top_p": cfg.eval_sample_top_p,
            }
        )
    else:
        generation_kwargs.update({"do_sample": False})
    with torch.inference_mode():
        with autocast_context():
            output = model.generate(prompt_ids, **generation_kwargs)
    completion_ids = output[0, prompt_ids.shape[1]:]
    return {
        "completion": tokenizer.decode(completion_ids, skip_special_tokens=True).strip(),
        "generated_tokens": int(completion_ids.numel()),
        "truncated": bool(completion_ids.numel() >= cfg.max_completion_length),
    }


def _decode_sampled_completions(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    cfg: SPADConfig,
    num_return_sequences: int,
) -> List[Dict[str, Any]]:
    if num_return_sequences <= 0:
        return []
    generation_kwargs: Dict[str, Any] = {
        "max_new_tokens": cfg.max_completion_length,
        "do_sample": True,
        "temperature": cfg.eval_sample_temperature,
        "top_p": cfg.eval_sample_top_p,
        "num_return_sequences": num_return_sequences,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    with torch.inference_mode():
        with autocast_context():
            output = model.generate(prompt_ids, **generation_kwargs)
    decoded = []
    for row in output:
        completion_ids = row[prompt_ids.shape[1]:]
        decoded.append(
            {
                "completion": tokenizer.decode(completion_ids, skip_special_tokens=True).strip(),
                "generated_tokens": int(completion_ids.numel()),
                "truncated": bool(completion_ids.numel() >= cfg.max_completion_length),
            }
        )
    return decoded


def _prepare_generation_inputs(inputs: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    allowed_keys = {"input_ids", "attention_mask", "token_type_ids"}
    return {k: v.to(device) for k, v in inputs.items() if k in allowed_keys}


def _decode_batch_outputs(
    model,
    tokenizer,
    prompts: List[str],
    cfg: SPADConfig,
    *,
    sample: bool,
    num_return_sequences: int = 1,
) -> List[List[Dict[str, Any]]]:
    if not prompts:
        return []
    generation_kwargs: Dict[str, Any] = {
        "max_new_tokens": cfg.max_completion_length,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if sample:
        generation_kwargs.update(
            {
                "do_sample": True,
                "temperature": cfg.eval_sample_temperature,
                "top_p": cfg.eval_sample_top_p,
                "num_return_sequences": num_return_sequences,
            }
        )
    else:
        generation_kwargs.update({"do_sample": False})

    old_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    try:
        inputs = tokenizer(
            text=prompts,
            return_tensors="pt",
            truncation=True,
            max_length=cfg.max_seq_length,
            padding=True,
            add_special_tokens=False,
        )
    finally:
        tokenizer.padding_side = old_padding_side
    inputs = _prepare_generation_inputs(inputs, model.device)
    input_width = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        with autocast_context():
            output = model.generate(**inputs, **generation_kwargs)

    grouped: List[List[Dict[str, Any]]] = [[] for _ in prompts]
    for output_idx, row in enumerate(output):
        prompt_idx = output_idx // num_return_sequences if sample else output_idx
        completion_ids = row[input_width:]
        grouped[prompt_idx].append(
            {
                "completion": tokenizer.decode(completion_ids, skip_special_tokens=True).strip(),
                "generated_tokens": int(completion_ids.numel()),
                "truncated": bool(completion_ids.numel() >= cfg.max_completion_length),
            }
        )
    return grouped


def quick_eval(
    model,
    tokenizer,
    eval_samples: List[MathSample],
    cfg: SPADConfig,
    episodes: int,
    details_path: Optional[str | Path] = None,
):
    model.eval()
    n = len(eval_samples) if episodes <= 0 else min(episodes, len(eval_samples))
    avg1_correct = 0
    avgk_correct = 0
    majority_correct = 0
    greedy_format_correct = 0
    sampled_format_correct = 0
    greedy_extracted = 0
    sampled_extracted = 0
    greedy_truncated = 0
    sampled_truncated = 0
    greedy_token_counts = []
    sampled_token_counts = []
    avgk_question_scores: List[float] = []
    majority_question_scores: List[float] = []
    majority_agreement_scores: List[float] = []
    unique_answer_counts: List[int] = []
    metric_sums: Dict[str, float] = {}
    by_type: Dict[str, Dict[str, Any]] = {}
    avg_k = max(1, int(cfg.eval_pass_at_k))
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    details_file = None
    if details_path is not None:
        details_path = Path(details_path)
        details_path.parent.mkdir(parents=True, exist_ok=True)
        details_file = open(details_path, "w", encoding="utf-8")
    selected_samples = eval_samples[:n]
    eval_batch_size = max(1, int(getattr(cfg, "eval_batch_size", 1)))
    eval_iter = tqdm(
        range(0, n, eval_batch_size),
        total=math.ceil(n / eval_batch_size) if n else 0,
        desc=f"[EVAL avg@1/avg@{avg_k}]",
        dynamic_ncols=True,
    )
    eval_idx = 0
    try:
        for batch_start in eval_iter:
            batch_samples = selected_samples[batch_start : batch_start + eval_batch_size]
            batch_prompts = [format_prompt(tokenizer, sample.question, sample.dataset_type) for sample in batch_samples]
            greedy_outputs = _decode_batch_outputs(model, tokenizer, batch_prompts, cfg, sample=False, num_return_sequences=1)
            sampled_outputs = _decode_batch_outputs(
                model,
                tokenizer,
                batch_prompts,
                cfg,
                sample=True,
                num_return_sequences=avg_k,
            )
            for sample, greedy_group, sample_outputs in zip(batch_samples, greedy_outputs, sampled_outputs):
                eval_idx += 1
                greedy_output = greedy_group[0]
                completion = greedy_output["completion"]
                final_answer = extract_final_answer(completion)
                is_correct = int(verify_answer(final_answer, sample.answer, dataset_type=sample.dataset_type))
                avg1_correct += is_correct
                greedy_extracted += int(bool(final_answer))
                greedy_format_ok = has_final_answer_format(completion)
                greedy_format_correct += int(greedy_format_ok)
                greedy_truncated += int(greedy_output["truncated"])
                token_count = int(greedy_output["generated_tokens"])
                greedy_token_counts.append(token_count)
                analysis = analyze_process(sample.question, completion, sample.answer, sample.dataset_type)
                sampled = []
                sample_answers = []
                sample_correct_count = 0
                for sample_idx, sample_output in enumerate(sample_outputs):
                    sample_completion = sample_output["completion"]
                    sample_answer = extract_final_answer(sample_completion)
                    sample_is_correct = verify_answer(sample_answer, sample.answer, dataset_type=sample.dataset_type)
                    sample_tokens = int(sample_output["generated_tokens"])
                    sample_correct_count += int(sample_is_correct)
                    sample_answers.append(sample_answer)
                    sampled_extracted += int(bool(sample_answer))
                    format_ok = has_final_answer_format(sample_completion)
                    sampled_format_correct += int(format_ok)
                    sampled_truncated += int(sample_output["truncated"])
                    sampled_token_counts.append(sample_tokens)
                    sampled.append(
                        {
                            "sample_index": sample_idx,
                            "correct": bool(sample_is_correct),
                            "extracted_answer": sample_answer,
                            "token_count": sample_tokens,
                            "format_ok": format_ok,
                            "truncated": bool(sample_output["truncated"]),
                            "completion": sample_completion,
                        }
                    )
                avgk_correct += sample_correct_count
                avgk_score = sample_correct_count / max(avg_k, 1)
                avgk_question_scores.append(avgk_score)
                majority, majority_count, unique_answers, majority_agreement = majority_answer(sample_answers)
                majority_is_correct = int(bool(majority) and verify_answer(majority, sample.answer, dataset_type=sample.dataset_type))
                majority_correct += majority_is_correct
                majority_question_scores.append(float(majority_is_correct))
                majority_agreement_scores.append(float(majority_agreement))
                unique_answer_counts.append(unique_answers)
                bucket = by_type.setdefault(
                    sample.dataset_type,
                    {
                        "n": 0,
                        "avg1_correct": 0,
                        "avgk_correct": 0,
                        "majority_correct": 0,
                        "avgk_scores": [],
                        "majority_scores": [],
                        "majority_agreement_scores": [],
                        "unique_answer_counts": [],
                        "greedy_tokens": [],
                        "sampled_tokens": [],
                        "greedy_format_correct": 0,
                        "sampled_format_correct": 0,
                        "greedy_extracted": 0,
                        "sampled_extracted": 0,
                        "greedy_truncated": 0,
                        "sampled_truncated": 0,
                        "metrics": {},
                    },
                )
                bucket["n"] += 1
                bucket["avg1_correct"] += is_correct
                bucket["avgk_correct"] += sample_correct_count
                bucket["majority_correct"] += majority_is_correct
                bucket["avgk_scores"].append(avgk_score)
                bucket["majority_scores"].append(float(majority_is_correct))
                bucket["majority_agreement_scores"].append(float(majority_agreement))
                bucket["unique_answer_counts"].append(unique_answers)
                bucket["greedy_tokens"].append(token_count)
                bucket["sampled_tokens"].extend([row["token_count"] for row in sampled])
                bucket["greedy_format_correct"] += int(greedy_format_ok)
                bucket["sampled_format_correct"] += sum(int(row["format_ok"]) for row in sampled)
                bucket["greedy_extracted"] += int(bool(final_answer))
                bucket["sampled_extracted"] += sum(int(bool(row["extracted_answer"])) for row in sampled)
                bucket["greedy_truncated"] += int(greedy_output["truncated"])
                bucket["sampled_truncated"] += sum(int(row["truncated"]) for row in sampled)
                process_metrics = process_metrics_dict(analysis)
                for k, v in process_metrics.items():
                    metric_sums[k] = metric_sums.get(k, 0.0) + v
                    bucket["metrics"][k] = bucket["metrics"].get(k, 0.0) + v
                if details_file is not None:
                    details_row = {
                        "source": sample.source,
                        "source_id": sample.source_id,
                        "dataset_type": sample.dataset_type,
                        "question": sample.question,
                        "gold_answer": sample.answer,
                        "avg1_correct": bool(is_correct),
                        "avgk_correct_count": sample_correct_count,
                        "avgk_score": avgk_score,
                        "avg_k": avg_k,
                        "greedy_extracted_answer": final_answer,
                        "greedy_completion": completion,
                        "greedy_token_count": token_count,
                        "greedy_format_ok": greedy_format_ok,
                        "greedy_truncated": bool(greedy_output["truncated"]),
                        "majority_answer": majority,
                        "majority_count": majority_count,
                        "majority_correct": bool(majority_is_correct),
                        "majority_agreement": majority_agreement,
                        "unique_answer_count": unique_answers,
                        "samples": sampled,
                        "process_metrics": process_metrics,
                    }
                    details_file.write(json.dumps(details_row, ensure_ascii=False) + "\n")
                eval_iter.set_postfix(
                    {
                        "avg@1": f"{avg1_correct / eval_idx:.3f}",
                        f"avg@{avg_k}": f"{avgk_correct / max(eval_idx * avg_k, 1):.3f}",
                        f"maj@{avg_k}": f"{majority_correct / eval_idx:.3f}",
                        "tok@1": f"{float(np.mean(greedy_token_counts)):.1f}",
                        f"tok@{avg_k}": f"{float(np.mean(sampled_token_counts)):.1f}" if sampled_token_counts else "0.0",
                        "gens": eval_idx * (avg_k + 1),
                    }
                )
    finally:
        if details_file is not None:
            details_file.close()
    model.train()
    tokens_at_1 = float(np.mean(greedy_token_counts)) if greedy_token_counts else 0.0
    tokens_at_k = float(np.mean(sampled_token_counts)) if sampled_token_counts else 0.0
    result = {
        "eval_n": n,
        "avg@1": avg1_correct / max(n, 1),
        "avg@1_correct": avg1_correct,
        "avg@1_total_samples": n,
        "avg@1_ci95": wilson_interval(avg1_correct, n),
        f"avg@{avg_k}": avgk_correct / max(n * avg_k, 1),
        f"avg@{avg_k}_correct_samples": avgk_correct,
        f"avg@{avg_k}_total_samples": n * avg_k,
        f"avg@{avg_k}_ci95": mean_interval(avgk_question_scores),
        f"majority@{avg_k}": majority_correct / max(n, 1),
        f"majority@{avg_k}_ci95": wilson_interval(majority_correct, n),
        f"majority_agreement@{avg_k}": float(np.mean(majority_agreement_scores)) if majority_agreement_scores else 0.0,
        f"unique_answers@{avg_k}": float(np.mean(unique_answer_counts)) if unique_answer_counts else 0.0,
        "avg_k": avg_k,
        "sample_temperature": cfg.eval_sample_temperature,
        "sample_top_p": cfg.eval_sample_top_p,
        "tokens@1": tokens_at_1,
        f"tokens@{avg_k}": tokens_at_k,
        "greedy_avg_tokens": tokens_at_1,
        f"sampled_avg@{avg_k}_tokens": tokens_at_k,
        "greedy_answer_extraction_rate": greedy_extracted / max(n, 1),
        f"sampled_answer_extraction_rate@{avg_k}": sampled_extracted / max(n * avg_k, 1),
        "greedy_format_accuracy": greedy_format_correct / max(n, 1),
        f"sampled_format_accuracy@{avg_k}": sampled_format_correct / max(n * avg_k, 1),
        "greedy_truncation_rate": greedy_truncated / max(n, 1),
        f"sampled_truncation_rate@{avg_k}": sampled_truncated / max(n * avg_k, 1),
    }
    for k, v in metric_sums.items():
        result[k] = v / max(n, 1)
    result["by_dataset_type"] = {
        dtype: {
            "n": bucket["n"],
            "avg@1": bucket["avg1_correct"] / max(bucket["n"], 1),
            "avg@1_ci95": wilson_interval(bucket["avg1_correct"], bucket["n"]),
            f"avg@{avg_k}": bucket["avgk_correct"] / max(bucket["n"] * avg_k, 1),
            f"avg@{avg_k}_ci95": mean_interval(bucket["avgk_scores"]),
            f"majority@{avg_k}": bucket["majority_correct"] / max(bucket["n"], 1),
            f"majority@{avg_k}_ci95": wilson_interval(bucket["majority_correct"], bucket["n"]),
            f"majority_agreement@{avg_k}": float(np.mean(bucket["majority_agreement_scores"])) if bucket["majority_agreement_scores"] else 0.0,
            f"unique_answers@{avg_k}": float(np.mean(bucket["unique_answer_counts"])) if bucket["unique_answer_counts"] else 0.0,
            "tokens@1": float(np.mean(bucket["greedy_tokens"])) if bucket["greedy_tokens"] else 0.0,
            f"tokens@{avg_k}": float(np.mean(bucket["sampled_tokens"])) if bucket["sampled_tokens"] else 0.0,
            "greedy_avg_tokens": float(np.mean(bucket["greedy_tokens"])) if bucket["greedy_tokens"] else 0.0,
            f"sampled_avg@{avg_k}_tokens": float(np.mean(bucket["sampled_tokens"])) if bucket["sampled_tokens"] else 0.0,
            "greedy_answer_extraction_rate": bucket["greedy_extracted"] / max(bucket["n"], 1),
            f"sampled_answer_extraction_rate@{avg_k}": bucket["sampled_extracted"] / max(bucket["n"] * avg_k, 1),
            "greedy_format_accuracy": bucket["greedy_format_correct"] / max(bucket["n"], 1),
            f"sampled_format_accuracy@{avg_k}": bucket["sampled_format_correct"] / max(bucket["n"] * avg_k, 1),
            "greedy_truncation_rate": bucket["greedy_truncated"] / max(bucket["n"], 1),
            f"sampled_truncation_rate@{avg_k}": bucket["sampled_truncated"] / max(bucket["n"] * avg_k, 1),
            **{k: v / max(bucket["n"], 1) for k, v in bucket["metrics"].items()},
        }
        for dtype, bucket in by_type.items()
    }
    if details_path is not None:
        result["details_path"] = str(details_path)
    return result


def train(cfg: SPADConfig, mode: str):
    os.makedirs(cfg.output_dir, exist_ok=True)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    sft_samples, rl_samples, eval_samples = build_dataset(cfg)
    print(
        f"[DATA] sft={len(sft_samples)} rl={len(rl_samples)} eval={len(eval_samples)} "
        f"files=({cfg.sft_file}, {cfg.rl_file}, {cfg.eval_file})"
    )
    model, tokenizer = load_model(cfg)
    sft_warmup(model, tokenizer, sft_samples, cfg)

    ema_state: Dict[str, torch.Tensor] = {
        name: param.data.clone()
        for name, param in model.named_parameters()
        if "lora" in name
    }
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.grpo_lr, weight_decay=cfg.weight_decay)
    metrics: Dict[str, List[Any]] = {"step": [], "loss": [], "reward": [], "adjusted_reward": [], "success_rate": [], "tokens": []}

    pbar = tqdm(range(cfg.grpo_steps), desc=mode)
    for step in pbar:
        sample = random.choice(rl_samples)
        group = rollout_group(model, tokenizer, sample, cfg, mode)
        compute_group_advantages(group, cfg, mode)
        loss_value = compute_grpo_loss(model, tokenizer, group, ema_state, cfg)

        if (step + 1) % cfg.grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            for name in ema_state:
                param = dict(model.named_parameters())[name]
                ema_state[name] = 0.999 * ema_state[name] + 0.001 * param.data.clone()

        rewards = [t.reward for t in group]
        adjusted = [t.adjusted_reward for t in group]
        tokens = [t.token_count for t in group]
        sr = sum(t.answer_correct for t in group) / len(group)
        metrics["step"].append(step)
        metrics["loss"].append(loss_value)
        metrics["reward"].append(float(np.mean(rewards)))
        metrics["adjusted_reward"].append(float(np.mean(adjusted)))
        metrics["success_rate"].append(sr)
        metrics["tokens"].append(float(np.mean(tokens)))
        if step % cfg.log_steps == 0:
            pbar.set_postfix({
                "loss": f"{loss_value:.3f}",
                "reward": f"{np.mean(adjusted):.2f}",
                "sr": f"{sr:.2f}",
                "tok": f"{np.mean(tokens):.0f}",
            })
        if step > 0 and step % cfg.save_steps == 0:
            save_checkpoint(model, tokenizer, cfg, mode, step)

    eval_result = quick_eval(
        model,
        tokenizer,
        eval_samples,
        cfg,
        cfg.eval_episodes,
        Path(cfg.output_dir) / f"eval_details_{mode}.jsonl",
    )
    metrics["final_eval"] = [eval_result]
    final_path = save_checkpoint(model, tokenizer, cfg, mode, "final")
    with open(Path(cfg.output_dir) / f"metrics_{mode}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"[EVAL] {eval_result}")
    print(f"[DONE] saved to {final_path}")


def save_checkpoint(model, tokenizer, cfg: SPADConfig, mode: str, step: int | str) -> str:
    path = Path(cfg.output_dir) / f"{mode}_checkpoint_{step}"
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return str(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_spad.yaml")
    parser.add_argument("--mode", choices=["baseline", "spad", "c_spad"], default="spad")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg, args.mode)


if __name__ == "__main__":
    main()
