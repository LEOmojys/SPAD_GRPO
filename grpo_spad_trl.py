"""
TRL-based reproduction of baseline / SPAD / C-SPAD.

This keeps TRL's GRPOTrainer generation, sampling, KL, buffering, logging and
checkpointing flow, while replacing sequence-level advantage broadcasting with
SPAD's quality-gated token-level step advantage.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from accelerate.utils import gather, gather_object
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# TRL 0.24 imports llm_blender through optional judge callbacks. The
# llm_blender release still expects this legacy transformers constant, while
# newer transformers versions removed it. We do not use llm_blender, but TRL's
# import path still needs the symbol to exist.
import transformers.utils.hub as transformers_hub

if not hasattr(transformers_hub, "TRANSFORMERS_CACHE"):
    transformers_hub.TRANSFORMERS_CACHE = os.environ.get(
        "TRANSFORMERS_CACHE",
        os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface" / "transformers")),
    )

import trl.import_utils as trl_import_utils

# We do not use TRL's optional model-merge callback. Some environments expose a
# broken/partial mergekit, llm_blender or weave install, which makes
# GRPOTrainer fail during import.
trl_import_utils._mergekit_available = False
trl_import_utils._llm_blender_available = False
trl_import_utils._weave_available = False

from trl import GRPOConfig, GRPOTrainer
from trl.data_utils import is_conversational
from trl.trainer.utils import nanmax, nanmin, pad

from grpo_spad import (
    SPADConfig,
    autocast_context,
    build_dataset,
    decompose_advantage,
    format_prompt,
    load_config,
    load_model,
    quick_eval,
    sft_warmup,
    torch_dtype_from_name,
)
from process_signals import (
    analyze_process,
    extract_final_answer,
    process_metrics_dict,
    split_steps,
    verify_answer,
)


def completion_to_text(completion: Any) -> str:
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        return str(completion[0].get("content", ""))
    return str(completion)


def make_answer_reward_func(cfg: SPADConfig):
    def configured_answer_reward_func(completions, answer, **kwargs):
        rewards = []
        dataset_types = kwargs.get("dataset_type") or ["gsm8k"] * len(completions)
        for completion, gold, dataset_type in zip(completions, answer, dataset_types):
            pred = extract_final_answer(completion_to_text(completion))
            rewards.append(
                cfg.answer_reward
                if verify_answer(pred, gold, dataset_type=dataset_type)
                else cfg.wrong_reward
            )
        return rewards

    configured_answer_reward_func.__name__ = "answer_reward_func"
    return configured_answer_reward_func


def make_format_reward_func(cfg: SPADConfig):
    def configured_format_reward_func(completions, **kwargs):
        rewards = []
        for completion in completions:
            text = completion_to_text(completion).lower()
            rewards.append(cfg.format_reward if ("final answer" in text or "\\boxed" in text) else 0.0)
        return rewards

    configured_format_reward_func.__name__ = "format_reward_func"
    return configured_format_reward_func


class SPADGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, spad_cfg: SPADConfig, mode: str = "spad", **kwargs):
        super().__init__(*args, **kwargs)
        self.spad_cfg = spad_cfg
        self.spad_mode = mode

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        prompts = [x["prompt"] for x in inputs]
        images = None

        (
            prompt_ids_list,
            completion_ids_list,
            num_items_in_batch,
            sampling_per_token_logps_list,
            forward_kwargs,
        ) = self._generate_with_autocast(prompts, images)

        prompt_ids = [torch.tensor(ids, device=device) for ids in prompt_ids_list]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(prompt_ids, padding_value=self.pad_token_id, padding_side="left")
        prompt_mask = pad(prompt_mask, padding_value=0, padding_side="left")

        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids_list]
        completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right")
        completion_mask = pad(completion_mask, padding_value=0, padding_side="right")

        if sampling_per_token_logps_list is not None:
            sampling_per_token_logps = [torch.tensor(logps, device=device) for logps in sampling_per_token_logps_list]
            sampling_per_token_logps = pad(sampling_per_token_logps, padding_value=0.0, padding_side="right")
        else:
            sampling_per_token_logps = None

        if self.mask_truncated_completions:
            eos_and_pad = [self.eos_token_id, self.pad_token_id]
            is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device)
            completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()

        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        with torch.no_grad():
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self.args.gradient_accumulation_steps % generate_every != 0 or (
                self.use_vllm and self.vllm_importance_sampling_correction
            ):
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                    **forward_kwargs,
                )
            else:
                old_per_token_logps = None

            if self.use_vllm and self.vllm_importance_sampling_correction:
                importance_sampling_ratio = torch.exp(old_per_token_logps - sampling_per_token_logps)
                importance_sampling_ratio = torch.clamp(
                    importance_sampling_ratio, max=self.vllm_importance_sampling_cap
                )

            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=batch_size,
                        **forward_kwargs,
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                            self.model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size=batch_size,
                            **forward_kwargs,
                        )
            else:
                ref_per_token_logps = None

        prompts_text = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        rewards_per_func = self._calculate_rewards(inputs, prompts, completions, completion_ids_list)
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        local_adjustments, local_process_scores = self._local_process_adjustments(inputs, completions_text)
        process_adjustments = gather(local_adjustments.to(device))
        rewards = rewards + process_adjustments

        local_lengths = torch.tensor([len(ids) for ids in completion_ids_list], dtype=torch.float32, device=device)
        all_lengths = gather(local_lengths)
        advantages = self._compute_spad_sequence_advantages(rewards, all_lengths)

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = advantages.clone()
        advantages = advantages[process_slice]

        active_sequence_mask = torch.ones_like(advantages, dtype=torch.bool)
        if self.spad_cfg.drop_zero_advantage_groups and mode == "train":
            grouped_advantages = all_process_advantages.view(-1, self.num_generations)
            active_groups = grouped_advantages.abs().sum(dim=1) > 1e-8
            active_sequences = active_groups.repeat_interleave(self.num_generations)
            active_sequence_mask = active_sequences[process_slice].to(device=device)
            completion_mask = completion_mask * active_sequence_mask.unsqueeze(1).int()

        token_advantages = self._build_token_advantages(
            completions_text=completions_text,
            completion_ids_list=completion_ids_list,
            sequence_advantages=advantages.detach().cpu().tolist(),
            process_scores=local_process_scores,
            max_len=completion_ids.size(1),
            device=device,
        )

        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(rewards.view(-1, self.num_generations).std(dim=1).mean().item())
        self._metrics[mode]["spad/effective_token_ratio"].append(
            ((token_advantages.abs() > 1e-8) * completion_mask.bool()).sum().float().div(
                completion_mask.sum().clamp(min=1)
            ).item()
        )
        if self.spad_cfg.drop_zero_advantage_groups and mode == "train":
            self._metrics[mode]["dapo/active_group_ratio"].append(
                active_sequence_mask.float().mean().item()
            )

        active_token_count = completion_mask.sum().to(torch.float32)
        num_items_in_batch = gather(active_token_count).sum().clamp(min=1.0)

        if self.log_completions:
            self._logs["prompt"].extend(gather_object(prompts_text))
            self._logs["completion"].extend(gather_object(completions_text))
            for i, name in enumerate(self.reward_func_names):
                self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
            self._logs["advantages"].extend(all_process_advantages.tolist())

        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "token_advantages": token_advantages,
            "num_items_in_batch": num_items_in_batch,
        }
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        if self.use_vllm and self.vllm_importance_sampling_correction:
            output["importance_sampling_ratio"] = importance_sampling_ratio
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        return output

    def _generate_with_autocast(self, prompts, images):
        model = getattr(self, "model_wrapped", self.model)
        original_attn = getattr(getattr(model, "config", None), "_attn_implementation", None)
        original_set_attn = getattr(model, "set_attn_implementation", None)
        original_class_set_attn = None
        generation_restore_callbacks = []

        if getattr(self.args, "use_transformers_paged", False) and callable(original_set_attn):

            from transformers.modeling_utils import PreTrainedModel

            original_class_set_attn = PreTrainedModel.set_attn_implementation

            def normalize_paged_attn_name(attn_implementation):
                if attn_implementation == "paged|sdpa_paged":
                    return "paged|sdpa"
                if attn_implementation == "paged|paged_attention":
                    return "paged|flash_attention_2"
                return attn_implementation

            def class_set_attn_implementation_compat(model_self, attn_implementation, *args, **kwargs):
                return original_class_set_attn(
                    model_self,
                    normalize_paged_attn_name(attn_implementation),
                    *args,
                    **kwargs,
                )

            def set_attn_implementation_compat(attn_implementation, *args, **kwargs):
                return original_set_attn(normalize_paged_attn_name(attn_implementation), *args, **kwargs)

            PreTrainedModel.set_attn_implementation = class_set_attn_implementation_compat
            model.set_attn_implementation = set_attn_implementation_compat

        if images is None and self._needs_text_only_generation_compat():
            text_only_unused_kwargs = {
                "mm_token_type_ids",
                "pixel_values",
                "image_grid_thw",
                "pixel_attention_mask",
                "image_sizes",
            }

            try:
                from transformers.generation.utils import GenerationMixin

                original_validate_model_kwargs = GenerationMixin._validate_model_kwargs

                def validate_model_kwargs_text_only_compat(generation_self, model_kwargs):
                    for name in text_only_unused_kwargs:
                        model_kwargs.pop(name, None)
                    return original_validate_model_kwargs(generation_self, model_kwargs)

                GenerationMixin._validate_model_kwargs = validate_model_kwargs_text_only_compat
                generation_restore_callbacks.append(
                    lambda: setattr(GenerationMixin, "_validate_model_kwargs", original_validate_model_kwargs)
                )
            except Exception:
                pass

            def iter_generation_targets(*roots):
                seen = set()
                stack = [root for root in roots if root is not None]
                while stack:
                    obj = stack.pop()
                    obj_id = id(obj)
                    if obj_id in seen:
                        continue
                    seen.add(obj_id)
                    yield obj
                    for attr_name in ("base_model", "model", "module"):
                        child = getattr(obj, attr_name, None)
                        if child is not None:
                            stack.append(child)

            def strip_text_only_kwargs(kwargs):
                for name in text_only_unused_kwargs:
                    kwargs.pop(name, None)

            def patch_instance_method(obj, attr_name):
                original = getattr(obj, attr_name, None)
                if not callable(original) or getattr(original, "_spad_text_only_compat", False):
                    return

                def method_text_only_compat(*args, **kwargs):
                    strip_text_only_kwargs(kwargs)
                    return original(*args, **kwargs)

                method_text_only_compat._spad_text_only_compat = True
                try:
                    setattr(obj, attr_name, method_text_only_compat)
                except Exception:
                    return
                generation_restore_callbacks.append(lambda obj=obj, attr_name=attr_name, original=original: setattr(obj, attr_name, original))

            for target in iter_generation_targets(self.model, getattr(self, "model_wrapped", None), model):
                patch_instance_method(target, "generate")
                patch_instance_method(target, "_old_generate")

        with autocast_context():
            try:
                return self._generate(prompts, images)
            finally:
                for restore in reversed(generation_restore_callbacks):
                    restore()
                if getattr(self.args, "use_transformers_paged", False) and callable(original_set_attn):
                    model.set_attn_implementation = original_set_attn
                    if original_class_set_attn is not None:
                        from transformers.modeling_utils import PreTrainedModel

                        PreTrainedModel.set_attn_implementation = original_class_set_attn
                if original_attn is not None and getattr(model, "config", None) is not None:
                    model.config._attn_implementation = original_attn

    def _needs_text_only_generation_compat(self) -> bool:
        processor = getattr(self, "processing_class", None)
        if processor is None:
            return False
        processor_name = processor.__class__.__name__.lower()
        return (
            hasattr(processor, "image_processor")
            or hasattr(processor, "video_processor")
            or "vl" in processor_name
            or "vision" in processor_name
        )

    def _local_process_adjustments(self, inputs, completions_text: List[str]):
        if self.spad_mode == "baseline":
            return torch.zeros(len(completions_text), dtype=torch.float32), [[] for _ in completions_text]

        adjustments = []
        score_rows = []
        metric_sums: Dict[str, float] = {}

        for row, completion in zip(inputs, completions_text):
            question = str(row.get("question", ""))
            gold = str(row.get("answer", ""))
            dataset_type = str(row.get("dataset_type", "gsm8k"))
            analysis = analyze_process(question, completion, gold, dataset_type)
            metrics = process_metrics_dict(analysis)
            for k, v in metrics.items():
                metric_sums[k] = metric_sums.get(k, 0.0) + float(v)

            adjustment = 0.0
            if self.spad_mode == "c_spad":
                token_count = len(self.processing_class(text=completion, add_special_tokens=False).input_ids)
                adjustment += self.spad_cfg.consistency_reward_weight * analysis.step_consistency_rate
                adjustment += self.spad_cfg.consistency_reward_weight * analysis.final_process_consistency
                adjustment -= self.spad_cfg.repetition_penalty_weight * analysis.repetition_rate
                adjustment -= self.spad_cfg.length_penalty_weight * max(
                    0, token_count - self.spad_cfg.max_completion_length * 0.75
                )

            scores = [
                s.score + (self.spad_cfg.consistency_weight * s.consistency if self.spad_mode == "c_spad" else 0.0)
                for s in analysis.steps
            ]
            adjustments.append(float(adjustment))
            score_rows.append(scores)

        n = max(len(completions_text), 1)
        mode = "train" if self.model.training else "eval"
        for k, v in metric_sums.items():
            self._metrics[mode][f"process/{k}"].append(v / n)
        return torch.tensor(adjustments, dtype=torch.float32), score_rows

    def _compute_spad_sequence_advantages(self, rewards: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        grouped = rewards.view(-1, self.num_generations)
        grouped_lengths = lengths.view(-1, self.num_generations)
        output = torch.zeros_like(grouped)

        for i in range(grouped.size(0)):
            r = grouped[i]
            std = r.std()
            if std < 1e-6 and self.spad_cfg.zero_advantage_fix:
                if torch.all(r <= 0):
                    mean = torch.cat([r, r.new_tensor([self.spad_cfg.virtual_reward])]).mean()
                    adv = r - mean
                elif torch.all(r > 0) and self.spad_cfg.positive_length_tiebreak:
                    l = grouped_lengths[i]
                    len_std = l.std()
                    adv = torch.zeros_like(r) if len_std < 1e-6 else (l.mean() - l) / (len_std + 1e-6)
                else:
                    adv = torch.zeros_like(r)
            else:
                adv = r - r.mean()
                if self.scale_rewards != "none":
                    adv = adv / (std + 1e-4)
            output[i] = adv
        return output.view(-1)

    def _build_token_advantages(
        self,
        completions_text: List[str],
        completion_ids_list: List[List[int]],
        sequence_advantages: List[float],
        process_scores: List[List[float]],
        max_len: int,
        device,
    ) -> torch.Tensor:
        if self.spad_mode == "baseline":
            seq = torch.tensor(sequence_advantages, dtype=torch.float32, device=device).unsqueeze(1)
            return seq.expand(len(sequence_advantages), max_len)

        rows = []
        for completion, token_ids, adv, scores in zip(
            completions_text, completion_ids_list, sequence_advantages, process_scores
        ):
            step_advs = decompose_advantage(float(adv), scores, self.spad_cfg)
            token_advs = self._expand_step_advantages(completion, step_advs, len(token_ids))
            token_advs = self._normalize_token_advantages(token_advs, float(adv))
            if len(token_advs) < max_len:
                token_advs.extend([0.0] * (max_len - len(token_advs)))
            rows.append(token_advs[:max_len])
        return torch.tensor(rows, dtype=torch.float32, device=device)

    def _normalize_token_advantages(self, token_advs: List[float], sequence_advantage: float) -> List[float]:
        if not self.spad_cfg.normalize_token_advantage or abs(sequence_advantage) < 1e-8:
            return token_advs
        active = [abs(v) for v in token_advs if abs(v) > 1e-8]
        if not active:
            return token_advs
        mean_abs = float(np.mean(active))
        if mean_abs < 1e-8:
            return token_advs
        scale = abs(sequence_advantage) / mean_abs
        scale = min(scale, max(float(self.spad_cfg.max_token_advantage_scale), 1.0))
        return [float(v * scale) for v in token_advs]

    def _expand_step_advantages(self, completion: str, step_advantages: List[float], target_len: int) -> List[float]:
        if not step_advantages:
            return [0.0] * target_len
        if len(step_advantages) == 1:
            return [float(step_advantages[0])] * target_len

        expanded: List[float] = []
        steps = split_steps(completion)
        for step, adv in zip(steps, step_advantages):
            n = len(self.processing_class(text=step, add_special_tokens=False).input_ids)
            expanded.extend([float(adv)] * max(n, 1))
        if len(expanded) < target_len:
            expanded.extend([float(step_advantages[-1])] * (target_len - len(expanded)))
        return expanded[:target_len]

    def _compute_loss(self, model, inputs):
        mode = "train" if self.model.training else "eval"
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        needs_entropy = self.top_entropy_quantile < 1.0 or self.spad_cfg.entropy_advantage_weighting
        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=needs_entropy,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            num_images=inputs.get("num_images"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            image_sizes=inputs.get("image_sizes"),
            token_type_ids=inputs.get("token_type_ids"),
        )

        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(entropies, completion_mask, 1 - self.top_entropy_quantile)
        else:
            entropy_mask = None

        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        advantage_tensor = inputs.get("token_advantages")
        if advantage_tensor is None:
            advantage_tensor = inputs["advantages"].unsqueeze(1).expand_as(per_token_logps)
        advantage_tensor = advantage_tensor.to(per_token_logps.device)
        if self.spad_cfg.entropy_advantage_weighting:
            entropy_weights = entropies.detach().clamp(min=1e-8).pow(self.spad_cfg.entropy_weight_power)
            entropy_weights = entropy_weights * completion_mask
            weight_mean = entropy_weights.sum(dim=1, keepdim=True) / completion_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            entropy_weights = entropy_weights / weight_mean.clamp(min=1e-8)
            entropy_weights = entropy_weights.clamp(
                min=float(self.spad_cfg.entropy_weight_min),
                max=float(self.spad_cfg.entropy_weight_max),
            )
            advantage_tensor = advantage_tensor * entropy_weights
            mode = "train" if self.model.training else "eval"
            active_weights = entropy_weights[completion_mask.bool()]
            if active_weights.numel() > 0:
                self._metrics[mode]["gtpo/entropy_weight_mean"].append(active_weights.mean().item())
                self._metrics[mode]["gtpo/entropy_weight_max"].append(active_weights.max().item())

        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif self.importance_sampling_level == "sequence":
            log_importance_weights = (log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else:
            raise ValueError(
                f"Unknown importance sampling level: {self.importance_sampling_level}. "
                "Possible values are 'token' and 'sequence'."
            )

        coef_1 = torch.exp(log_importance_weights)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        if self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)

        per_token_loss1 = coef_1 * advantage_tensor
        per_token_loss2 = coef_2 * advantage_tensor
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask
        if self.use_vllm and self.vllm_importance_sampling_correction:
            per_token_loss = per_token_loss * inputs["importance_sampling_ratio"]
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if self.loss_type == "grpo":
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "dapo":
            normalizer = inputs["num_items_in_batch"] / self.accelerator.num_processes
            loss = (per_token_loss * completion_mask).sum() / normalizer
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        completion_token_count = completion_mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:
                return x.mean()
            return (x * completion_mask).sum() / completion_token_count

        if self.beta != 0.0:
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        if entropies is not None:
            mean_entropy = masked_batch_mean(entropies)
            self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())
        else:
            self._metrics[mode]["entropy"].append(0.0)

        sign_tensor = advantage_tensor if advantage_tensor.shape == coef_1.shape else advantage_tensor.expand_as(coef_1)
        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (sign_tensor < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (sign_tensor > 0)
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = masked_batch_mean(is_low_clipped.float())
        high_clip = masked_batch_mean(is_high_clipped.float())
        clip_ratio = masked_batch_mean(is_region_clipped.float())
        gathered_low_clip = self.accelerator.gather(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
        gathered_high_clip = self.accelerator.gather(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
        gathered_clip_ratio = self.accelerator.gather(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())
        return loss


def build_trl_dataset(tokenizer, samples) -> Dataset:
    rows = []
    for sample in samples:
        rows.append(
            {
                "prompt": format_prompt(tokenizer, sample.question, sample.dataset_type),
                "question": sample.question,
                "answer": sample.answer,
                "dataset_type": sample.dataset_type,
                "source": sample.source,
                "source_id": sample.source_id,
            }
        )
    return Dataset.from_list(rows)


def make_trl_config(
    cfg: SPADConfig,
    output_dir: str,
    *,
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
    vllm_gpu_memory_utilization: float = 0.3,
    vllm_tensor_parallel_size: int = 1,
    vllm_enable_sleep_mode: bool = False,
) -> GRPOConfig:
    if cfg.per_device_train_batch_size % cfg.num_generations != 0:
        raise ValueError(
            "per_device_train_batch_size must be divisible by num_generations "
            f"(got {cfg.per_device_train_batch_size} and {cfg.num_generations})."
        )
    return GRPOConfig(
        output_dir=output_dir,
        learning_rate=cfg.grpo_lr,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_steps=cfg.warmup_steps,
        max_steps=cfg.grpo_steps,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.grad_accum_steps,
        generation_batch_size=generation_batch_size,
        steps_per_generation=None if generation_batch_size is not None else cfg.steps_per_generation,
        max_prompt_length=cfg.max_seq_length,
        max_completion_length=cfg.max_completion_length,
        num_generations=cfg.num_generations,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        beta=cfg.beta_kl,
        max_grad_norm=cfg.max_grad_norm,
        epsilon=cfg.epsilon_clip,
        epsilon_high=cfg.epsilon_high,
        scale_rewards=cfg.scale_rewards,
        loss_type=cfg.loss_type,
        mask_truncated_completions=cfg.mask_truncated_completions,
        top_entropy_quantile=cfg.top_entropy_quantile,
        logging_steps=cfg.log_steps,
        save_strategy="no" if cfg.save_steps <= 0 else "steps",
        save_steps=max(1, cfg.save_steps),
        report_to=[],
        bf16=cfg.dtype == "bfloat16",
        fp16=cfg.dtype == "float16",
        remove_unused_columns=False,
        gradient_checkpointing=False,
        use_transformers_paged=use_transformers_paged,
        cache_implementation=cache_implementation,
        use_vllm=use_vllm,
        vllm_mode=vllm_mode,
        vllm_model_impl=vllm_model_impl,
        vllm_server_base_url=vllm_server_base_url,
        vllm_server_host=vllm_server_host,
        vllm_server_port=vllm_server_port,
        vllm_server_timeout=vllm_server_timeout,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=vllm_tensor_parallel_size,
        vllm_enable_sleep_mode=vllm_enable_sleep_mode,
    )


def train_trl(cfg: SPADConfig, mode: str, output_dir: Optional[str] = None):
    output_dir = output_dir or f"{cfg.output_dir}_trl"
    os.makedirs(output_dir, exist_ok=True)

    sft_samples, rl_samples, eval_samples = build_dataset(cfg)
    model, tokenizer = load_model(cfg)
    sft_warmup(model, tokenizer, sft_samples, cfg)
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}
    base_model = getattr(model, "base_model", None)
    if base_model is not None and not hasattr(base_model, "warnings_issued"):
        base_model.warnings_issued = model.warnings_issued

    train_dataset = build_trl_dataset(tokenizer, rl_samples)
    eval_dataset = build_trl_dataset(tokenizer, eval_samples)
    trl_args = make_trl_config(cfg, output_dir)

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
            Path(output_dir) / f"eval_details_{mode}_trl.jsonl",
        )
    metrics = {
        "mode": mode,
        "final_eval": [] if eval_result is None else [eval_result],
        "trainer_log_history": trainer.state.log_history,
    }
    with open(Path(output_dir) / f"metrics_{mode}_trl.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    if eval_result is not None:
        print(f"[EVAL] {eval_result}")
    else:
        print("[EVAL] skipped final eval; run --eval-only for formal metrics.")
    print(f"[DONE] saved to {output_dir}")


def load_peft_checkpoint_for_eval(cfg: SPADConfig, checkpoint_dir: str):
    torch_dtype = torch_dtype_from_name(cfg.dtype)
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

    base_model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **kwargs)
    if not cfg.load_in_4bit and torch.cuda.is_available():
        base_model = base_model.to("cuda")
    model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    if not cfg.load_in_4bit and torch.cuda.is_available():
        model = model.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def eval_trl_checkpoint(cfg: SPADConfig, mode: str, output_dir: Optional[str] = None, checkpoint_dir: Optional[str] = None):
    output_dir = output_dir or f"{cfg.output_dir}_trl"
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = checkpoint_dir or str(Path(output_dir) / f"{mode}_checkpoint_final")
    _, _, eval_samples = build_dataset(cfg)
    model, tokenizer = load_peft_checkpoint_for_eval(cfg, checkpoint_dir)
    eval_result = quick_eval(
        model,
        tokenizer,
        eval_samples,
        cfg,
        cfg.eval_episodes,
        Path(output_dir) / f"eval_details_{mode}_trl.jsonl",
    )

    trainer_state_path = Path(output_dir) / "checkpoint-100" / "trainer_state.json"
    log_history = []
    if trainer_state_path.exists():
        with open(trainer_state_path, "r", encoding="utf-8") as f:
            log_history = json.load(f).get("log_history", [])

    metrics = {
        "mode": mode,
        "checkpoint_dir": checkpoint_dir,
        "final_eval": [eval_result],
        "trainer_log_history": log_history,
    }
    with open(Path(output_dir) / f"metrics_{mode}_trl.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"[EVAL] {eval_result}")
    print(f"[DONE] metrics saved to {Path(output_dir) / f'metrics_{mode}_trl.json'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_spad_only.yaml")
    parser.add_argument("--mode", choices=["baseline", "spad", "c_spad"], default="spad")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--eval-pass-at-k", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--max-completion-length", type=int, default=None)
    parser.add_argument("--num-generations", type=int, default=None)
    parser.add_argument("--per-device-train-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--grpo-steps", type=int, default=None)
    parser.add_argument("--logging-steps", type=int, default=None)
    parser.add_argument("--save-steps", type=int, default=None)
    parser.add_argument("--skip-final-eval", action="store_true")
    parser.add_argument("--no-process-metrics", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
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
    if args.skip_final_eval:
        cfg.skip_final_eval = True
    if args.no_process_metrics:
        cfg.log_process_metrics = False
    if args.eval_only:
        eval_trl_checkpoint(cfg, args.mode, args.output_dir, args.checkpoint_dir)
    else:
        train_trl(cfg, args.mode, args.output_dir)


if __name__ == "__main__":
    main()
