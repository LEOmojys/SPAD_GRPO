# SPAD-GRPO

SPAD-GRPO is a research codebase for studying process-aware variants of Group
Relative Policy Optimization (GRPO) on mathematical chain-of-thought reasoning.

The project implements three training modes:

- `baseline`: standard CoT-GRPO with final-answer and format rewards.
- `spad`: Statistical Process Advantage Decomposition, which redistributes
  sequence-level advantages over reasoning steps.
- `c_spad`: SPAD plus consistency-aware reward shaping and process weighting.

The main implementation is based on TRL's `GRPOTrainer`. An optional Unsloth
entry point is provided for faster local LoRA training when the environment
supports it.

## Repository Status

This repository is intended to contain source code, configs, scripts, and
documentation only. Large local artifacts are intentionally excluded:

- model weights under `Models/`
- generated checkpoints and metrics under `results/`
- virtual environments and package caches
- private manuscript files

The lightweight JSONL datasets under `dataset/` are versioned in this project
when their licenses permit redistribution. Use the layout below to place model
weights and generated outputs locally when running experiments.

## Project Layout

```text
grpo_spad.py                Shared config loading, model loading, evaluation,
                            process scoring, and legacy handwritten trainer
grpo_spad_trl.py            TRL GRPOTrainer implementation with SPAD/C-SPAD
grpo_spad_unsloth.py        Unsloth-backed TRL GRPO entry point
process_signals.py          Deterministic answer parsing and process diagnostics

configs/
  formal/                   RL-only baseline/SPAD/C-SPAD configs
  compare/                  Controlled ablation baselines
  eval/                     Cross-dataset evaluation configs
  archive/                  Older configs

scripts/
  run_formal_experiments.ps1
  run_compare_experiments.ps1
  run_unsloth_fast.ps1

test_spad_logic.py
test_trl_spad_logic.py
requirements.txt
ENVIRONMENT.md
```

## Installation

The project was developed on Windows with Python 3.12, CUDA 12.4, and
PyTorch 2.6.0. Linux should also work with compatible CUDA/PyTorch wheels.

Create an environment:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
```

Install PyTorch first. For CUDA 12.4:

```powershell
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124
```

Then install project dependencies:

```powershell
pip install -r requirements.txt
```

See [ENVIRONMENT.md](ENVIRONMENT.md) for local Unsloth notes and environment
details.

## Model Setup

Configs expect local model directories such as:

```text
Models/qwen3_1_7b/
Models/qwen3_5_2b/
```

Model weights are not included in this repository. Download or copy compatible
Hugging Face model snapshots into `Models/`, or edit the config `model.name`
field to point to your own local or remote model path.

For Qwen3-1.7B experiments, this project uses standard causal LM modules:

```text
q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
```

## Dataset Setup

The repository includes the aligned JSONL datasets used by the current
experiments:

```text
dataset/
  gsm8k/
    train.jsonl
    test.jsonl
  math/
    math.jsonl
  agieval/
    math.jsonl
    sat_math.jsonl
  bbh/
    multistep_arithmetic_two.jsonl
```

Each JSONL row should include at least:

```json
{"question": "...", "answer": "..."}
```

GSM8K-style answers may use the standard `#### final_answer` suffix. The
loader normalizes gold answers automatically.

## Quick Checks

```powershell
python test_spad_logic.py
python test_trl_spad_logic.py
python grpo_spad_trl.py --help
```

## Training

TRL baseline:

```powershell
python grpo_spad_trl.py --config configs/formal/cot_baseline_rl_only.yaml --mode baseline
```

TRL SPAD:

```powershell
python grpo_spad_trl.py --config configs/formal/spad_rl_only.yaml --mode spad
```

TRL C-SPAD:

```powershell
python grpo_spad_trl.py --config configs/formal/c_spad_rl_only.yaml --mode c_spad
```

Optional Unsloth baseline:

```powershell
.\.conda_unsloth\python.exe grpo_spad_unsloth.py --config configs/formal/cot_baseline_rl_only_qwen3_17b_unsloth_fast.yaml --mode baseline
```

Wrapper for Unsloth runs:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_unsloth_fast.ps1 -Mode baseline
powershell -ExecutionPolicy Bypass -File scripts\run_unsloth_fast.ps1 -Mode spad
powershell -ExecutionPolicy Bypass -File scripts\run_unsloth_fast.ps1 -Mode c_spad
```

Runtime overrides are available for fast local experiments:

```powershell
.\.conda_unsloth\python.exe grpo_spad_unsloth.py `
  --config configs/formal/cot_baseline_rl_only_qwen3_17b_unsloth_fast.yaml `
  --mode baseline `
  --max-completion-length 256 `
  --num-generations 4 `
  --per-device-train-batch-size 4 `
  --gradient-accumulation-steps 4 `
  --lora-dropout 0
```

## Evaluation

Evaluate an Unsloth LoRA checkpoint:

```powershell
.\.conda_unsloth\python.exe grpo_spad_unsloth.py `
  --config configs/formal/cot_baseline_rl_only_qwen3_17b_unsloth_fast.yaml `
  --mode baseline `
  --eval-only `
  --checkpoint-dir results/formal/cot_baseline_rl_only_qwen3_17b_unsloth_fast_unsloth\baseline_checkpoint_final `
  --eval-pass-at-k 8
```

For a short smoke evaluation:

```powershell
.\.conda_unsloth\python.exe grpo_spad_unsloth.py `
  --config configs/formal/cot_baseline_rl_only_qwen3_17b_unsloth_fast.yaml `
  --mode baseline `
  --eval-only `
  --checkpoint-dir results/formal/cot_baseline_rl_only_qwen3_17b_unsloth_fast_unsloth\checkpoint-20 `
  --eval-episodes 20 `
  --eval-pass-at-k 8
```

Evaluation writes:

```text
metrics_<mode>_unsloth_eval.json
eval_details_<mode>_unsloth.jsonl
```

The JSONL details file contains per-problem completions, extracted answers,
correctness flags, token counts, format checks, and truncation flags.

## Metrics

Primary metrics:

- `avg@1`: greedy final-answer accuracy (`do_sample=False`).
- `avg@8`: mean final-answer accuracy over 8 stochastic samples per problem
  (`temperature=1.0` by default).
- `majority@8`: accuracy after majority voting over 8 sampled final answers.

Sampling stability:

- `majority_agreement@8`: fraction of extracted sampled answers matching the
  most frequent sampled answer.
- `unique_answers@8`: mean number of unique extracted answers among 8 samples.

Format and efficiency:

- `greedy_answer_extraction_rate`
- `sampled_answer_extraction_rate@8`
- `greedy_format_accuracy`
- `sampled_format_accuracy@8`
- `greedy_avg_tokens`
- `sampled_avg@8_tokens`
- `greedy_truncation_rate`
- `sampled_truncation_rate@8`

Heuristic process diagnostics:

- `expression_accuracy`
- `step_consistency_rate`
- `final_process_consistency`
- `repetition_rate`
- `invalid_step_rate`
- `contradiction_rate`

The process diagnostics are deterministic heuristics for SPAD/C-SPAD analysis.
They should be treated as auxiliary evidence, not as standard benchmark
metrics.

## Method Notes

SPAD decomposes a sequence-level GRPO advantage into step-level token weights.
For positive advantages, higher-scoring reasoning steps receive more weight.
For negative advantages, lower-quality steps receive more penalty. C-SPAD adds
process-consistency shaping to discourage repeated, contradictory, or internally
inconsistent reasoning.

This project does not claim to fully reproduce DAPO, GTPO, or other large-scale
RL recipes. Configs under `configs/compare/` are controlled lightweight
ablations inspired by those methods.

## References

- Cobbe et al. 2021. *Training Verifiers to Solve Math Word Problems.*
  arXiv:2110.14168.
- Wei et al. 2022. *Chain-of-Thought Prompting Elicits Reasoning in Large
  Language Models.* arXiv:2201.11903.
- Wang et al. 2022. *Self-Consistency Improves Chain of Thought Reasoning in
  Language Models.* arXiv:2203.11171.
- Holtzman et al. 2019. *The Curious Case of Neural Text Degeneration.*
  arXiv:1904.09751.
- Lightman et al. 2023. *Let's Verify Step by Step.* arXiv:2305.20050.

## License

No license has been selected yet. Add a license before publishing if you want
others to use or redistribute the code.
