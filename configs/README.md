# Config Layout

```text
configs/active/
  cot_baseline.yaml
  spad_answer_first.yaml
  c_spad_answer_first.yaml

configs/formal/
  cot_baseline_rl_only.yaml
  spad_rl_only.yaml
  c_spad_rl_only.yaml

configs/compare/
  dr_grpo_rl_only.yaml
  dapo_lite_rl_only.yaml
  gtpo_lite_rl_only.yaml

configs/archive/
  config_spad_only.yaml
  config_c_spad.yaml
  config_spad.yaml

configs/eval/
  gsm8k.yaml
  math.yaml
  agieval_math.yaml
  agieval_sat_math.yaml
  bbh_arithmetic.yaml
```

Use active configs for shorter TRL experiments. Use formal configs for final
RL-only GSM8K training runs. Use compare configs for frontier GRPO ablations:
Dr.GRPO, DAPO-lite, and GTPO-lite. Archive configs preserve older hand-written
and ablation settings.

Use eval configs for fair cross-dataset checkpoint evaluation. They use
deterministic generation, longer evaluation context/completion budgets, and
`eval_episodes: 0` for full-set evaluation. BBH is eval-only.

Formal training command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_formal_experiments.ps1
```

Comparison training command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_compare_experiments.ps1
```
