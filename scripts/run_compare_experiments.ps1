$ErrorActionPreference = "Stop"

python grpo_spad_trl.py --config configs/compare/dr_grpo_rl_only.yaml --mode baseline
python grpo_spad_trl.py --config configs/compare/dapo_lite_rl_only.yaml --mode baseline
python grpo_spad_trl.py --config configs/compare/gtpo_lite_rl_only.yaml --mode baseline
