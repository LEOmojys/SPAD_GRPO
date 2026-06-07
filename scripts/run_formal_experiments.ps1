$ErrorActionPreference = "Stop"

python grpo_spad_trl.py --config configs/formal/cot_baseline_rl_only.yaml --mode baseline
python grpo_spad_trl.py --config configs/formal/spad_rl_only.yaml --mode spad
python grpo_spad_trl.py --config configs/formal/c_spad_rl_only.yaml --mode c_spad

