# Environment Setup

This project was tested on:

```text
OS: Windows
Python: 3.12
CUDA: 12.4
torch: 2.6.0+cu124
transformers: 5.5.0
trl: 0.24.0
peft: 0.19.1
bitsandbytes: 0.49.2
accelerate: 1.6.0
```

It should also run on Linux with Python 3.10. On Python 3.10, do not install
`pandas>=3`; pandas is optional and not needed for training.

## Fresh Setup

Create and activate a virtual environment:

```powershell
cd E:\pyproject\SAE_DRP
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
```

Install PyTorch for your device first. For CUDA 12.4:

```powershell
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124
```

Then install project dependencies:

```powershell
pip install -r requirements.txt
```

On Linux servers with a newer driver, such as CUDA 13.0 shown by
`nvidia-smi`, CUDA 12.x PyTorch wheels are still normally usable because NVIDIA
drivers are backward-compatible with older CUDA runtimes. For example:

```bash
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Notes

- `torchao==0.15.0` is not required. In the current environment it warns about an incompatible torch version and is skipped.
- `causal-conv1d` and `flash-linear-attention` are optional acceleration packages. If absent, Qwen falls back to the torch implementation.
- `mergekit` and `llm-blender` are not used directly by the project, but some
  TRL versions import them through optional callback/judge paths. They are kept
  in `requirements.txt` to avoid import-time failures.
- `weave`, `unsloth`, and `unsloth_zoo` are not required for the current training scripts.
- Keep model checkpoints and local model directories such as `Models/qwen3_5_2b` outside `requirements.txt`; they must be copied or downloaded separately.

## Optional Unsloth Backend

The project includes an Unsloth-backed entry point. On the local Windows
Unsloth environment, call the Conda Python directly instead of activating it:

```powershell
.\.conda_unsloth\python.exe grpo_spad_unsloth.py --config configs/formal/cot_baseline_rl_only_unsloth_fast.yaml --mode baseline
```

Or use the wrapper script:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_unsloth_fast.ps1 -Mode baseline
powershell -ExecutionPolicy Bypass -File scripts\run_unsloth_fast.ps1 -Mode spad
powershell -ExecutionPolicy Bypass -File scripts\run_unsloth_fast.ps1 -Mode c_spad
```

By default the wrapper uses the Qwen3-1.7B configs under
`Models/qwen3_1_7b`. Pass `-Model qwen3_5_2b` to use the older Qwen3.5-2B
configs.

The `*_unsloth_fast.yaml` configs keep the GRPO group/batch structure but set
LoRA dropout to `0.0`, reduce `max_completion_length` to `512`, and stop at
`200` RL steps to avoid the later drift seen in longer runs. Do not use
`--use-transformers-paged` for `Models/qwen3_5_2b`: its hybrid
`linear_attention` layers are not supported by Transformers paged generation,
so the script falls back to default generation when that flag is supplied.

## Smoke Test

After installation:

```powershell
python grpo_spad_trl.py --help
python test_spad_logic.py
python test_trl_spad_logic.py
```

For formal RL-only training:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_formal_experiments.ps1
```
