# Model Layout

Model weights are intentionally not tracked by git. Place local Hugging Face
model snapshots here, or edit config `model.name` to point to your own model
path.

Common local layout:

```text
Models/
  qwen3_1_7b/
  qwen3_5_2b/
```

Do not commit `.safetensors`, `.bin`, or checkpoint files to this repository.
