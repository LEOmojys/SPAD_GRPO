# Dataset Layout

Datasets are intentionally not tracked by git. Place local JSONL files here
before running training or evaluation.

Expected layout:

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

Minimum row schema:

```json
{"question": "...", "answer": "..."}
```

Optional fields:

```json
{"source": "gsm8k", "source_id": "123", "dataset_type": "gsm8k"}
```

For GSM8K, answers may use the common format:

```text
reasoning text #### final_answer
```

The loader extracts the text after `####` as the gold answer.
