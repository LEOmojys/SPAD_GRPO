from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SOURCE_ROOT = Path(r"E:\pyproject\data")
DEFAULT_OUTPUT_ROOT = Path("dataset")


def normalize_answer(text: Any) -> str:
    value = str(text).strip()
    value = value.replace("$", "")
    value = value.replace(",", "")
    value = value.strip().rstrip(".")
    return value


def extract_boxed(solution: str) -> str:
    text = str(solution)
    starts = [m.end() for m in re.finditer(r"\\boxed\{", text)]
    for start in reversed(starts):
        depth = 1
        chars: list[str] = []
        for ch in text[start:]:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return normalize_answer("".join(chars))
            chars.append(ch)
    return ""


def extract_math_answer(solution: str) -> str:
    boxed = extract_boxed(solution)
    if boxed:
        return boxed
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?(?:/\d+)?", str(solution).replace(",", ""))
    return normalize_answer(nums[-1]) if nums else ""


def cot_answer(solution: str, final_answer: str) -> str:
    body = str(solution).strip()
    final = normalize_answer(final_answer)
    if body:
        return f"{body}\n#### {final}"
    return f"#### {final}"


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            question = str(row.get("question", "")).strip()
            answer = str(row.get("answer", "")).strip()
            if not question or not answer:
                continue
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def convert_math(source_root: Path, output_root: Path) -> int:
    source = source_root / "math" / "math.json"
    with source.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    rows = []
    for source_id, item in raw.items():
        solution = str(item.get("solution", "")).strip()
        final = extract_math_answer(solution)
        if not final:
            continue
        rows.append(
            {
                "question": str(item.get("problem", "")).strip(),
                "answer": cot_answer(solution, final),
                "raw_answer": solution,
                "gold": final,
                "source": "MATH",
                "source_id": source_id,
                "dataset_type": "math",
                "level": item.get("level"),
                "type": item.get("type"),
            }
        )
    return write_jsonl(output_root / "math" / "math.jsonl", rows)


def load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def format_options(options: list[Any] | None) -> str:
    if not options:
        return ""
    return "\n".join(str(option).strip() for option in options if str(option).strip())


def convert_agieval_math(source_root: Path, output_root: Path) -> int:
    source = source_root / "AGIEval" / "data" / "v1" / "math.jsonl"
    rows = []
    for idx, item in enumerate(load_jsonl(source)):
        solution = str(item.get("other", {}).get("solution", "")).strip()
        final = normalize_answer(item.get("answer", ""))
        rows.append(
            {
                "question": str(item.get("question", "")).strip(),
                "answer": cot_answer(solution, final),
                "raw_answer": solution,
                "gold": final,
                "source": "AGIEval/math",
                "source_id": idx,
                "dataset_type": "agieval_math",
                "level": item.get("other", {}).get("level"),
                "type": item.get("other", {}).get("type"),
            }
        )
    return write_jsonl(output_root / "agieval" / "math.jsonl", rows)


def convert_agieval_sat_math(source_root: Path, output_root: Path) -> int:
    source = source_root / "AGIEval" / "data" / "v1" / "sat-math.jsonl"
    rows = []
    for idx, item in enumerate(load_jsonl(source)):
        options = format_options(item.get("options"))
        question = str(item.get("question", "")).strip()
        if options:
            question = f"{question}\nOptions:\n{options}\nAnswer with the option letter only."
        solution = str(item.get("other", {}).get("solution", "")).strip()
        final = normalize_answer(item.get("label", ""))
        rows.append(
            {
                "question": question,
                "answer": cot_answer(solution, final),
                "raw_answer": solution,
                "gold": final,
                "source": "AGIEval/sat-math",
                "source_id": idx,
                "dataset_type": "agieval_mc",
            }
        )
    return write_jsonl(output_root / "agieval" / "sat_math.jsonl", rows)


def convert_bbh_multistep(source_root: Path, output_root: Path) -> int:
    source = source_root / "BBH" / "data" / "multistep_arithmetic_two.json"
    with source.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    rows = []
    for idx, item in enumerate(raw.get("examples", [])):
        final = normalize_answer(item.get("target", ""))
        rows.append(
            {
                "question": f"Evaluate the arithmetic expression:\n{str(item.get('input', '')).strip()}",
                "answer": f"#### {final}",
                "raw_answer": f"#### {final}",
                "gold": final,
                "source": "BBH/multistep_arithmetic_two",
                "source_id": idx,
                "dataset_type": "bbh_arithmetic",
                "eval_only": True,
            }
        )
    return write_jsonl(output_root / "bbh" / "multistep_arithmetic_two.jsonl", rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    counts = {
        "math/math.jsonl": convert_math(args.source_root, args.output_root),
        "agieval/math.jsonl": convert_agieval_math(args.source_root, args.output_root),
        "agieval/sat_math.jsonl": convert_agieval_sat_math(args.source_root, args.output_root),
        "bbh/multistep_arithmetic_two.jsonl": convert_bbh_multistep(args.source_root, args.output_root),
    }

    manifest = {
        "source_root": str(args.source_root),
        "output_root": str(args.output_root),
        "format": "JSONL with question, answer, raw_answer, gold, source, source_id, dataset_type",
        "notes": [
            "answer uses GSM8K-compatible '#### final' suffix.",
            "BBH is marked eval_only and should not be used for training.",
            "AGIEval sat_math expects the model to answer with the option letter.",
        ],
        "counts": counts,
    }
    manifest_path = args.output_root / "aligned_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
