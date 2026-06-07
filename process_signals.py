"""
Process signals and metrics for math CoT GRPO experiments.

This module is intentionally model-free. It provides deterministic, shared
post-processing for SPAD/C-SPAD training and evaluation.
"""

from __future__ import annotations

import ast
import json
import math
import operator
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class MathSample:
    question: str
    answer: str
    raw_answer: str = ""
    source: str = ""
    source_id: str = ""
    dataset_type: str = "gsm8k"


@dataclass
class StepSignal:
    text: str
    score: float
    consistency: float
    components: Dict[str, float] = field(default_factory=dict)


@dataclass
class ProcessAnalysis:
    steps: List[StepSignal]
    final_answer: str
    expression_accuracy: float
    step_consistency_rate: float
    final_process_consistency: float
    repetition_rate: float
    invalid_step_rate: float
    contradiction_rate: float


_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_ALLOWED_UNARY = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def load_gsm8k_json(path: str | Path, max_samples: Optional[int] = None) -> List[MathSample]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        if path.suffix.lower() == ".jsonl":
            raw = [json.loads(line) for line in f if line.strip()]
        else:
            raw = json.load(f)
    samples: List[MathSample] = []
    selected = raw if max_samples is None or max_samples <= 0 else raw[:max_samples]
    for item in selected:
        question = item.get("question", item.get("q", "")).strip()
        raw_answer = str(item.get("answer", "")).strip()
        answer = extract_gold_answer(raw_answer)
        source = str(item.get("source", "")).strip()
        source_id = str(item.get("source_id", "")).strip()
        dataset_type = str(item.get("dataset_type", "")).strip() or infer_dataset_type(path, source, question)
        if question and answer:
            samples.append(
                MathSample(
                    question=question,
                    answer=answer,
                    raw_answer=raw_answer,
                    source=source,
                    source_id=source_id,
                    dataset_type=dataset_type,
                )
            )
    return samples


def infer_dataset_type(path: str | Path, source: str = "", question: str = "") -> str:
    marker = f"{Path(path).as_posix()} {source} {question[:120]}".lower()
    if "sat-math" in marker or "sat_math" in marker:
        return "agieval_mc"
    if "bbh" in marker or "multistep_arithmetic" in marker:
        return "bbh_arithmetic"
    if "agieval" in marker:
        return "agieval_math"
    if "/math/" in marker or "\\math\\" in marker or "source\": \"math" in marker:
        return "math"
    return "gsm8k"


def extract_gold_answer(raw_answer: str) -> str:
    text = str(raw_answer)
    if "####" in text:
        return normalize_answer(text.split("####")[-1])
    return normalize_answer(text)


def normalize_answer(answer: str) -> str:
    text = str(answer).strip()
    text = re.sub(r"\\boxed\{([^{}]+)\}", r"\1", text)
    text = re.sub(r"\\text\{([^{}]*)\}", r"\1", text)
    text = text.replace(",", "")
    text = text.replace("$", "")
    text = text.strip().rstrip(".")
    return text


def extract_final_answer(text: str) -> str:
    if not text:
        return ""
    patterns = [
        r"Final Answer\s*:\s*(.+)",
        r"final answer is\s*(.+)",
        r"answer is\s*(.+)",
        r"\\boxed\{([^{}]+)\}",
        r"####\s*(.+)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return normalize_answer(matches[-1].splitlines()[0])
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?(?:/\d+)?", text.replace(",", ""))
    return normalize_answer(nums[-1]) if nums else ""


def verify_answer(pred: str, gold: str, rel_tol: float = 1e-6, dataset_type: str = "gsm8k") -> bool:
    if dataset_type == "agieval_mc":
        pred_choice = extract_choice_answer(pred)
        gold_choice = extract_choice_answer(gold)
        return bool(pred_choice and gold_choice and pred_choice == gold_choice)

    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)
    if pred_norm == gold_norm:
        return True
    pred_val = _to_float(pred_norm)
    gold_val = _to_float(gold_norm)
    if pred_val is not None and gold_val is not None:
        return math.isclose(pred_val, gold_val, rel_tol=rel_tol, abs_tol=rel_tol)

    if dataset_type in {"math", "agieval_math"}:
        return normalize_symbolic_answer(pred_norm) == normalize_symbolic_answer(gold_norm)
    return False


def extract_choice_answer(text: str) -> str:
    value = normalize_answer(text).strip().upper()
    matches = re.findall(r"\b([A-E])\b", value)
    if matches:
        return matches[-1]
    matches = re.findall(r"\(([A-E])\)", value)
    return matches[-1] if matches else ""


def normalize_symbolic_answer(answer: str) -> str:
    text = normalize_answer(answer)
    replacements = {
        "\\left": "",
        "\\right": "",
        "\\cdot": "*",
        "\\times": "*",
        "\\div": "/",
        "\\le": "<=",
        "\\leq": "<=",
        "\\ge": ">=",
        "\\geq": ">=",
        "\\infty": "infty",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    text = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", text)
    text = re.sub(r"\s+", "", text)
    text = text.strip("{}")
    return text.lower()


def format_cot_solution(raw_answer: str, final_answer: Optional[str] = None) -> str:
    final = final_answer or extract_gold_answer(raw_answer)
    body = str(raw_answer).split("####")[0].strip()
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    steps = []
    for idx, line in enumerate(lines, start=1):
        cleaned = re.sub(r"<<([^=<>]+)=([^<>]+)>>", r"\1 = \2", line)
        steps.append(f"Step {idx}: {cleaned}")
    if not steps:
        steps.append(f"Step 1: The answer can be computed from the problem statement.")
    steps.append(f"Final Answer: \\boxed{{{normalize_answer(final)}}}")
    return "\n".join(steps)


def split_steps(text: str) -> List[str]:
    body = str(text).strip()
    if not body:
        return []
    body = re.sub(r"</?think>", "", body, flags=re.IGNORECASE).strip()
    parts = re.split(r"(?im)(?=^\s*(?:Step\s*\d+|[0-9]+\.)\s*[:.)-])", body)
    steps = [p.strip() for p in parts if p.strip()]
    if len(steps) <= 1:
        steps = [p.strip() for p in re.split(r"\n+", body) if p.strip()]
    if len(steps) <= 1:
        steps = [p.strip() for p in re.split(r"(?<=[.!?])\s+", body) if p.strip()]
    return steps


def analyze_process(
    question: str,
    completion: str,
    gold_answer: str = "",
    dataset_type: str = "gsm8k",
) -> ProcessAnalysis:
    steps = split_steps(completion)
    final_answer = extract_final_answer(completion)
    step_signals: List[StepSignal] = []
    equation_total = 0
    equation_ok = 0
    invalid_steps = 0
    repeated_steps = 0
    contradictions = 0
    previous_results: List[float] = []
    previous_texts: List[str] = []

    for step in steps:
        components = score_step_components(step, question, previous_texts)
        eqs = extract_equations(step)
        local_eq_total = len(eqs)
        local_eq_ok = 0
        local_results: List[float] = []
        local_invalid_expr = False
        for left, right in eqs:
            equation_total += 1
            equation_ok_flag, value = verify_equation(left, right)
            if equation_ok_flag:
                equation_ok += 1
                local_eq_ok += 1
            else:
                components["invalid_expr_penalty"] = -1.0
                local_invalid_expr = True
            if value is not None:
                local_results.append(value)

        consistency = score_consistency(step, previous_results)
        if local_invalid_expr:
            consistency = min(consistency, -0.5)
        if consistency < 0:
            contradictions += 1
        if components.get("repetition_penalty", 0.0) < 0:
            repeated_steps += 1
        if components.get("invalid_expr_penalty", 0.0) < 0 or components.get("progress_score", 0.0) <= 0:
            invalid_steps += 1

        score = sum(components.values())
        if local_eq_total:
            score += local_eq_ok / max(local_eq_total, 1)
        step_signals.append(
            StepSignal(
                text=step,
                score=float(max(-3.0, min(3.0, score))),
                consistency=float(max(-1.0, min(1.0, consistency))),
                components=components,
            )
        )
        previous_results.extend(local_results)
        previous_texts.append(step)

    n_steps = max(len(steps), 1)
    expr_acc = equation_ok / equation_total if equation_total else 1.0
    step_cons = sum(1 for s in step_signals if s.consistency >= 0) / n_steps
    fpc = compute_final_process_consistency(final_answer, steps, gold_answer, dataset_type)
    return ProcessAnalysis(
        steps=step_signals,
        final_answer=final_answer,
        expression_accuracy=expr_acc,
        step_consistency_rate=step_cons,
        final_process_consistency=fpc,
        repetition_rate=repeated_steps / n_steps,
        invalid_step_rate=invalid_steps / n_steps,
        contradiction_rate=contradictions / n_steps,
    )


def score_step_components(step: str, question: str, previous_steps: Iterable[str]) -> Dict[str, float]:
    text = step.strip()
    lower = text.lower()
    q_words = set(re.findall(r"[a-zA-Z]{3,}", question.lower()))
    s_words = set(re.findall(r"[a-zA-Z]{3,}", lower))
    relevance = len(q_words & s_words) / max(len(q_words), 1)
    has_math = bool(re.search(r"\d|\+|-|\*|/|=|\\frac", text))
    has_final = "final answer" in lower or "\\boxed" in text
    progress = 1.0 if has_math or has_final else 0.2
    repetition = 0.0
    for prev in previous_steps:
        if text_similarity(prev, text) >= 0.82:
            repetition = -1.0
            break
    overlong = -0.5 if len(text.split()) > 80 else 0.0
    format_score = 0.3 if re.match(r"(?i)^\s*(step\s*\d+|[0-9]+\.)", text) or has_final else 0.0
    return {
        "math_expr_score": 0.8 if has_math else 0.0,
        "progress_score": progress,
        "answer_relevance": min(0.6, relevance),
        "format_score": format_score,
        "repetition_penalty": repetition,
        "overlong_penalty": overlong,
        "invalid_expr_penalty": 0.0,
    }


def text_similarity(a: str, b: str) -> float:
    aw = set(re.findall(r"\w+", a.lower()))
    bw = set(re.findall(r"\w+", b.lower()))
    if not aw or not bw:
        return 0.0
    return len(aw & bw) / len(aw | bw)


def extract_equations(text: str) -> List[Tuple[str, str]]:
    cleaned = normalize_math_text(text)
    equations: List[Tuple[str, str]] = []
    for match in re.finditer(r"([0-9][0-9\.\s+\-*/()%]*)=+\s*([-+]?[0-9]+(?:\.[0-9]+)?)", cleaned):
        left = match.group(1).strip()
        right = match.group(2).strip()
        if any(op in left for op in ["+", "-", "*", "/", "%"]):
            equations.append((left, right))
    return equations


def normalize_math_text(text: str) -> str:
    cleaned = str(text)
    cleaned = cleaned.replace(",", "")
    cleaned = cleaned.replace("×", "*").replace("÷", "/")
    cleaned = cleaned.replace("^", "**")
    cleaned = re.sub(r"\$|USD|dollars?", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<<([^=<>]+)=([^<>]+)>>", r"\1=\2", cleaned)
    return cleaned


def verify_equation(left: str, right: str) -> Tuple[bool, Optional[float]]:
    left_val = safe_eval_numeric(left)
    right_val = safe_eval_numeric(right)
    if left_val is None or right_val is None:
        return False, left_val
    return math.isclose(left_val, right_val, rel_tol=1e-6, abs_tol=1e-6), right_val


def safe_eval_numeric(expr: str) -> Optional[float]:
    expr = normalize_math_text(expr)
    expr = re.sub(r"[^0-9+\-*/().% ]", "", expr)
    expr = expr.replace("%", "/100")
    if not expr.strip():
        return None
    try:
        node = ast.parse(expr, mode="eval")
        return float(_eval_ast(node.body))
    except Exception:
        return None


def _eval_ast(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_eval_ast(node.left), _eval_ast(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
        return _ALLOWED_UNARY[type(node.op)](_eval_ast(node.operand))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def _to_float(text: str) -> Optional[float]:
    try:
        if "/" in text and re.fullmatch(r"[-+]?\d+(?:\.\d+)?/[-+]?\d+(?:\.\d+)?", text):
            n, d = text.split("/", 1)
            return float(n) / float(d)
        return float(text)
    except Exception:
        return None


def score_consistency(step: str, previous_results: List[float]) -> float:
    if not previous_results:
        return 0.0
    nums = [_to_float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", step.replace(",", ""))]
    nums = [x for x in nums if x is not None]
    if not nums:
        return 0.0
    # Penalize if the step appears to restate a recent result with a nearby but different value.
    for prev in previous_results[-3:]:
        for num in nums:
            if abs(num - prev) <= max(2.0, abs(prev) * 0.05) and not math.isclose(num, prev, rel_tol=1e-6, abs_tol=1e-6):
                return -0.5
    return 0.2


def compute_final_process_consistency(
    final_answer: str,
    steps: List[str],
    gold_answer: str = "",
    dataset_type: str = "gsm8k",
) -> float:
    if not steps or not final_answer:
        return 0.0
    final_norm = normalize_answer(final_answer)
    if gold_answer and verify_answer(final_norm, gold_answer, dataset_type=dataset_type):
        return 1.0
    body = "\n".join(steps[:-1]) if len(steps) > 1 else steps[0]
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?(?:/\d+)?", body.replace(",", ""))
    if dataset_type != "agieval_mc" and nums and verify_answer(final_norm, nums[-1], dataset_type=dataset_type):
        return 1.0
    return 0.0


def process_metrics_dict(analysis: ProcessAnalysis) -> Dict[str, float]:
    return {
        "expression_accuracy": analysis.expression_accuracy,
        "step_consistency_rate": analysis.step_consistency_rate,
        "final_process_consistency": analysis.final_process_consistency,
        "repetition_rate": analysis.repetition_rate,
        "invalid_step_rate": analysis.invalid_step_rate,
        "contradiction_rate": analysis.contradiction_rate,
    }
