from process_signals import analyze_process, extract_final_answer, verify_answer
from grpo_spad import SPADConfig, Trajectory, compute_group_advantages


def main():
    question = "Natalia sold clips to 48 friends in April and half as many in May. How many total?"
    good = """Step 1: May clips are 48 / 2 = 24.
Step 2: Total clips are 48 + 24 = 72.
Final Answer: \\boxed{72}"""
    bad = """Step 1: May clips are 48 / 2 = 26.
Step 2: Total clips are 48 + 26 = 74.
Step 3: We should check this again and again.
Final Answer: \\boxed{74}"""

    assert extract_final_answer(good) == "72"
    assert verify_answer("72", "72")
    analysis_good = analyze_process(question, good, "72")
    analysis_bad = analyze_process(question, bad, "72")
    print("good", analysis_good.expression_accuracy, analysis_good.step_consistency_rate, analysis_good.final_process_consistency)
    print("bad", analysis_bad.expression_accuracy, analysis_bad.step_consistency_rate, analysis_bad.final_process_consistency)
    assert analysis_good.expression_accuracy > analysis_bad.expression_accuracy
    assert analysis_good.final_process_consistency == 1.0

    cfg = SPADConfig()
    group = [
        Trajectory("", good, 1.0, 1.0, True, "72", 30, process_scores=[s.score for s in analysis_good.steps]),
        Trajectory("", bad, -0.5, -0.5, False, "74", 45, process_scores=[s.score for s in analysis_bad.steps]),
    ]
    compute_group_advantages(group, cfg, "spad")
    print("good advantages", group[0].step_advantages)
    print("bad advantages", group[1].step_advantages)
    assert sum(group[0].step_advantages) > 0
    assert sum(group[1].step_advantages) < 0


def test_spad_logic():
    main()


if __name__ == "__main__":
    main()
