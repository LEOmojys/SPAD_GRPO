from types import SimpleNamespace

from grpo_spad_trl import SPADConfig, SPADGRPOTrainer
import torch


class TinyTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=str(text).split())


def main():
    cfg = SPADConfig()
    trainer = object.__new__(SPADGRPOTrainer)
    trainer.spad_cfg = cfg
    trainer.spad_mode = "spad"
    trainer.num_generations = 4
    trainer.scale_rewards = "group"
    trainer.processing_class = TinyTokenizer()

    rewards = torch.tensor([1.0, 1.0, -0.5, -0.5])
    lengths = torch.tensor([80.0, 120.0, 90.0, 110.0])
    seq_adv = trainer._compute_spad_sequence_advantages(rewards, lengths)
    print("seq_adv", seq_adv.tolist())
    assert seq_adv[0] > 0 and seq_adv[1] > 0
    assert seq_adv[2] < 0 and seq_adv[3] < 0

    completion = "Step 1: 48 / 2 = 24.\nStep 2: 48 + 24 = 72.\nFinal Answer: \\boxed{72}"
    token_adv = trainer._build_token_advantages(
        completions_text=[completion],
        completion_ids_list=[list(range(18))],
        sequence_advantages=[1.0],
        process_scores=[[2.2, 2.0, 1.5]],
        max_len=18,
        device=torch.device("cpu"),
    )
    print("token_adv_sum", float(token_adv.sum()))
    assert token_adv.shape == (1, 18)
    assert token_adv.abs().sum() > 0


if __name__ == "__main__":
    main()
