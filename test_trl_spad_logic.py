from types import SimpleNamespace

import torch

from grpo_spad import SPADConfig


class TinyTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=str(text).split())


class TinySPADTrainer:
    def _compute_spad_sequence_advantages(self, rewards: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        grouped = rewards.view(-1, self.num_generations)
        grouped_lengths = lengths.view(-1, self.num_generations)
        output = torch.zeros_like(grouped)

        for i in range(grouped.size(0)):
            r = grouped[i]
            std = r.std()
            if std < 1e-6 and self.spad_cfg.zero_advantage_fix:
                if torch.all(r <= 0):
                    mean = torch.cat([r, r.new_tensor([self.spad_cfg.virtual_reward])]).mean()
                    adv = r - mean
                elif torch.all(r > 0) and self.spad_cfg.positive_length_tiebreak:
                    l = grouped_lengths[i]
                    len_std = l.std()
                    adv = torch.zeros_like(r) if len_std < 1e-6 else (l.mean() - l) / (len_std + 1e-6)
                else:
                    adv = torch.zeros_like(r)
            else:
                adv = r - r.mean()
                if self.scale_rewards != "none":
                    adv = adv / (std + 1e-4)
            output[i] = adv
        return output.view(-1)

    def _build_token_advantages(
        self,
        completions_text,
        completion_ids_list,
        sequence_advantages,
        process_scores,
        max_len,
        device,
    ) -> torch.Tensor:
        from grpo_spad import decompose_advantage
        from process_signals import split_steps

        rows = []
        for completion, token_ids, adv, scores in zip(
            completions_text, completion_ids_list, sequence_advantages, process_scores
        ):
            step_advs = decompose_advantage(float(adv), scores, self.spad_cfg)
            expanded = []
            steps = split_steps(completion)
            for step, step_adv in zip(steps, step_advs):
                n = len(self.processing_class(text=step, add_special_tokens=False).input_ids)
                expanded.extend([float(step_adv)] * max(n, 1))
            if len(expanded) < len(token_ids) and step_advs:
                expanded.extend([float(step_advs[-1])] * (len(token_ids) - len(expanded)))
            if len(expanded) < max_len:
                expanded.extend([0.0] * (max_len - len(expanded)))
            rows.append(expanded[:max_len])
        return torch.tensor(rows, dtype=torch.float32, device=device)


def main():
    cfg = SPADConfig()
    trainer = TinySPADTrainer()
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


def test_trl_spad_logic():
    main()


if __name__ == "__main__":
    main()
