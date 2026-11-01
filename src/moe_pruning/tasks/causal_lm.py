"""Decoder-only objective (OLMoE and other causal LMs).

* Calibration: ``human`` and ``response`` are joined with a fixed template
  (or the tokenizer's chat template when it has one). All tokens are fed to
  the model and contribute to MoE statistics; the loss labels cover only the
  response (teacher forcing).
* Evaluation: next-token prediction on the evaluation document (truncated to
  MAX_SOURCE_LENGTH tokens). The mean per-token NLL is a genuine causal LM
  loss, so exp(loss) is reported as perplexity.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from moe_pruning.data.calibration import TokenizedPair
from moe_pruning.data.collators import collate_causal
from moe_pruning.tasks.base import TaskObjective, clean_decode

PROMPT_TEMPLATE = "User: {human}\nAssistant: "
QUALITATIVE_PROMPT_TOKENS = 64
QUALITATIVE_NEW_TOKENS = 32


@dataclass(frozen=True)
class CausalExample:
    example_id: int
    input_ids: list[int]
    labels: list[int]  # identical to input_ids; the model shifts internally


class CausalLMTask(TaskObjective):
    name = "causal_lm"
    stacks = ("decoder",)
    metric_definition = "mean per-token next-token cross-entropy on the evaluation documents"
    reports_perplexity = True

    def _prompt_ids(self, tokenizer, human: str) -> list[int]:
        if getattr(tokenizer, "chat_template", None):
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": human}], add_generation_prompt=True, tokenize=True
            )
        return tokenizer(PROMPT_TEMPLATE.format(human=human)).input_ids

    def tokenize_calibration(self, records, tokenizer, max_source_length, max_target_length):
        out = []
        eos = tokenizer.eos_token_id
        for i, rec in enumerate(records):
            prompt = self._prompt_ids(tokenizer, rec.human)[:max_source_length]
            response = tokenizer(rec.response, add_special_tokens=False).input_ids[: max_target_length - 1] + [eos]
            out.append(TokenizedPair(i, prompt + response, [-100] * len(prompt) + response))
        return out

    def collate(self, items, pad_token_id):
        return collate_causal(items, pad_token_id)

    def eval_settings(self, cfg) -> dict:
        return {"task": self.name, "max_length": cfg.max_source_length}

    def build_eval_example(self, example_id, text, tokenizer, cfg):
        ids = tokenizer(text).input_ids[: cfg.max_source_length]
        return CausalExample(example_id, ids, list(ids))

    @staticmethod
    def _shift(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return logits[:, :-1], labels[:, 1:]

    def token_nll(self, logits, labels):
        labels = labels.to(logits.device)  # sharded models return logits on the last GPU
        logits, labels = self._shift(logits, labels)
        nll = torch.nn.functional.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100, reduction="none"
        ).view(labels.shape)
        mask = labels != -100
        return (nll * mask).sum(dim=-1), mask.sum(dim=-1)

    def predictions(self, logits, labels):
        labels = labels.to(logits.device)
        logits, labels = self._shift(logits, labels)
        argmax = logits.argmax(dim=-1).cpu()
        mask = (labels != -100).cpu()
        return [argmax[i][mask[i]].tolist() for i in range(labels.shape[0])]

    @torch.no_grad()
    def qualitative(self, model, tokenizer, examples, device, cfg):
        out = []
        for ex in examples:
            n_prompt = max(1, min(QUALITATIVE_PROMPT_TOKENS, len(ex.input_ids) // 2))
            prompt = torch.tensor([ex.input_ids[:n_prompt]], device=device)
            generated = model.generate(
                input_ids=prompt, attention_mask=torch.ones_like(prompt),
                max_new_tokens=QUALITATIVE_NEW_TOKENS, do_sample=False, num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
            )
            out.append(
                {
                    "example_id": ex.example_id,
                    "inputs": {
                        "Prompt": clean_decode(tokenizer, ex.input_ids[:n_prompt]),
                        "Reference continuation": clean_decode(
                            tokenizer, ex.input_ids[n_prompt : n_prompt + QUALITATIVE_NEW_TOKENS]
                        ),
                    },
                    "prediction": clean_decode(tokenizer, generated[0, n_prompt:].tolist()),
                }
            )
        return out
