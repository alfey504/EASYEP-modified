"""Encoder-decoder objective (T5 / Switch Transformers).

* Calibration: ``human`` -> encoder input, ``response`` -> decoder labels
  (teacher forcing; the model shifts the labels right).
* Evaluation: deterministic T5 span corruption of the document; the loss is
  the mean per-token cross-entropy of the sentinel-delimited targets. This is
  a denoising loss, not perplexity.
"""

from __future__ import annotations

import torch

from moe_pruning.data.calibration import tokenize_calibration
from moe_pruning.data.collators import collate_seq2seq
from moe_pruning.data.span_corruption import corrupt_example
from moe_pruning.tasks.base import TaskObjective, clean_decode


class Seq2SeqDenoisingTask(TaskObjective):
    name = "seq2seq_span_denoising"
    stacks = ("encoder", "decoder")
    metric_definition = (
        "mean per-token cross-entropy of T5 span-corruption targets (teacher forced); "
        "not autoregressive perplexity"
    )
    reports_perplexity = False

    def tokenize_calibration(self, records, tokenizer, max_source_length, max_target_length):
        return tokenize_calibration(records, tokenizer, max_source_length, max_target_length)

    def collate(self, items, pad_token_id):
        return collate_seq2seq(items, pad_token_id)

    def eval_settings(self, cfg) -> dict:
        return {
            "task": self.name,
            "noise_density": cfg.eval_noise_density,
            "mean_noise_span_length": cfg.eval_mean_span_length,
            "seed": cfg.eval_seed,
            "max_source_length": cfg.max_source_length,
            "max_target_length": cfg.max_target_length,
        }

    def build_eval_example(self, example_id, text, tokenizer, cfg):
        ids = tokenizer(text, add_special_tokens=False).input_ids
        return corrupt_example(
            example_id, ids, tokenizer, cfg.eval_noise_density, cfg.eval_mean_span_length,
            cfg.eval_seed, cfg.max_source_length, cfg.max_target_length,
        )

    def token_nll(self, logits, labels):
        # Decoder logits are already aligned with the labels.
        nll = torch.nn.functional.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100, reduction="none"
        ).view(labels.shape)
        mask = labels != -100
        return (nll * mask).sum(dim=-1), mask.sum(dim=-1)

    def predictions(self, logits, labels):
        argmax = logits.argmax(dim=-1).cpu()
        mask = (labels != -100).cpu()
        return [argmax[i][mask[i]].tolist() for i in range(labels.shape[0])]

    @torch.no_grad()
    def qualitative(self, model, tokenizer, examples, device, cfg):
        out = []
        for ex in examples:
            input_ids = torch.tensor([ex.input_ids], device=device)
            generated = model.generate(
                input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                max_new_tokens=cfg.max_target_length, do_sample=False, num_beams=1,
            )
            out.append(
                {
                    "example_id": ex.example_id,
                    "inputs": {
                        "Encoder input (original)": clean_decode(tokenizer, ex.source_ids),
                        "Corrupted input": clean_decode(tokenizer, ex.input_ids),
                        "Target": clean_decode(tokenizer, ex.labels),
                    },
                    "prediction": clean_decode(tokenizer, generated[0].tolist()),
                }
            )
        return out
