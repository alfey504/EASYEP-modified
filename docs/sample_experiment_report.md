# Sample experiment report: switch-base-8, first sparse layer, 2 experts

**Status: pipeline validation run.** Calibration used the 48-pair synthetic smoke-test file (`data/calibration.sample.jsonl`), not a real calibration corpus. The numbers show what the pipeline measures and how. They are not a research result.

## Setup

| | |
|---|---|
| Model | `google/switch-base-8` (619.3M parameters, 12 sparse layers × 8 experts, top-1, capacity 64) |
| Device / dtype | Apple M2, MPS, float32 (torch 2.15, transformers 5.22) |
| Calibration | 48 `human`/`response` pairs, 43 train / 5 validation (seed 42); 442 encoder + 1193 decoder tokens |
| Evaluation | 232 C4 `en/validation` documents accepted out of 100,000 scanned (0.23%), 10,750 target tokens |
| Span corruption | noise density 0.15, mean span 3.0, seed 42 |
| Pruning | `PRUNE_LAYERS=first` → `encoder.block.1`, `PRUNE_EXPERTS_PER_LAYER=2` |
| Bias training | layer MSE, Adam, lr 0.001, 100 steps, batch 1 (the `.env` defaults) |

## Architecture checks

* `inspect_model.py` found all 12 sparse layers by inspecting the loaded model (6 encoder, 6 decoder).
* Probe: layer output equals $\sum g\,E(x)$ with relative error 0 in all 12 layers, so the gate is applied once, by the parent, outside the expert.
* After replacement, the same probe confirms that replaced slots return $b_e$ and that the router's gate scales them once.

## EASY-EP ranking (`encoder.block.1`)

| Expert | Rank | $I_e$ | Share | Tokens | Routing freq. | Avg gate | Avg ‖E‖ | Avg $s_t$ | Replaced |
|---|---|---|---|---|---|---|---|---|---|
| 5 | 0 | 197.6 | 0.29% | 30 | 6.8% | 0.633 | 252.1 | 0.031 | ✔ |
| 0 | 1 | 203.9 | 0.30% | 11 | 2.5% | 0.739 | 263.7 | 0.069 | ✔ |
| 6 | 2 | 339.9 | 0.50% | 67 | 15.2% | 0.468 | 289.0 | 0.028 | |
| 3 | 3 | 421.4 | 0.61% | 32 | 7.2% | 0.619 | 229.7 | 0.047 | |
| 2 | 4 | 1206.7 | 1.76% | 50 | 11.3% | 0.627 | 430.8 | 0.086 | |
| 7 | 5 | 1211.2 | 1.77% | 132 | 29.9% | 0.550 | 531.1 | 0.034 | |
| 4 | 6 | 1809.1 | 2.64% | 102 | 23.1% | 0.565 | 425.2 | 0.062 | |
| 1 | 7 | 63165.5 | 92.1% | 18 | 4.1% | 0.882 | 5375.1 | 0.310 | |

Selection differs per layer (see `expert_scores.csv`). In 8 of 12 layers a single expert holds more than 50% of the importance, and in several layers more than 90%. This comes from the outlier activations typical of T5-style models. Expert 0 was chosen even though it had only 11 calibration tokens.

## Replacement

* Removed 9,437,184 parameters (2 × `wi`+`wo`), added 1,536 bias parameters: a net reduction of 1.52% of the model.
* The replaced and exported checkpoints contain no `wi`/`wo` tensors for the replaced experts, and the export check found none.

Bias initialisation (training split):

| Expert | Tokens | ‖b‖ | Weighted rel. residual $\frac{\sum s g^2\lVert E-b\rVert^2}{\sum s g^2\lVert E\rVert^2}$ | Gate-weighted rel. residual |
|---|---|---|---|---|
| 0 | 10 | 223.1 | 0.552 | 0.668 |
| 5 | 26 | 162.4 | 0.709 | 0.749 |

Even on the tokens the constant was fitted to, it explains only about 25–45% of the weighted output energy.

## Bias training (layer MSE on calibration)

| Setting | Train MSE (init → final) | Val MSE (init → final) | Closed-form optimum (train / val) | Skipped steps* |
|---|---|---|---|---|
| lr 0.001, 100 steps (`.env` default) | 2429.5 → 2428.6 | 1806.5 → 1804.8 | 2313.7 / 1557.6 | 43 |
| lr 1.0, 300 steps (variant `switch8-easyep-first-lr1`) | 2429.5 → 2333.6 | 1806.5 → 1595.6 | 2313.7 / 1557.6 | 125 |

\*Batches with no token routed to a replaced expert: the loss does not depend on the biases there, so the update is skipped.

With lr 0.001 the biases moved by ‖Δb‖ = 0.18 and 0.27 (against ‖b‖ ≈ 160–220), so default training barely changes anything. Training time was 1.3 s on MPS, with 1,536 trainable parameters.

## Evaluation on C4 support subset (identical examples and masks)

Metric: mean per-token denoising loss (teacher-forced cross-entropy of span-corruption targets). This is not perplexity.

| Stage | Token loss | Δ | Rel. | Top-1 agreement | Examples worse | Median Δ per example |
|---|---|---|---|---|---|---|
| Original | 1.7053 | – | – | – | – | – |
| Replaced, before training | 1.7153 | +0.0100 | +0.58% | 96.13% | 60.8% | +0.0088 |
| Replaced, after training (lr 0.001) | 1.7153 | +0.0099 | +0.58% | 96.13% | 60.8% | +0.0088 |
| Replaced, after training (lr 1.0 variant) | 1.7134 | +0.0081 | +0.47% | 96.29% | 59.5% | +0.0072 |

Layer reconstruction of `encoder.block.1` on C4. The original weights of only the replaced experts were loaded for this measurement.

| Stage | Tokens routed to replaced experts | Rel. error (all tokens) | Rel. error (replaced tokens) |
|---|---|---|---|
| Before training | 10,133 / 47,678 (21.3%) | 0.0022 | 1.016 |
| After training (lr 0.001) | same | 0.0022 | 1.016 |
| After training (lr 1.0) | same | 0.0019 | 0.914 |

Runtime for each evaluation stage was about 35 s for 232 documents. Peak MPS allocation was about 2.6 GB and process RSS under 0.7 GB; only one full model was ever loaded.

## Findings and limitations

1. **What EASY-EP identified did not hold up as "replaceable by a constant".** On C4, the constant replacement of experts 0 and 5 is no better than outputting zero on the tokens routed to them (relative error 1.016 before training). Training at a suitable learning rate improves this only to 0.914.
2. **Why the full model barely changes.** The two experts carry a small share of that layer's output energy (relative error over all tokens: 0.2%), and only one layer was pruned. The +0.58% loss increase therefore says little about whether the constant is a good approximation. The layer probe is the more informative measurement.
3. **Domain shift between calibration and evaluation.** The replaced experts received 9% of calibration encoder tokens but 21% of C4 tokens. A bias fitted on 36 tokens of short Q&A does not transfer to web documents.
4. **Default training hyperparameters are mis-scaled.** Adam with lr 0.001 moves each coordinate by at most about 0.1 in 100 steps, while the biases have coordinates of order 1–10. The closed-form optimum of the same objective is available, and the trainer reports it.
5. **The $s_t$ weighting is not the layer-MSE optimum.** The initialisation minimises the contribution-weighted error; the layer-MSE optimum is the $g^2$-weighted mean. On the calibration validation split the closed form gives 1558 against 1806 for the $s_t$-weighted initialisation, so for this objective the $s_t$ weighting costs reconstruction quality.
6. **Small, synthetic calibration set.** Expert 0's bias was fitted on 10 tokens. Real conclusions need the real calibration corpus (`CALIBRATION_MAX_SAMPLES=128` or more).
7. The C4 subset is heuristic: preview shows marketing pages that mention "customer service".

## Suggested next experiments

* Re-run with the real calibration data and `C4_MAX_SCANNED≈300000`, which should reach 500 documents at the observed 0.23% acceptance rate.
* Compare the selection against controls: random experts, lowest routing frequency, and highest relative residual (the "most constant" experts). The pipeline already records everything needed.
* Initialise biases with the closed-form layer-MSE optimum and compare it with the $s_t$-weighted initialisation.
* `PRUNE_LAYERS=all` with 1–2 experts per layer, to measure error accumulation across layers.
