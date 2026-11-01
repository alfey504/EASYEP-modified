# EASY-EP expert selection with gate-weighted bias replacement

Research code that combines

* **EASY-EP** ([paper](https://arxiv.org/abs/2504.06792), [code](https://github.com/RUCAIBox/EASYEP)) – output-aware expert importance × expert-level token contribution – to pick the least important experts of a Mixture-of-Experts layer, with
* **STEP-style replacement** ([adaptation](https://github.com/alfey504/STEP-adapted)) – instead of deleting those experts, each is replaced by a trainable constant vector $b_e$ that the *unchanged* router still scales by its gate: $\hat y_{e,t} = g_{e,t}\, b_e$.

Research question: *can EASY-EP identify experts that can be replaced with gate-weighted bias vectors while preserving the original model's behaviour?* The pipeline measures this directly (layer reconstruction error and end-to-end loss) rather than assuming it.

Supported models (each is a separate plugin, see [Model plugins](#model-plugins)):

| Plugin | Model | Architecture | Profile |
|---|---|---|---|
| `switch_transformers` | `google/switch-base-8` | encoder–decoder, 12 MoE layers × 8 experts, top-1, capacity 64 | `configs/switch-base-8.env` |
| `olmoe` | `allenai/OLMoE-1B-7B-0125` (the model used in STEP) | decoder-only, 16 MoE layers × 64 experts, top-8 | `configs/olmoe-1b-7b-0125.env` |

switch-base-8 runs on an Apple-silicon Mac (MPS, 16 GB). OLMoE-1B-7B (6.9B parameters, ~14 GB in 16-bit) needs ≥ 24 GB of GPU memory. It can be split over **Kaggle's 2× T4** (see [Running on Kaggle](#running-on-kaggle-gpu-t4-x2)).

See [docs/sample_experiment_report.md](docs/sample_experiment_report.md) for results of a first run.

---

## Installation

```bash
uv sync                 # Python >= 3.10, torch, transformers 5, accelerate, datasets, safetensors, ...
cp .env.example .env    # then edit .env
```

On hosted notebooks (Kaggle, Colab) use `pip install -e .`, which reuses the preinstalled CUDA build of torch. Tested with Python 3.10 (torch 2.14, transformers 5.20) and Python 3.14 (torch 2.15, transformers 5.22).

Put your calibration data at `CALIBRATION_DATASET_PATH` (default `./data/calibration.jsonl`), one JSON object per line:

```json
{"human": "How do I reset my password?", "response": "Navigate to account settings and select reset password."}
```

`data/calibration.sample.jsonl` contains 48 short synthetic support Q&A pairs used only for smoke-testing the pipeline (`--set CALIBRATION_DATASET_PATH=./data/calibration.sample.jsonl`). It is not meant for real experiments.

## Workflow

Every step is a separate script that reads `.env`, validates its input artifacts and writes its own outputs to `EXPERIMENT_DIR/EXPERIMENT_NAME/`. Configuration is layered: `.env` < `--profile configs/<model>.env` (or `MOE_PROFILE`) < environment variables < `--set KEY=VALUE` (repeatable), e.g. `--set PRUNE_EXPERTS_PER_LAYER=3`. Each profile sets its own `EXPERIMENT_NAME` and `PRUNED_MODEL_PATH`, so the models' results never mix.

```bash
# same commands for both models; pick the profile
uv run python scripts/collect_stats.py --profile configs/switch-base-8.env
uv run python scripts/collect_stats.py --profile configs/olmoe-1b-7b-0125.env
```

Full sequence:

```bash
uv run python scripts/inspect_model.py                         # architecture report + gate-application probe
uv run python scripts/build_c4_support_eval.py --preview 10    # inspect what the filter accepts
uv run python scripts/build_c4_support_eval.py                 # write the C4 support evaluation subset
uv run python scripts/collect_stats.py                         # EASY-EP statistics (original model)
uv run python scripts/score_experts.py                         # per-layer ranking + selection
uv run python scripts/replace_experts.py                       # bias init + replacement + replaced checkpoint
uv run python scripts/evaluate.py --stage before_training      # evaluates the original model first if needed
uv run python scripts/train_biases.py                          # layer-MSE training of the biases only
uv run python scripts/evaluate.py --stage after_training
uv run python scripts/export_model.py                          # compact checkpoint + reload verification
```

`evaluate.py --stage original` evaluates only the baseline. Stages never hold two full models in memory: the baseline is evaluated and released before a pruned model is loaded.

### Artifacts

| Path | Produced by | Content |
|---|---|---|
| `outputs/experiments/<name>/architecture_report.json` | inspect_model | layers, routers, dims, gate-consistency probe |
| `data/evaluation/c4_support_eval.jsonl` + `.manifest.json` | build_c4_support_eval | documents, filter scores, counts, rejection reasons, config |
| `outputs/stats/<name>/expert_stats.pt` + manifest | collect_stats | float64 running sums per layer/expert |
| `outputs/experiments/<name>/expert_scores.{json,csv}`, `expert_selection.json` | score_experts | ranking and selected experts |
| `outputs/experiments/<name>/replaced-model/`, `bias_initialization.json` | replace_experts | replaced checkpoint, per-expert init diagnostics |
| `outputs/cache/<name>/layer_reference/` | train_biases | cached original layer inputs/outputs |
| `outputs/experiments/<name>/trained-model/`, `training_report.json` | train_biases | trained checkpoint, loss curves |
| `outputs/experiments/<name>/evaluation/` | evaluate | per-stage metrics, per-example losses, `comparison.{json,md}` |
| `PRUNED_MODEL_PATH` | export_model | final checkpoint, manifest, metadata, `export_verification.json` |
| `outputs/experiments/<name>/config/`, `experiment.log` | every script | config snapshot per run, log |

---

## Architecture overview

```
configs/                    per-model profiles layered on top of .env
src/moe_pruning/
  config.py                 .env + profile loading, validation, --set overrides
  models/                   MODEL PLUGINS (everything model-specific)
    __init__.py             registry: ModelPlugin, register_plugin, get_plugin, MODEL_PLUGINS
    switch_transformers/    adapter.py (Switch routing, capacity, per-expert modules)
    olmoe/                  adapter.py (top-k router, masking), experts.py (PrunedFusedExperts)
  tasks/                    objectives chosen by plugins
    seq2seq_denoising.py    human->encoder, response->decoder; T5 span-corruption evaluation
    causal_lm.py            prompt+response; next-token evaluation (+ perplexity)
  model/
    architecture.py         MoEAdapter interface, MoELayerInfo, MoEForwardRecord, layer selection
    instrumentation.py      gate-consistency checks, record collector
    loader.py               plugin-based model/tokenizer loading, CheckpointReader (original weights)
  data/                     calibration JSONL, collators (right padding, per-stack masks), span corruption, C4 filter
  scoring/                  easyep.py (formulas), statistics.py (streaming accumulators, top-k)
  pruning/                  bias_expert.py, zero_expert.py, bias_initialization.py, expert_selection.py,
                            replacement.py, manifest.py
  training/bias_trainer.py  reference capture, layer-MSE training, closed-form reference (top-1)
  evaluation/               evaluator (+ exact reference probe), metrics
  serialization/checkpoint.py  custom safetensors format, load_pruned_model
  utils/                    device, logging, reproducibility, paths, cli
```

The scoring, initialisation, replacement, training, evaluation and serialization code is model-agnostic. It talks to a model only through its plugin:

* **`MoEAdapter`**: discover layers, get the router and experts, route (with or without a pruning mask), run one MoE layer, replace experts, instrument a forward pass, check that removed weights are gone, and load an original expert from disk. A forward pass yields, per layer, an `MoEForwardRecord` with the residual stream, MoE input and output, the top-k expert indices and gates, and the *un-gated* expert outputs.
* **`TaskObjective`**: turns calibration records and evaluation documents into model inputs, provides per-stack real-token masks, and defines the loss, predictions and qualitative generations.

## Model plugins

A plugin bundles an adapter, a task and the HF auto class. The plugin is selected automatically from the model's `config.model_type`, so switching models only means changing `HF_MODEL_ID` or `MODEL_PATH` (or the profile).

**Adding a model** (e.g. Mixtral or Qwen-MoE):

1. Write an adapter subclassing `moe_pruning.model.architecture.MoEAdapter`. Use `models/olmoe/adapter.py` as the template for transformers-5 fused experts (`gate_up_proj`/`down_proj` 3D tensors) and `models/switch_transformers/adapter.py` for per-expert modules.
2. Pick a task (`CausalLMTask` or `Seq2SeqDenoisingTask`) or write one.
3. Register the plugin:

```python
from transformers import AutoModelForCausalLM
from moe_pruning.models import ModelPlugin, register_plugin
from moe_pruning.tasks.causal_lm import CausalLMTask
from my_package.mixtral_adapter import MixtralAdapter

register_plugin(ModelPlugin(
    name="mixtral", model_types=("mixtral",),
    adapter_factory=MixtralAdapter, task_factory=CausalLMTask,
    auto_model_class=AutoModelForCausalLM, description="Mixtral sparse MoE",
))
```

4. Make it discoverable in one of three ways: put it in `src/moe_pruning/models/<name>/` (imported automatically), list the module in `MODEL_PLUGINS=my_package.mixtral_plugin` in `.env`, or expose it as a Python entry point in the `moe_pruning.models` group.
5. Add a profile `configs/<model>.env`, run `scripts/inspect_model.py` (its probe verifies that the layer output equals $\sum_k g_k E_k(x)$), and copy `tests/test_olmoe.py` for the new model's tests.

## How OLMoE is handled

Verified against `transformers/models/olmoe/modeling_olmoe.py` (5.22) and the hub checkpoint:

* `OlmoeDecoderLayer`: $\text{out} = h + \text{MoE}(\text{RMSNorm}(h))$, where $h$ is the residual stream after attention. Every layer is sparse (`model.layers.{i}.mlp`, layer ids `layers.{i}`).
* Router (`OlmoeTopKRouter`): $p=\text{softmax}(Wx)$ in float32, top-8 of 64. The top-k weights are **not** renormalised (`norm_topk_prob=False`). There is no capacity limit, so no tokens are dropped.
* Gate application: `OlmoeExperts.forward` multiplies each expert output by its top-k weight exactly once, so `BiasExpert` again returns $b_e$ only. The probe in `inspect_model.py` checks this.
* **Fused experts:** transformers 5 stores all 64 experts of a layer as two 3D tensors (`gate_up_proj` [64, 2·1024, 2048], `down_proj` [64, 2048, 1024]), while the hub checkpoint stores them per expert (`experts.{e}.gate_proj/up_proj/down_proj`) and transformers fuses them on load as `[gate; up]`. A single expert can't be swapped as a module, so the adapter replaces the layer's container with `PrunedFusedExperts`. That container:
  * keeps **only the retained experts' rows**, so the removed weights really leave memory and the checkpoint, and records their original ids;
  * serves removed experts through their replacement module (`replaced.{e}`: `BiasExpert` or `ZeroExpert`), keyed by the *original* router index.
* EASY-EP with top-k: $c_{e,t}=g_{e,t}\lVert E_e(x_t)\rVert$ for each of the token's 8 experts, and $s_t$ is computed from the full routed MoE output, exactly as in EASY-EP's DeepSeek code. The fused kernel doesn't expose per-expert outputs, so the instrumentation recomputes them from the same weights and verifies $\sum_k g_k E_k(x) = y$.
* `reroute` with top-k: tokens choose their top-8 among the remaining experts, and the gates are the original probabilities (renormalised only if `norm_topk_prob`).
* Calibration (`CausalLMTask`): the prompt is `"User: {human}\nAssistant: "`, or the tokenizer's chat template if it has one, followed by the response and EOS. All tokens feed the MoE statistics; only the response tokens are labelled.
* Evaluation: next-token loss on the C4 documents (first `MAX_SOURCE_LENGTH` tokens) plus perplexity, which is meaningful for a causal LM.
* Precision: the profile uses bfloat16. Bias vectors are kept in float32 and cast to the hidden-state dtype, so small updates aren't lost. The gate-consistency checks use a bfloat16-appropriate tolerance.
* Bias training: with top-k a token can mix several replaced experts, which couples their biases. The trainer therefore reports no closed-form optimum for OLMoE.

## How Switch Transformer routing is handled

Verified by reading `transformers/models/switch_transformers/modeling_switch_transformers.py` (transformers 5.22) and by a numerical probe in `inspect_model.py`:

* 12 sparse layers: encoder and decoder blocks 1, 3, 5, 7, 9, 11 (`*.block.{i}.layer.{-1}.mlp`), 8 experts each, $d_{model}=768$, $d_{ff}=3072$, ReLU.
* `SwitchTransformersLayerFF`: $\text{out} = h + \text{MoE}(\text{RMSNorm}(h))$.
* Router: $p=\text{softmax}(W x)$ in float32, top-1 expert, and a **per-sequence expert capacity** of 64 tokens: tokens beyond capacity are dropped (zero MoE output, residual only).
* **Gate application:** `SwitchTransformersExperts.forward` computes `expert(x) * routing_weights`. The parent applies $g_t$ (the top-1 probability) exactly once, so `BiasExpert` returns $b_e$ only. The probe confirms $y_t = g_t E_{e(t)}(x_t)$ with a relative error of 0 on the real model, and again after replacement with $E=b$.
* Padding: the HF router routes padding tokens and lets them consume capacity. All batches are **right-padded**, so padding never takes capacity from real tokens. Padding is excluded from every statistic, loss and metric (the test suite checks that a padded batch gives exactly the statistics of the unpadded sequences).
* Encoder layers are fed the `human` text and decoder layers the teacher-forced `response`. Statistics are collected from both.

## EASY-EP scoring

Verified against the official code (`pruning/model_new.py`, `pruning/expert_selection.py`):

$$c_{e,t} = g_{e,t}\,\lVert E_e(x_t)\rVert_2 \qquad\text{(output-aware importance)}$$

$$s_t = \max\bigl(1-\cos(h_t,\; h_t + y_t),\,0\bigr),\quad y_t=\textstyle\sum_e g_{e,t}E_e(x_t) \qquad\text{(token contribution, ``simibr``)}$$

$$I_e = \sum_{\text{samples}}\sum_{t} c_{e,t}\, s_t$$

EASY-EP keeps the top-$k$ experts by $I_e$; here the experts with the **lowest** $I_e$ are replaced, independently in each configured layer (ties broken by expert id).

Switch-specific adaptations (all documented in `scoring/easyep.py`):

| EASY-EP (DeepSeek) | Here (Switch) |
|---|---|
| top-k sigmoid scores, normalised, × `route_scale` | top-1 softmax probability — in both cases the weight that actually scales the output |
| $s_t$ uses routed experts only (shared experts excluded) | Switch has no shared experts; MoE output = routed output |
| $h_t$ = residual before `ffn_norm` | $h_t$ = residual entering `SwitchTransformersLayerFF` |
| no capacity limit | capacity-dropped tokens have no expert and $s_t=0$; counted and reported |
| 25 longest calibration samples | all `CALIBRATION_MAX_SAMPLES` samples |
| unpadded single sequences | padding masked out |

Attention scores (STEP's token selection) are **not** used anywhere.

Unobserved experts (no calibration tokens) are marked `observed=false`, ranked last and excluded from selection unless `PRUNE_UNOBSERVED_POLICY=allow`.

## Replacement methods

`REPLACEMENT_METHOD` selects what happens to the experts EASY-EP selects. In all three cases the original expert weights are removed from memory and from the checkpoint, the router's weights are unchanged, and the selection is identical, so the methods can be compared directly.

| Method | Slot module | Tokens routed to a removed expert | Extra parameters | Trainable |
|---|---|---|---|---|
| `bias` | `BiasExpert` | output $g\,b_e$ (gate applied once by the parent) | $d$ per expert | yes (`train_biases.py`) |
| `drop` | `ZeroExpert` | zero MoE output; residual only | 0 | no |
| `reroute` | `ZeroExpert` (errors if reached) + routing mask | sent to their best remaining expert $e'$, output $p_{e'}E_{e'}(x)$ | 0 | no |

`reroute` is a forward hook on the router (`SwitchTransformersAdapter.set_routing_mask`) that repeats HF's top-1 and capacity assignment with the removed experts excluded. The gate is the new expert's *original* softmax probability (no renormalisation). Tokens whose best expert is kept therefore get exactly the same expert and gate. Rerouted tokens do consume capacity, so they can push other tokens over an expert's 64-token limit. The mask is listed in the manifest and reinstalled by `load_pruned_model`.

For `drop` and `reroute` there is no training stage: evaluate with `--stage before_training`. `export_model.py` exports the replaced model by default.

The evaluation probe recomputes the exact original layer output (original routing, all original experts) for every token, so its layer error is comparable across methods.

## Bias initialisation

For the training-split tokens $T_e$ routed to expert $e$:

$$b_e=\frac{\sum_{t\in T_e}s_t\,g_{e,t}^{2}\,E_e(h_t)}{\sum_{t\in T_e}s_t\,g_{e,t}^{2}}$$

**Derivation.** Replacing $E_e$ by $b$ changes the layer output for token $t$ by $g_{e,t}(E_e(h_t)-b)$. Weighting the squared error by the token's contribution gives

$$J(b)=\sum_{t\in T_e}s_t\lVert g_{e,t}E_e(h_t)-g_{e,t}b\rVert^2=\sum_{t\in T_e}s_t g_{e,t}^2\lVert E_e(h_t)-b\rVert^2 ,$$

a convex quadratic with $\nabla_b J=-2\sum_t s_t g_{e,t}^2(E_e(h_t)-b)=0$, which gives the weighted mean above.

Implementation: running float64 sums $\sum s g^2 E$ and $\sum s g^2$ (plus $\sum s g^2\lVert E\rVert^2$ for residual diagnostics), using the gate that actually multiplies the expert output, only for real tokens actually dispatched to $e$. If $\sum s g^2=0$ the code falls back to the gate-weighted mean, then to the plain mean, and logs it. A never-activated expert raises an error, unless `PRUNE_UNOBSERVED_POLICY=allow`, in which case it gets a zero vector, logged and flagged in the manifest.

`bias_initialization.json` reports, per expert, the relative residual $\sum w\lVert E-b\rVert^2/\sum w\lVert E\rVert^2$, which measures how well *any* constant can represent that expert.

## Bias training (layer-level reconstruction)

$$\mathcal{L}_{MSE}=\frac{1}{N}\sum_{i=1}^{N}\lVert F_{\text{original}}(H_i)-F_{\text{replaced}}(H_i)\rVert_2^2$$

over the real tokens of the calibration training split. $H_i$ (MoE input after the pre-norm) and $F_{\text{original}}(H_i)$ are captured from the original model and cached on disk. The replaced layer is then run on exactly the same $H_i$, so the router, routing decisions and retained experts are identical in both computations. Only `BiasExpert.bias` parameters are trainable; this is verified, and routers and retained experts are checked to be bit-identical after training. The model stays in `eval()` mode (Switch experts contain dropout). The calibration validation split is used for monitoring; C4 is never used.

Because routing is fixed for cached inputs, $\mathcal L_{MSE}$ has the closed-form minimiser $b^*_e=\sum g^2E/\sum g^2$, the same as the initialisation **without** the $s_t$ weight. The trainer reports this optimum's train and validation loss as a reference for the optimiser.

Layer-level reconstruction is a **local** objective: when several layers are pruned, upstream changes alter downstream hidden states and routing, which it does not capture.

## Evaluation

* C4 subset (`build_c4_support_eval.py`): streams `allenai/c4` `en` / `validation`, matches support phrases case-insensitively on word boundaries (generic words such as "help" or "service" on their own never match), matches URL tokens (`/help/`, `support.`), optionally filters on allowed domains, enforces length bounds, and removes duplicates by normalised-text hash and URL. A document needs at least `C4_MIN_KEYWORD_MATCHES` distinct phrases, or a URL match plus one phrase. Each record keeps `text`, `url` and `timestamp`, plus `filter_score`, the matched keywords and the C4 row index.
* **Limitation:** C4 is a general web crawl. The result is a heuristically filtered subset, not a professionally curated support corpus. Preview shows that marketing pages mentioning "customer service" pass the filter.
* Deterministic T5 span corruption (NumPy port of the T5 reference `random_spans_noise_mask`), with sentinel `<extra_id_N>` tokens, noise density 0.15, mean span length 3 and a per-example RNG seeded with `(EVAL_SEED, example_id)`. Corrupted examples are cached and fingerprinted. Every stage must match the baseline fingerprint and example ids.
* Metric: **mean per-token denoising loss** (teacher-forced cross-entropy of the span targets). It is *not* autoregressive perplexity. Also reported: Δ and % versus the original, per-example deltas, teacher-forced top-1 agreement with the original, and greedy sentinel-span predictions for a few examples. switch-base-8 is not instruction-tuned, so these predictions are not support answers.
* **Layer reconstruction on C4:** the pruned stages load only the original weights of the replaced experts (memory-mapped, a few MB) and measure $\lVert y_{\text{original}}-y_{\text{replaced}}\rVert^2$ of each pruned layer on its actual input during evaluation.

## Checkpoint format

```
pruned-model/
  config.json, generation_config.json, tokenizer files
  model.safetensors          retained experts, non-MoE weights, routers, trained biases; no replaced-expert weights
  pruning_manifest.json      replaced experts, state prefixes, bias-init diagnostics, training summary, tied aliases
  experiment_config.json     full config + stats/selection/training/evaluation metadata
  export_verification.json   absence check + reload logit difference
  README.md
```

```python
from moe_pruning.serialization.checkpoint import load_pruned_model
model, tokenizer, manifest = load_pruned_model("outputs/pruned-model", device="mps")
```

The loader builds the architecture from `config.json` on the meta device, installs `BiasExpert` modules in the slots listed in the manifest, loads the tensors (restoring tied aliases) with strict key validation, and re-ties the weights. Missing original-expert keys are accepted only for the manifest's experts, because only those slots no longer expect them; any other missing or unexpected key is an error. **Plain `from_pretrained()` does not work for this format**: it would re-create the removed experts with random weights.

## Configuration (`.env`)

| Variable | Meaning |
|---|---|
| `MODEL_PATH` | Local model directory. Empty: download `HF_MODEL_ID`. A non-empty invalid path is an error (no silent download). |
| `HF_MODEL_ID`, `HF_REVISION` | Hugging Face model id and optional revision. |
| `MODEL_DTYPE` | `float32` (default, recommended on MPS), `float16`, `bfloat16` (OLMoE profile). |
| `MODEL_PLUGINS` | Extra plugin modules to import (comma-separated); built-in plugins are always available. |
| `DEVICE` | `auto` (CUDA → MPS → CPU) or explicit `cuda`/`mps`/`cpu`. An unavailable explicit device is an error. |
| `DEVICE_MAP` | Empty = whole model on `DEVICE`. `auto`, `balanced`, `balanced_low_0`, `sequential` or a JSON object = split over several GPUs with accelerate. |
| `MAX_MEMORY` | Per-device cap for `DEVICE_MAP`, e.g. `0:13GiB,1:13GiB` (leaves room for activations). |
| `HF_CACHE_DIR` | Download cache for Hugging Face models (empty = default). |
| `EXPERTS_IMPLEMENTATION` | transformers 5 experts kernel for fused-expert models (empty = default; `eager`, `grouped_mm`, `batched_mm`). |
| `CALIBRATION_DATASET_PATH` | JSONL with `human`/`response`. |
| `EVALUATION_DATASET_PATH` | Output of the C4 builder / input of evaluation. |
| `CALIBRATION_MAX_SAMPLES`, `CALIBRATION_BATCH_SIZE` | Calibration size and batch size. |
| `MAX_SOURCE_LENGTH`, `MAX_TARGET_LENGTH` | Token limits. Seq2seq: encoder / decoder. Causal: prompt / response (calibration); `MAX_SOURCE_LENGTH` is also the evaluation document length. |
| `C4_DATASET_ID`, `C4_CONFIG`, `C4_SPLIT` | Streamed C4 source. |
| `C4_DOMAIN` | Label recorded in the manifest. |
| `C4_MAX_SAMPLES`, `C4_MAX_SCANNED` | Target accepted documents; scan limit. |
| `C4_MIN_TEXT_LENGTH`, `C4_MAX_TEXT_LENGTH` | Character length bounds. |
| `C4_MIN_KEYWORD_MATCHES` | Distinct support phrases required (URL match lowers this to 1). |
| `C4_SUPPORT_KEYWORDS`, `C4_URL_KEYWORDS`, `C4_ALLOWED_DOMAINS` | Comma-separated filter lists. |
| `PRUNING_METHOD` | `easyep`. |
| `PRUNE_EXPERTS_PER_LAYER` | Experts replaced in each selected layer. |
| `PRUNE_LAYERS` | `first`, `all`, or comma-separated ids such as `encoder.block.1,decoder.block.3`. |
| `PRUNE_UNOBSERVED_POLICY` | `exclude` (default) or `allow`. |
| `REPLACEMENT_METHOD` | `bias` (default), `drop` or `reroute`; see *Replacement methods*. |
| `BIAS_INITIALIZATION` | `gate_token_weighted_mean`. |
| `BIAS_TRAINING_OBJECTIVE` | `layer_mse`. |
| `BIAS_LEARNING_RATE`, `BIAS_TRAINING_STEPS`, `BIAS_BATCH_SIZE` | Adam settings for bias training. |
| `BIAS_VALIDATION_FRACTION` | Held-out share of calibration examples (seeded split). |
| `BIAS_EVAL_INTERVAL` | Steps between validation evaluations. |
| `EVAL_MAX_SAMPLES`, `EVAL_BATCH_SIZE` | Evaluation size / batch size. |
| `EVAL_NOISE_DENSITY`, `EVAL_MEAN_SPAN_LENGTH`, `EVAL_SEED` | Span corruption settings. |
| `EVAL_QUALITATIVE_SAMPLES` | Examples with generated predictions. |
| `SEED` | Global seed (also the calibration split seed). |
| `EXPERIMENT_NAME` | Sub-directory for all artifacts of one experiment. |
| `PRUNED_MODEL_PATH`, `EXPERIMENT_DIR`, `STATS_DIR`, `CACHE_DIR` | Output locations. |
| `LOG_LEVEL` | Python logging level. |
| `CHECKPOINT_FORMAT` | `full` = self-contained checkpoints; `delta` = only bias vectors + manifest, rebuilt from the base model on load. |

## Tests

```bash
uv run pytest            # 122 tests; unit tests use tiny random Switch and OLMoE models on CPU
uv run pytest -m integration   # tests on the real google/switch-base-8 (skipped if not cached)
```

They cover:
- **Config and device selection:** CUDA → MPS → CPU priority; no silent fallback.
- **Adapter:** layer discovery, mapping of recorded expert outputs to tokens, capacity drops.
- **Data:** JSONL validation, right-padding masks, span-corruption determinism and invertibility, C4 filter rules.
- **EASY-EP:** formulas, padding exclusion, per-layer independence, unobserved experts, determinism.
- **Bias initialisation:** the analytic formula, optimality (zero gradient), gate and contribution influence, zero denominators.
- **BiasExpert:** shape, dtype, gradients, and that the gate is applied exactly once inside a real Switch layer.
- **Replacement:** router unchanged, retained experts unchanged, parameters removed.
- **Training:** only biases trainable, frozen weights unchanged, loss definition, closed form.
- **Serialization:** removed weights absent, strict loader, output equality.
- **Replacement methods:** drop zeroes only the removed experts' tokens; reroute sends them to the best remaining expert with the original gate, leaves every other token unchanged and respects capacity; checkpoints round-trip for every method.
- **Evaluation:** the reference probe matches the true layer error for every method.
- **OLMoE:** discovery, top-k instrumentation, expert views versus the raw weights, original experts loaded from both the hub and fused layouts, causal tokenization and loss, top-k statistics with padding, fused-expert compaction for bias/drop/reroute, repeated compaction, checkpoint round trips (compact tensor shapes, routing mask restored), the exact probe, and bias training.
- **Plugins:** built-in discovery, loading an external plugin module, conflict detection, profile layering.
- **Placement and Kaggle support:** a model split over two real devices (CPU + accelerator, with accelerate-style input hooks): forward pass, instrumentation, replacement modules on each layer's device, the probe, labels on the logits' device, and bias training on the layer's device. Also: accelerate dispatch, delta checkpoint round trips for both models and all methods, base-model mismatch detection, float16 overflow errors, and parsing of the placement settings and Kaggle profiles.

## Reproducibility notes

Seeds are fixed (`SEED`, `EVAL_SEED`, per-example corruption RNG, seeded calibration split). Every run stores its config snapshot, the environment (torch/transformers versions, device) and memory use. MPS and CUDA do not guarantee bit-identical results for every kernel (reductions, `index_add_`); differences are at float32 rounding level. Deterministic algorithms are not forced, because some operations are unsupported on MPS. No `torch.compile`, FlashAttention or CUDA-only operations are used, and nothing needed a CPU fallback on MPS.

## Known limitations

* A constant vector cannot represent an input-dependent expert. The initialisation diagnostics show that a constant leaves 55–75% of the (weighted) output energy of the replaced experts unexplained (see the sample report).
* Calibration and evaluation domains differ (short Q&A vs. C4 web pages), and routing frequencies shift between them. An expert that is unimportant on the calibration set can receive many more tokens on the evaluation set.
* T5/Switch activations contain extreme outliers. In many layers one expert holds more than 90% of the EASY-EP importance, so the ranking among the remaining experts rests on small differences.
* Expert capacity is fixed at 64 tokens per sequence in the HF config. At `MAX_SOURCE_LENGTH=256` some tokens can be dropped, both in the original and in the pruned model.
* The C4 subset is heuristic (see above).
* The layer-MSE objective is local, and with Adam at `BIAS_LEARNING_RATE=0.001` the biases (‖b‖ ≈ 10²) barely move in 100 steps. Use a learning rate on the scale of the bias coordinates, or the closed-form optimum.

## Running OLMoE

OLMoE support is implemented and tested on a tiny random OLMoE with the real `allenai/OLMoE-1B-7B-0125` tokenizer: unit tests, plus an end-to-end run of every script on MPS in bfloat16. It has **not** been run on the real 6.9B model, which needs about 14 GB in bfloat16 plus a 28 GB download, more than the development Mac has. On a GPU machine:

```bash
P="--profile configs/olmoe-1b-7b-0125.env"
uv run python scripts/inspect_model.py $P
uv run python scripts/collect_stats.py $P
uv run python scripts/score_experts.py $P          # 16 of 64 experts per layer by default
uv run python scripts/replace_experts.py $P
uv run python scripts/evaluate.py --stage before_training $P
uv run python scripts/train_biases.py $P            # bias method only
uv run python scripts/evaluate.py --stage after_training $P
uv run python scripts/export_model.py $P
```

Notes:
* Set `BIAS_LEARNING_RATE` relative to the bias magnitudes (`bias_norm` in `bias_initialization.json`): Adam moves each coordinate by about the learning rate per step.
* The evaluation probe reads the removed experts' original weights lazily from the downloaded shards, so it needs no second model copy.
* `PrunedFusedExperts` loops over the routed experts in Python, as transformers' eager experts implementation does. It is correct but slower than `grouped_mm`.

## Running on Kaggle (GPU T4 x2)

Use the notebook [`kaggle/easyep_kaggle.ipynb`](kaggle/easyep_kaggle.ipynb) (branch `kaggle_adapted`) and the profiles in `configs/kaggle/`:

| Profile | Model | Precision | Placement |
|---|---|---|---|
| `configs/kaggle/olmoe-1b-7b-t4x2.env` | OLMoE-1B-7B-0125 | float16 | `DEVICE_MAP=auto`, 13 GiB per T4 |
| `configs/kaggle/switch-base-8-t4.env` | switch-base-8 | float32 | one T4 |

How the Kaggle limits are handled:

* **2 × 15 GB GPUs:** `DEVICE_MAP=auto` places the decoder layers on both T4s with accelerate. Every stage follows each layer's device: replacement modules, reference experts and bias training run on the GPU that holds the layer, inputs go to the embedding's GPU, and labels go to the logits' GPU.
* **No bfloat16 on T4:** the OLMoE profile uses float16. OLMoE was trained in bfloat16, so inf/NaN are checked during inspection, statistics and evaluation, with an error explaining the cause. switch-base-8 stays in float32, because T5-family models overflow in float16.
* **20 GB `/kaggle/working`:**
  * The Hugging Face download goes to `/kaggle/temp/hf`.
  * The layer-reference cache goes to `/kaggle/temp/cache`, stored in the model's 16-bit dtype.
  * `CHECKPOINT_FORMAT=delta` saves only the bias vectors and the manifest per stage: a few MB instead of 14 GB. A delta checkpoint is rebuilt by loading the base model and re-applying the recorded replacement.
  * `scripts/convert_model.py` writes a 14 GB float16 copy of OLMoE once. Saved as a Kaggle Dataset and used through `MODEL_PATH`, it avoids the 28 GB float32 download in later sessions.
* **12 h sessions:** each stage writes its results under `/kaggle/working/outputs`, so a new session can continue from the last finished stage. Re-attach the same inputs and settings.
* **Private repository:** clone with a GitHub token stored as a Kaggle Secret, e.g. `from kaggle_secrets import UserSecretsClient; token = UserSecretsClient().get_secret("GITHUB_TOKEN")`, and `https://{token}@github.com/...`.

Not yet run on real Kaggle hardware. The multi-device code paths are tested by splitting a tiny OLMoE over CPU + MPS. The float16 + accelerate + delta-checkpoint pipeline was run end to end on a tiny OLMoE. If float16 overflows on the real OLMoE, the alternatives are a GPU with bfloat16 support (A100/L4) or float32 on larger GPUs.
