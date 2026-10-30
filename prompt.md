# Project: EASY-EP Expert Pruning with Gate-Weighted Bias Approximation

## Role

Act as a senior Machine Learning Engineer with experience in PyTorch, Mixture-of-Experts (MoE) architectures, model compression, and Hugging Face Transformers.

Build a clean, modular, well-documented Python project that implements EASY-EP-based expert selection combined with gate-weighted, token-importance-weighted bias-vector replacement.

The implementation must be easy to understand, experiment with, and extend. Avoid putting the entire implementation into a few large Python files. Keep the model-specific logic separate from the pruning algorithms.

Work incrementally. First inspect the model architecture and explain the implementation plan. Then implement and test each stage independently before moving to the next. Do not skip testing or implement placeholder methods that appear functional but do not perform the actual operations.

---

# 1. Research background and objective

Read and understand the following resources before implementing the algorithms:

1. EASY-EP:
   https://github.com/RUCAIBox/EASYEP

2. My STEP adaptation:
   https://github.com/alfey504/STEP-adapted

3. Original STEP:
   https://github.com/HikvisionResearch/STEP

4. Initial model:
   https://huggingface.co/google/switch-base-8

The objective is to experiment with combining two approaches:

**EASY-EP: Expert selection**

Use EASY-EP's output-aware expert importance calculation and expert-level token contribution calculation to identify experts that contribute relatively little to the model's output.

**STEP-inspired expert replacement**

Instead of simply deleting the selected experts, replace them with lightweight, trainable bias vectors.

The original router must remain operational. When a token is routed to a replaced expert, the model should return that expert's bias vector, which must be scaled by the original router's gate score exactly once.

The proposed replacement is:

Original MoE contribution:

\[
y_{e,t}=g_{e,t}E_e(h_t)
\]

Replacement contribution:

\[
\hat y_{e,t}=g_{e,t}b_e
\]

Where:

- \(h_t\) is the token's hidden representation.
- \(E_e\) is the original expert.
- \(g_{e,t}\) is the actual router probability used to scale the expert output.
- \(b_e\) is the trainable replacement vector.

The primary research question is:

Can EASY-EP identify experts that can be replaced with gate-weighted bias vectors while preserving the original model's behaviour?

Important: Do not assume that every low-importance expert can be accurately approximated by a constant vector. Measure the reconstruction error and report the limitations.

## 2. Hardware and initial model

The first implementation will run on:

- Apple M2 MacBook Air
- 16 GB unified memory
- PyTorch
- Apple MPS acceleration when available

Use:

`google/switch-base-8`

This is a relatively small Switch Transformer encoder-decoder MoE model with eight experts.

The architecture is different from OLMoE, which will be considered in a later stage on a machine with more memory.

Do not hardcode the architecture around OLMoE or DeepSeek.

Implement a model-specific adapter for Switch Transformer, while keeping the pruning, scoring, bias initialization, and training algorithms reusable.

Before implementing pruning, inspect the actual Hugging Face model architecture and identify:

- All sparse MoE layers.
- Encoder and decoder MoE layers.
- Router modules.
- Number of experts in each layer.
- Hidden dimensions.
- Expert input and output dimensions.
- Where router probabilities are applied to expert outputs.
- How tokens are assigned to experts.

Do not assume module names without inspecting the loaded model.

Generate a model architecture report that can be inspected before any pruning operation.

---

# 3. Project and environment configuration

This must be a `uv` Python project.

Use `pyproject.toml` and maintain a `uv.lock` file.

Use `python-dotenv` to load configuration from a root `.env` file.

All important configuration should be centralized. Avoid scattering hardcoded values throughout the codebase.

Create:

- `.env`
- `.env.example`
- `.gitignore`
- `pyproject.toml`
- `uv.lock`
- `README.md`

Never commit the actual `.env` file.

## 3.1 Model configuration

The model must support both local model loading and Hugging Face downloads.

Use the following behaviour:

- If `MODEL_PATH` contains a valid local directory, load the model from that directory.
- If `MODEL_PATH` is empty or unset, download/load the model identified by `HF_MODEL_ID`.
- If a non-empty local path is invalid, raise an informative error instead of silently downloading another model.

Initial Hugging Face model:

`google/switch-base-8`

## 3.2 Proposed .env configuration

Create an `.env.example` with the following configuration:

```dotenv
# ==========================================
# MODEL CONFIGURATION
# ==========================================

MODEL_PATH=
HF_MODEL_ID=google/switch-base-8
HF_REVISION=

MODEL_DTYPE=float32

# ==========================================
# DEVICE CONFIGURATION
# ==========================================

DEVICE=auto

# ==========================================
# DATASET CONFIGURATION
# ==========================================

CALIBRATION_DATASET_PATH=./data/calibration.jsonl
EVALUATION_DATASET_PATH=./data/evaluation/c4_support_eval.jsonl

CALIBRATION_MAX_SAMPLES=128
CALIBRATION_BATCH_SIZE=1

MAX_SOURCE_LENGTH=256
MAX_TARGET_LENGTH=128

# ==========================================
# C4 EVALUATION DATASET
# ==========================================

C4_DATASET_ID=allenai/c4
C4_CONFIG=en
C4_SPLIT=validation

C4_DOMAIN=support

C4_MAX_SAMPLES=500
C4_MAX_SCANNED=100000
C4_MIN_TEXT_LENGTH=200
C4_MAX_TEXT_LENGTH=8000
C4_MIN_KEYWORD_MATCHES=2

C4_SUPPORT_KEYWORDS=customer support,technical support,customer service,help center,help centre,troubleshooting,frequently asked questions,FAQ,refund policy,return policy,warranty claim,order tracking,billing support,account recovery,password reset,error code,support ticket,knowledge base,help desk

C4_URL_KEYWORDS=support,help,faq,knowledge-base,customer-service

C4_ALLOWED_DOMAINS=

# ==========================================
# EXPERT PRUNING
# ==========================================

PRUNING_METHOD=easyep

PRUNE_EXPERTS_PER_LAYER=2

# Use "first" for initial testing.
# Use "all" to prune every detected sparse layer.
# Also support comma-separated layer identifiers.
PRUNE_LAYERS=first

# ==========================================
# BIAS INITIALIZATION
# ==========================================

BIAS_INITIALIZATION=gate_token_weighted_mean

# ==========================================
# BIAS TRAINING
# ==========================================

BIAS_TRAINING_OBJECTIVE=layer_mse

BIAS_LEARNING_RATE=0.001
BIAS_TRAINING_STEPS=100
BIAS_BATCH_SIZE=1

BIAS_VALIDATION_FRACTION=0.1

# ==========================================
# EVALUATION
# ==========================================

EVAL_MAX_SAMPLES=500
EVAL_BATCH_SIZE=1

EVAL_NOISE_DENSITY=0.15
EVAL_MEAN_SPAN_LENGTH=3.0
EVAL_SEED=42

# ==========================================
# REPRODUCIBILITY
# ==========================================

SEED=42

# ==========================================
# OUTPUT PATHS
# ==========================================

PRUNED_MODEL_PATH=./outputs/pruned-model

EXPERIMENT_DIR=./outputs/experiments
STATS_DIR=./outputs/stats
CACHE_DIR=./outputs/cache
```

All these values must be read from the environment and validated.

Allow optional configuration overrides through command-line arguments, but `.env` should be the primary configuration source.

`PRUNE_EXPERTS_PER_LAYER` must be adjustable without modifying Python code.

For the initial experiment, two experts per selected layer should be replaced.

`PRUNE_LAYERS=first` should select the first detected sparse MoE layer. Support `all` and explicit layer identifiers as well.

---

# 4. Device selection

Implement a centralized device selection utility.

When:

```dotenv
DEVICE=auto
```

Use this priority:

1. CUDA if available.
2. MPS if available.
3. CPU otherwise.

If the user explicitly selects `cuda`, `mps`, or `cpu`, validate that the requested device is available.

Do not silently switch to another device when an explicitly requested device is unavailable.

Print the selected device and model data type at startup.

Use FP32 as the initial default because of compatibility and numerical stability.

The implementation must be conservative with memory because it will run on a 16 GB Mac.

Avoid keeping multiple complete copies of the model in memory unnecessarily.

Use small batches and configurable sequence lengths.

Avoid requiring `torch.compile`, CUDA-only operations, FlashAttention, or other dependencies that are not compatible with the initial hardware.

If an operation is unsupported by MPS, provide a documented fallback where appropriate.

---

# 5. Dataset handling

Implement separate dataset loaders for calibration and evaluation.

Do not mix their formats.

## 5.1 Calibration dataset

The calibration dataset is a JSONL file.

Each line follows this format:

```json
{
  "human": "How do I reset my password?",
  "response": "Navigate to account settings and select reset password."
}
```

Requirements:

- Read the file line by line.
- Validate JSON syntax.
- Require both `human` and `response`.
- Skip or report malformed and empty records.
- Provide useful error messages.
- Support a configurable maximum number of samples.
- Use the `human` field as the encoder input.
- Use the `response` field as the decoder target.
- Use teacher forcing when running the encoder-decoder model.

Do not treat this dataset as plain text concatenation.

For Switch Transformer, construct the appropriate encoder inputs and decoder labels using the Hugging Face tokenizer.

Padding tokens must be ignored correctly when computing training and evaluation losses.

## 5.2 C4 evaluation dataset generation

Create a separate script:

`scripts/build_c4_support_eval.py`

The purpose is to generate a support-domain evaluation dataset from C4.

Use:

- Dataset: `allenai/c4`
- Configuration: `en`
- Split: `validation`
- Streaming enabled.

Do not download the entire C4 dataset onto the Mac.

C4 contains raw web-document text, not question-and-answer pairs.

Therefore, do not fabricate `human` and `response` fields.

The generated JSONL should preserve the original document information:

```json
{
  "text": "Original C4 document text...",
  "url": "https://example.com/help/article",
  "timestamp": "2019-04-25T12:57:54Z"
}
```

The domain-specific filtering should use:

1. Text keyword and phrase matching.
2. URL keyword matching.
3. Optional allowed-domain filtering.
4. Configurable minimum keyword-match thresholds.
5. Minimum and maximum document lengths.

Use the support-related keywords configured in `.env`.

Avoid accepting documents merely because they contain generic terms such as "help" or "service".

Implement sensible matching rules and provide a filter score so that the quality of selected documents can be inspected.

The script must:

- Stream the C4 validation split.
- Filter documents.
- Remove duplicate records.
- Use deterministic sampling.
- Stop when the requested sample count is reached or the scan limit is reached.
- Save the selected records to `EVALUATION_DATASET_PATH`.
- Report scanned, accepted, rejected, and duplicate counts.
- Report the most common rejection reasons.
- Save a generation manifest containing dataset information, filtering configuration, sample count, and random seed.

Provide a preview mode that displays selected documents for manual inspection before generating the complete evaluation dataset.

If insufficient documents are found, provide a useful diagnostic and allow the user to increase `C4_MAX_SCANNED`.

Important research limitation:

C4 is a general web crawl. The support-domain dataset produced by this script is a heuristically filtered subset, not a professionally curated support corpus. Document this limitation clearly in the README and experiment reports.

## 5.3 Evaluation objective

The initial model is a T5-style encoder-decoder model trained with span corruption.

Therefore, evaluate it using a T5-compatible denoising objective, not ordinary next-token generation.

Implement deterministic span corruption on the C4 evaluation text.

Use the tokenizer's extra sentinel tokens.

Initial settings:

- Noise density: 0.15
- Mean noise span length: 3.0
- Random seed: 42

The corrupted document becomes the encoder input.

The removed spans, including their appropriate sentinel tokens, become the decoder target.

Use the same deterministic corruption for every comparison.

The baseline and pruned model must evaluate exactly the same examples with exactly the same masks.

Report mean per-token denoising loss. Do not incorrectly label this metric as standard autoregressive perplexity.

---

# 6. Model inspection

Create:

`scripts/inspect_model.py`

This must:

1. Load the configured model.
2. Inspect its modules.
3. Identify all sparse MoE layers.
4. Identify the router module in each sparse layer.
5. Identify all expert modules.
6. Determine the model's actual number of experts.
7. Identify input and output hidden dimensions.
8. Identify where the gate score is applied.
9. Report encoder versus decoder locations.
10. Save a machine-readable architecture report.

Do not assume all MoE implementations use identical module names or forward methods.

Build a Switch Transformer-specific adapter to handle its actual implementation.

The architecture adapter should expose reusable methods such as:

- Discover MoE layers.
- Get experts in a layer.
- Get router probabilities.
- Collect expert inputs and outputs.
- Replace an expert.
- Execute an individual MoE layer.
- Inspect the hidden dimensions.

Add tests to verify that the architecture adapter works with the actual downloaded model.

---

# 7. EASY-EP expert importance calculation

Read the original EASY-EP implementation and paper carefully.

Implement the actual published expert importance and token contribution methodology, rather than inventing a simplified scoring method.

Relevant conceptual components:

### A. Output-aware expert importance

The importance of an expert should account for:

- Router gate scores.
- Expert output magnitudes.
- Token contributions.

The conceptual output-aware component is:

\[
c_{e,t}=g_{e,t}\|E_e(h_t)\|_2
\]

### B. Expert-level token contribution

Implement the token contribution calculation used in EASY-EP.

The original methodology uses representation changes to estimate how much the expert contributes to a token's representation.

Verify the precise equations and implementation details against the original EASY-EP paper and repository.

Do not silently substitute STEP's attention-based token selection.

### C. Combined importance

Use the published EASY-EP combination to obtain an importance score for each expert.

Conceptually:

\[
I_e=\sum_t c_{e,t}s_t
\]

Where \(s_t\) represents the expert-level token contribution.

The final implementation must match the verified EASY-EP methodology as closely as the Switch Transformer architecture permits.

Document any architectural adaptations necessary for Switch Transformer, especially its top-1 routing behaviour.

IMPORTANT:

EASY-EP's original implementation may make assumptions about a different MoE architecture. Adapt its logic to Switch Transformer without changing the mathematical objective unnecessarily.

If an exact operation cannot be reproduced, explicitly document the difference.

Do not use attention scores as a substitute for EASY-EP token contribution.

## 7.1 Statistics collection

Create:

`scripts/collect_stats.py`

Collect the information required for EASY-EP scoring.

For each MoE layer and each expert, collect:

- Number of assigned tokens.
- Router probabilities.
- Expert input representations.
- Expert output representations.
- Token contribution scores.
- Output-aware importance statistics.

For efficiency, use PyTorch hooks or a dedicated architecture instrumentation layer.

The implementation must correctly associate each expert output with the corresponding token and routing score.

Take care with Switch Transformer's sparse expert execution: experts may receive only the tokens assigned to them.

Handle padding tokens correctly and never include padding in the importance calculation.

For an encoder-decoder model, collect information from both encoder and decoder MoE layers.

Use:

- `human` for encoder input.
- `response` for teacher-forced decoder target.

The statistics should be collected using the original, unmodified model.

Use `torch.no_grad()` where gradients are not needed.

Avoid storing all intermediate activations in RAM. Prefer streaming accumulators or disk-backed caches where appropriate.

---

# 8. Expert scoring and selection

Create:

`scripts/score_experts.py`

For each sparse MoE layer:

1. Load collected statistics.
2. Calculate EASY-EP expert importance.
3. Generate a ranking.
4. Identify the least important experts.
5. Select the configured number of experts for replacement.

Use:

```dotenv
PRUNE_EXPERTS_PER_LAYER=2
```

The selected experts must be those with the lowest EASY-EP importance.

Selection must be independently performed for each configured MoE layer.

Do not select the same expert IDs across all layers unless the scores indicate that they should be selected.

The scoring should be deterministic.

Generate:

- `expert_scores.json`
- `expert_scores.csv`
- `expert_selection.json`

Include:

- Layer name.
- Expert ID.
- Importance score.
- Routing frequency.
- Average gate score.
- Average expert output magnitude.
- Average token contribution.
- Whether the expert was selected for replacement.

If an expert has no observations in the calibration dataset, mark it as unobserved.

Do not automatically assume an unobserved expert is unimportant. Exclude unobserved experts from pruning by default unless explicitly configured otherwise.

---

# 9. Gate-conditioned, token-weighted bias initialization

This is the central custom component of the project.

Create a reusable class:

`BiasExpert`

The original expert:

\[
E_e(h_t)
\]

Must be replaced with a trainable parameter:

\[
b_e\in\mathbb{R}^{d_{\text{model}}}
\]

For each token routed to the replaced expert:

\[
\hat E_e(h_t)=b_e
\]

The router must remain unchanged.

The resulting contribution must be:

\[
\hat y_{e,t}=g_{e,t}b_e
\]

The gate multiplication must happen exactly once.

Inspect the Hugging Face Switch implementation to determine where the router probability is applied.

If the parent MoE module already multiplies the expert result by the router probability, the replacement module must return only the bias vector.

Do not multiply the gate a second time inside `BiasExpert`.

Write a unit test that confirms the output scales correctly with different router probabilities.

## 9.1 Bias initialization formula

Do not initialize the bias using an ordinary unweighted average.

Use the following gate-conditioned, token-weighted initialization:

\[
\boxed{
b_e=
\frac{
\sum_{t\in T_e}s_tg_{e,t}^{2}E_e(h_t)
}{
\sum_{t\in T_e}s_tg_{e,t}^{2}
}
}
\]

Where:

- \(T_e\) contains tokens routed to expert \(e\).
- \(s_t\) is the EASY-EP token contribution score.
- \(g_{e,t}\) is the actual post-routing gate probability.
- \(E_e(h_t)\) is the original expert output.

This is derived by minimizing the weighted reconstruction objective:

\[
\min_{b_e}
\sum_{t\in T_e}
s_tg_{e,t}^{2}
\left\|E_e(h_t)-b_e\right\|_2^2
\]

Important implementation details:

1. Use the actual gate probability that scales the expert contribution, not raw router logits.
2. Use the same token contribution definition as the verified EASY-EP implementation.
3. Use only tokens actually assigned to that expert.
4. Accumulate statistics using adequate numerical precision.
5. Handle zero denominators safely.
6. Log when a fallback initialization is used.

If an expert has never been activated, do not invent an estimated mean output. Mark it as unobserved and follow the configured pruning policy.

The initialized bias must have the expert's output dimension.

## 9.2 Replacement implementation

Create:

`src/moe_pruning/pruning/bias_expert.py`

The replacement should be a proper `torch.nn.Module`.

It must:

- Store the bias vector as `torch.nn.Parameter`.
- Return a broadcasted copy of the bias for the incoming tokens.
- Preserve expected input and output shapes.
- Preserve device and dtype.
- Support gradients.
- Be compatible with the parent Switch MoE router implementation.

Avoid architecture-specific logic inside the bias class.

Create a separate replacement utility that installs the bias modules into the appropriate expert slots.

Preserve:

- Original router module.
- Original routing decisions.
- Original router weights.
- All non-pruned experts.

Do not change the number of expert slots or the router's indexing.

---

# 10. Applying pruning

Create:

`scripts/replace_experts.py`

This script should:

1. Load the original model.
2. Load the expert selection results.
3. Verify the selected experts.
4. Initialize replacement bias vectors using the weighted formula.
5. Replace selected expert modules.
6. Keep the original router untouched.
7. Report how many parameters were removed.
8. Save the intermediate replaced model.

The implementation should physically remove the original expert parameters from the in-memory model after replacement.

Do not simply place a gating mask around the original expert.

The replaced model should execute its normal forward pass using the bias modules.

Generate a pruning manifest containing the model ID, layer names, replaced expert IDs, bias initialization details, and configuration.

---

# 11. Model comparison before fine-tuning

Create:

`scripts/evaluate.py`

Evaluate three stages:

1. Original model.
2. Model after bias replacement and before bias fine-tuning.
3. Model after bias fine-tuning.

Use the same C4 support evaluation subset, deterministic corruption, and evaluation settings.

Calculate:

- Mean token-level denoising loss.
- Loss difference relative to the original model.
- Relative percentage increase in loss.
- Output reconstruction error where relevant.

Also provide a small qualitative comparison:

- Encoder input.
- Corrupted input.
- Original model prediction.
- Replaced model prediction.
- Post-training model prediction.

Since the model is not instruction-tuned, do not claim its predictions represent production-ready customer support answers.

Keep a copy of baseline metrics and evaluation configuration so subsequent experiments are comparable.

Avoid holding original and pruned copies of the full model in memory simultaneously on the Mac. Run baseline and pruned evaluation sequentially, saving their metrics and outputs.

---

# 12. Fine-tuning only the replacement bias vectors

Create:

`scripts/train_biases.py`

Use layer-level output reconstruction MSE as the initial optimization objective.

This is the chosen first implementation because it directly measures whether the replacement vectors preserve the output of the original MoE layer without requiring full-model distillation.

The process should be:

1. Load the original model.
2. Run calibration samples through the original model.
3. Capture the input hidden states and reference output of each selected sparse MoE layer.
4. Cache these reference tensors to disk or process them in small batches.
5. Install the bias replacements.
6. Freeze all original model parameters.
7. Freeze all router parameters.
8. Train only the replacement bias vectors.

The reconstruction objective:

\[
\mathcal{L}_{MSE}
=
\frac{1}{N}
\sum_{i=1}^{N}
\left\|
F_{\text{original}}(H_i)
-
F_{\text{replaced}}(H_i)
\right\|_2^2
\]

Where:

- \(H_i\) is an original hidden-state input to a sparse MoE layer.
- \(F_{\text{original}}\) is the original sparse MoE layer.
- \(F_{\text{replaced}}\) is the same layer after expert replacement.

Important:

- Compare the MoE layer output, not arbitrary intermediate tensors.
- Make sure the reference and replaced outputs correspond to the same hidden-state input.
- Use the original router and original retained experts in both computations.
- Preserve the router's gate weighting.
- Ignore padding positions where appropriate.
- Train all selected replacement biases in a layer together.
- Use a separate calibration validation subset for monitoring.
- Do not use the C4 evaluation dataset for bias optimization.

Use AdamW or Adam, with the learning rate controlled by `.env`.

Only the replacement bias parameters may have gradients.

Explicitly verify that:

```python
trainable_parameters
```

contains only `BiasExpert.bias` parameters.

All original model parameters must remain frozen.

Record:

- Initial reconstruction loss.
- Training reconstruction loss.
- Validation reconstruction loss.
- Final reconstruction loss.
- Number of trainable parameters.
- Training time.
- Device used.

Do not silently move the training process to CPU.

Document that layer-level reconstruction is a local approximation objective. It does not guarantee identical full-model outputs when multiple layers have been pruned because upstream changes can affect downstream hidden states and routing.

---

# 13. Model serialization and export

Create:

`scripts/export_model.py`

Save the final model to:

```dotenv
PRUNED_MODEL_PATH=./outputs/pruned-model
```

The exported model must contain:

- Original weights for retained experts.
- Original weights for non-MoE layers.
- Original router weights.
- Trained replacement bias vectors.
- Original model configuration.
- Tokenizer files.
- Pruning manifest.
- Experiment metadata.

Critically, the original weights of replaced experts must not be stored in the final checkpoint.

Do not merely save a normal Hugging Face checkpoint containing all original expert weights and a pruning mask.

Implement a custom reloadable checkpoint format.

For example:

```text
pruned-model/
├── config.json
├── tokenizer files...
├── model.safetensors
├── pruning_manifest.json
├── experiment_config.json
└── README.md
```

Implement:

```python
load_pruned_model(model_path, device)
```

The loader must:

1. Read the saved configuration.
2. Instantiate the appropriate Switch Transformer architecture.
3. Read the pruning manifest.
4. Replace the specified expert modules with `BiasExpert` modules before loading checkpoint weights.
5. Load retained weights and trained bias vectors.
6. Restore the original routers.
7. Validate missing and unexpected state-dict keys.
8. Return a working model that can perform inference.

Use `safetensors` for checkpoint weights.

Ensure that missing original expert parameters are permitted only when they correspond to the experts listed in the pruning manifest.

Provide a test confirming:

- The final checkpoint does not contain the removed expert weight tensors.
- The custom loader can reconstruct the model.
- The reloaded model produces the same outputs as the in-memory pruned model within numerical tolerance.

Document that the custom loader is required to load this checkpoint; ordinary Hugging Face `from_pretrained()` should not be claimed to work unless explicitly implemented and tested.

---

# 14. Suggested project structure

Use a modular structure similar to this:

```text
moe-pruning-experiment/
│
├── .env
├── .env.example
├── .gitignore
├── pyproject.toml
├── uv.lock
├── README.md
│
├── configs/
│   └── experiment.yaml
│
├── data/
│   ├── calibration.jsonl
│   └── evaluation/
│
├── src/
│   └── moe_pruning/
│       │
│       ├── __init__.py
│       ├── config.py
│       │
│       ├── model/
│       │   ├── loader.py
│       │   ├── architecture.py
│       │   ├── switch_adapter.py
│       │   └── instrumentation.py
│       │
│       ├── data/
│       │   ├── calibration.py
│       │   ├── c4_support.py
│       │   ├── collators.py
│       │   └── span_corruption.py
│       │
│       ├── scoring/
│       │   ├── easyep.py
│       │   └── statistics.py
│       │
│       ├── pruning/
│       │   ├── bias_expert.py
│       │   ├── bias_initialization.py
│       │   ├── expert_selection.py
│       │   └── replacement.py
│       │
│       ├── training/
│       │   └── bias_trainer.py
│       │
│       ├── evaluation/
│       │   ├── evaluator.py
│       │   └── metrics.py
│       │
│       ├── serialization/
│       │   └── checkpoint.py
│       │
│       └── utils/
│           ├── device.py
│           ├── logging.py
│           ├── reproducibility.py
│           └── paths.py
│
├── scripts/
│   ├── inspect_model.py
│   ├── build_c4_support_eval.py
│   ├── collect_stats.py
│   ├── score_experts.py
│   ├── replace_experts.py
│   ├── evaluate.py
│   ├── train_biases.py
│   └── export_model.py
│
├── tests/
│   ├── test_config.py
│   ├── test_data.py
│   ├── test_easyep.py
│   ├── test_bias_initialization.py
│   ├── test_bias_expert.py
│   ├── test_replacement.py
│   ├── test_training.py
│   └── test_serialization.py
│
└── outputs/
    ├── stats/
    ├── cache/
    ├── experiments/
    └── pruned-model/
```

You may improve this structure when justified, but preserve clear separation of concerns.

Avoid circular imports.

Avoid unnecessarily abstracting simple operations.

Keep the algorithm implementation readable enough that a researcher can modify the scoring and initialization methods independently.

---

# 15. Expected execution workflow

The complete workflow should be executable step by step.

First install the project:

```bash
uv sync
```

Inspect the model:

```bash
uv run python scripts/inspect_model.py
```

Generate the support-domain evaluation dataset:

```bash
uv run python scripts/build_c4_support_eval.py
```

Preview and manually inspect selected documents before relying on the dataset for experiments.

Collect calibration statistics:

```bash
uv run python scripts/collect_stats.py
```

Calculate EASY-EP scores:

```bash
uv run python scripts/score_experts.py
```

Replace the selected experts with initialized bias vectors:

```bash
uv run python scripts/replace_experts.py
```

Evaluate the model before bias training:

```bash
uv run python scripts/evaluate.py --stage before_training
```

Fine-tune the replacement bias vectors:

```bash
uv run python scripts/train_biases.py
```

Evaluate after bias training:

```bash
uv run python scripts/evaluate.py --stage after_training
```

Export the final model:

```bash
uv run python scripts/export_model.py
```

The steps must be independently executable.

Each stage should save its artifacts so the user does not need to repeat expensive previous stages unless its inputs or configuration have changed.

Validate input artifacts and configuration consistency before executing a stage.

---

# 16. Testing requirements

Write meaningful unit tests rather than relying solely on checking whether scripts execute.

At minimum, test:

### Bias initialization

- The weighted bias matches the analytical formula.
- The bias minimizes the weighted reconstruction objective.
- Higher gate probabilities have the expected greater influence.
- Higher token contribution scores have the expected greater influence.
- Zero denominators are handled safely.

### Bias replacement

- Original expert outputs are replaced.
- The router remains unchanged.
- Gate weighting is applied exactly once.
- Replaced experts return the correct tensor shape.
- Replaced modules preserve dtype and device.
- Retained experts produce unchanged outputs.

### EASY-EP scoring

- Scores are calculated independently for each layer.
- Padding tokens are excluded.
- Unobserved experts are handled correctly.
- Expert ranking is deterministic.

### Bias training

- Only replacement vectors are trainable.
- Router weights remain frozen.
- Retained expert weights remain frozen.
- Reconstruction loss is calculated correctly.
- Bias parameters actually change during optimization.

### Serialization

- Replaced original expert weights are absent from the saved checkpoint.
- The custom loader restores the architecture.
- Reconstructed outputs match the in-memory pruned model within a reasonable numerical tolerance.

### Device support

Test device detection and configuration validation.

Provide CPU-compatible unit tests even if the complete model experiment is run on MPS.

Do not require downloading the full model for every unit test.

---

# 17. Logging and experiment reproducibility

Every experiment must have its own output directory.

Save:

- Full experiment configuration.
- Model ID and revision.
- Device and dtype.
- Calibration dataset metadata.
- Evaluation dataset metadata.
- Random seeds.
- EASY-EP scores.
- Selected experts.
- Bias initialization statistics.
- Training losses.
- Evaluation metrics.
- Runtime and memory measurements.
- Final pruning manifest.

Use Python logging rather than scattered print statements.

Include progress bars for long-running calibration and evaluation loops.

Use fixed random seeds wherever possible.

When MPS or CUDA does not guarantee deterministic execution for a particular operation, document the limitation.

---

# 18. Implementation sequence and deliverables

Implement this in phases.

## Phase 1: Project foundation

Set up the `uv` project, dependencies, environment handling, device detection, logging, and model loader.

Deliverable: a working model inspection script.

Do not proceed until the model loads successfully on the available device and its MoE architecture is correctly identified.

## Phase 2: Dataset pipeline

Implement calibration JSONL loading and C4 support-domain streaming/filtering.

Deliverable: validated calibration records and a generated support evaluation dataset.

## Phase 3: EASY-EP instrumentation and scoring

Implement expert statistics collection and EASY-EP importance calculations.

Deliverable: per-layer expert rankings and inspectable statistics.

Compare the implementation with the official EASY-EP equations and document any Switch-specific adaptations.

## Phase 4: Bias replacement

Implement the gate- and token-weighted initialization, bias module, and expert replacement mechanism.

Deliverable: a working model with selected experts replaced by bias vectors.

Verify router weighting and retained expert behaviour.

## Phase 5: Before-training evaluation

Compare the original and replaced models using identical C4 span-corruption examples.

Deliverable: baseline and pre-training reconstruction metrics.

## Phase 6: Bias fine-tuning

Implement layer-level output reconstruction MSE.

Deliverable: trained bias vectors and a complete training report.

## Phase 7: Final evaluation and serialization

Evaluate the fine-tuned model and export a compact, reloadable checkpoint.

Deliverable: a model that can be independently reloaded and executed without retaining the original weights of pruned experts.

---

# 19. Important engineering constraints

1. Do not put the entire implementation into one Python script.
2. Do not hardcode paths, pruning ratios, sequence lengths, training steps, or device selection.
3. Keep model-specific code separate from pruning algorithms.
4. Never modify the original router as part of this experiment.
5. Never accidentally apply gate weighting twice.
6. Do not train retained expert weights.
7. Do not train the original router.
8. Do not use the C4 evaluation set to select experts or train biases.
9. Do not keep unnecessary full model copies in memory.
10. Do not silently replace EASY-EP scoring with STEP attention scoring.
11. Do not silently replace missing expert weights with zero outputs.
12. Do not claim that a saved model is reloadable until the custom loader has been tested.
13. Do not report denoising loss as standard autoregressive perplexity.
14. Keep all intermediate experiment artifacts reproducible.
15. Prefer straightforward, readable PyTorch implementations over premature optimization.

---

# 20. Final acceptance criteria

The project is considered complete when the following workflow works end to end:

1. Load `google/switch-base-8` from Hugging Face when `MODEL_PATH` is empty.
2. Load a local model when `MODEL_PATH` is configured.
3. Automatically detect CUDA, then MPS, then CPU when `DEVICE=auto`.
4. Load the user's `human` / `response` calibration JSONL.
5. Generate a support-domain evaluation dataset from C4 without downloading all of C4.
6. Inspect all supported MoE layers.
7. Collect the statistics required for EASY-EP.
8. Rank experts independently for each layer using EASY-EP.
9. Replace the configured number of low-importance experts with gate-weighted bias modules.
10. Confirm that the original expert parameters are removed from the replaced modules.
11. Compare the original and replaced model using the same evaluation examples.
12. Fine-tune only the replacement bias vectors using layer-level output reconstruction MSE.
13. Compare the original model, pre-training pruned model, and post-training pruned model.
14. Export a checkpoint without the removed expert weights.
15. Reload the exported model through the custom loader and successfully execute a forward pass.

At completion, provide:

- A concise architecture overview.
- The mathematical implementation of EASY-EP scoring.
- The mathematical derivation and implementation of bias initialization.
- Explanation of how Switch Transformer routing is handled.
- Installation and execution instructions.
- A description of all `.env` variables.
- Unit-test results.
- A sample experiment report.
- Known limitations and recommendations for moving to OLMoE.

**Start by inspecting the model architecture, the official EASY-EP implementation, and the existing project directory. Present the implementation plan and identify any compatibility issues before writing the main pruning code. Then proceed through the phases, implementing and validating each stage rather than generating the entire project in one untested pass.**