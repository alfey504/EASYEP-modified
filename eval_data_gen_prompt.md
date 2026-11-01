# Evaluation data generation prompt

Use the prompt below with a capable LLM to generate a synthetic support-domain **evaluation** set, as an alternative or complement to the C4 subset from `scripts/build_c4_support_eval.py`.

**Keep evaluation independent of calibration:**
- Evaluation must measure behaviour on text the pruning never saw. Generate it in a **separate session**, ideally with a **different model** than the one used for the calibration data.
- Use **different fictional companies**. Run the overlap check at the bottom before using the file.
- Prefer the C4 builder (real web text) as the main benchmark. Synthetic documents are cleaner and more on-topic, but they are written by an LLM, which can make the results look better than they would on real text. Report which set you used.

**What the pipeline needs:**
- The evaluation set is made of **documents**, not question/answer pairs. Each line is `{"text": ..., "url": ..., "timestamp": ...}`; only `text` is required.
- switch-base-8 blanks out ~15% of each document and scores how well the model fills the gaps. OLMoE scores next-token prediction on the document. Both use up to `MAX_SOURCE_LENGTH` tokens (256 for Switch, 512 for OLMoE), so documents of **200–450 words** are used almost completely.

**Steps:**
- Ask for **20 documents per request** and change `{BATCH_TYPES}` and `{COMPANY_SEED}` each time. 10–25 batches give 200–500 documents (the pipeline uses the first `EVAL_MAX_SAMPLES`, default 500).
- Save as `data/evaluation/synthetic_support_eval.jsonl` and point `EVALUATION_DATASET_PATH` at it, so the C4 file is not overwritten.

---

## Prompt

```text
You are writing evaluation documents for a research project that measures how much a language
model's ability to model customer-support text degrades after compression. The documents must read
like real public support content found on company websites: help-center articles, FAQ pages,
troubleshooting guides, policy pages and community support threads.

Produce exactly 20 records in JSON Lines format: one JSON object per line, no other text, no
Markdown code fences, no numbering, no blank lines.

Each object has exactly three string fields:
  "text":      the full document as plain text
  "url":       a plausible but fictional page address on an example.com subdomain,
               e.g. "https://help.northwind-gear.example.com/returns/start-a-return"
  "timestamp": an ISO 8601 date-time between 2018-01-01 and 2024-12-31, e.g. "2021-06-14T09:32:00Z"

Document types for this batch: {BATCH_TYPES}
Invent companies and products for this batch based on: {COMPANY_SEED}
(Use 4-6 different fictional companies in the batch; never use real brands or real domains.)

Requirements:

1. Content
   - Each document covers one concrete support topic in depth: steps, conditions, exceptions,
     time limits, fees, error messages, settings names, contact options.
   - Include the natural structure of web support pages written as plain text: a title line,
     short paragraphs, numbered steps written inline ("Step 1: ... Step 2: ..."), FAQ-style
     "Q: ... A: ..." pairs, notes and warnings. Use " | " or sentences instead of tables.
   - Content must be internally consistent and plausible (realistic policies, prices, durations).
   - Do NOT write customer/agent chat transcripts as the main format (that is what the calibration
     data looks like). At most 3 documents per batch may be community forum threads with a question
     and several replies.

2. Length and variety
   - Each "text" is 200 to 450 words (count carefully; never below 180 or above 500).
   - Vary topic, company, product type (software, apps, online retail, telecom, banking, consumer
     electronics, travel, utilities, healthcare scheduling, education platforms), tone (formal
     policy language, friendly how-to, terse technical notes) and structure.
   - No two documents may cover the same situation for the same kind of product.
   - Avoid boilerplate repeated across documents (e.g. the same closing sentence).

3. Safety and privacy
   - No real people, companies, products, URLs, phone numbers or email addresses. Use fictional
     names, example.com addresses and 555-01xx phone numbers.
   - No medical, legal or financial advice beyond ordinary account and product support.

4. Output validity
   - Every line must be valid JSON that parses on its own.
   - Escape double quotes inside text as \". No raw line breaks inside strings: separate paragraphs
     with a single space or with " \n " written as the two characters backslash and n.
   - Exactly the fields "text", "url", "timestamp"; all non-empty strings.

Output the 20 JSON lines now.
```

### Suggested `{BATCH_TYPES}` (rotate)

1. `help-center articles about account sign-in, password reset and two-factor authentication`
2. `refund, return and exchange policy pages`
3. `billing FAQ pages: invoices, payment methods, failed payments, currency and taxes`
4. `troubleshooting guides for app crashes, sync errors and numbered error codes`
5. `device setup guides: routers, printers, smart-home devices, wearables`
6. `shipping and order tracking FAQ pages, delivery exceptions, customs`
7. `warranty terms and repair-request procedures`
8. `subscription management pages: plans, upgrades, cancellations, trials, renewals`
9. `privacy and data pages: data export, account deletion, cookies, security alerts`
10. `service status and outage notices, known-issue articles with workarounds`
11. `community forum threads where users and moderators solve a support problem (max 3 per batch, rest help articles)`
12. `knowledge-base articles for business customers: admin settings, user roles, API keys, SSO`

### Suggested `{COMPANY_SEED}` (change every batch)

`outdoor gear retailer, budget airline, regional bank`, `smart thermostat maker, music streaming app, internet provider`, `meal-kit delivery, laptop manufacturer, insurance portal`, `fitness tracker, online bookstore, mobile carrier`, `photo-editing software, furniture shop, energy utility`. Add your own combinations as needed.

---

## Checks after generating

```bash
# 1. Every line parses, has text, and documents have the intended length
uv run python -c "
from pathlib import Path
from moe_pruning.data.c4_support import load_eval_documents
docs = load_eval_documents(Path('data/evaluation/synthetic_support_eval.jsonl'), 10_000)
words = sorted(len(d.text.split()) for d in docs)
print(len(docs), 'documents | words min/median/max:', words[0], words[len(words)//2], words[-1])
"

# 2. Duplicates and overlap with the calibration data (8-word shingles; both should be ~0)
uv run python -c "
import json
def shingles(t, n=8):
    w = t.lower().split(); return {' '.join(w[i:i+n]) for i in range(len(w)-n+1)}
ev = [json.loads(l)['text'] for l in open('data/evaluation/synthetic_support_eval.jsonl') if l.strip()]
cal = [json.loads(l) for l in open('data/calibration.jsonl') if l.strip()]
cal_sh = set().union(*(shingles(r['human'] + ' ' + r['response']) for r in cal))
dup = len(ev) - len({t.strip().lower() for t in ev})
overlap = sum(1 for t in ev if shingles(t) & cal_sh)
print(dup, 'duplicate documents |', overlap, 'documents sharing an 8-word sequence with calibration')
"
```

Then use it:

```bash
uv run python scripts/evaluate.py --stage before_training \
  --set EVALUATION_DATASET_PATH=./data/evaluation/synthetic_support_eval.jsonl
```

(On Kaggle: upload the file into the `easyep-data` dataset and set `EVALUATION_DATASET_PATH` in the notebook's `EXTRA`.) Changing the evaluation file re-evaluates the original model automatically, because the cached examples and baseline are tied to the file's checksum.
