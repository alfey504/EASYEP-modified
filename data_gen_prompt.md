# Calibration data generation prompt

Use the prompt below with a capable LLM to generate `data/calibration.jsonl`.

- Ask for one batch of 50 records per request and change `{BATCH_TOPICS}` for each batch, so the set covers many support areas. 4–10 batches (200–500 records) is a good size; the pipeline uses the first `CALIBRATION_MAX_SAMPLES` (default 128).
- Concatenate the batches into `data/calibration.jsonl`, then run the checks at the bottom.

Note: synthetic data is a stand-in for real support conversations. If you have real (anonymised) tickets or chat logs, prefer those.

---

## Prompt

```text
You are generating calibration data for a research project that prunes a Mixture-of-Experts
language model. The data is used to measure which experts the model relies on when it processes
customer-support text, so it must look like realistic customer-support exchanges.

Produce exactly 50 records in JSON Lines format: one JSON object per line, no other text, no
Markdown code fences, no numbering, no blank lines.

Each object has exactly two string fields:
  "human":    what the customer writes to support
  "response": the support agent's reply

Example of the format (do not reuse these examples):
{"human": "How do I reset my password?", "response": "Go to Account Settings, choose Security and select Reset Password. We will email you a reset link that is valid for 30 minutes."}
{"human": "My order #48213 arrived with a cracked screen protector. Can I get a replacement?", "response": "I'm sorry it arrived damaged. Please reply with a photo of the item and the packaging, and we will ship a replacement at no cost within two business days."}

Topics for this batch: {BATCH_TOPICS}

Requirements:

1. Realism and domain
   - Typical help-desk, help-center and FAQ situations: accounts and login, passwords and
     two-factor authentication, billing and refunds, subscriptions, orders, shipping and tracking,
     returns and warranty claims, technical troubleshooting, error codes, device setup,
     connectivity, privacy and data requests, and complaints.
   - Spread the records across several fictional companies and product types (software, apps,
     online shops, telecom, banking, consumer electronics, travel, utilities). Invent company and
     product names; never use real brands.
   - Responses are helpful, specific and correct for the described situation: concrete steps,
     settings names, time frames or policies. They may ask a clarifying question when the request
     is genuinely ambiguous.

2. Variety (important: avoid repetitive templates)
   - Vary the customer's tone (neutral, polite, frustrated, confused, brief, detailed) and
     writing style (full sentences, short fragments, occasional typos or informal wording).
   - Vary the request type: how-to questions, problem reports, policy questions, complaints,
     follow-ups that reference an earlier ticket, requests with order or ticket numbers.
   - Vary length:
       human:    roughly 5 to 120 words (most 10-60)
       response: roughly 15 to 150 words (most 30-90)
     Never exceed 180 words in either field.
   - Do not start more than 3 records in a batch with the same first two words.
   - No two records may describe the same situation.

3. Safety and privacy
   - No real personal data. Use obviously fictional names, order numbers, emails
     (e.g. jane.doe@example.com) and phone numbers (e.g. 555-0142).
   - No real URLs; use example.com or describe the location in the app/website instead.
   - No offensive content, legal threats with real entities, or medical/financial advice
     beyond ordinary account support.

4. Output validity
   - Every line must be valid JSON that parses on its own.
   - Use double quotes for keys and strings; escape any double quotes inside text as \".
   - No line breaks inside a string (write paragraphs as one line; use "1) ... 2) ..." for steps).
   - Both fields must be non-empty strings. No additional fields.

Output the 50 JSON lines now.
```

### Suggested `{BATCH_TOPICS}` per batch

1. `account access: login problems, password reset, two-factor authentication, locked or hacked accounts, changing email or username`
2. `billing: double charges, failed payments, refunds, invoices, changing payment method, pricing questions`
3. `subscriptions: upgrades, downgrades, cancellations, pauses, free trials, renewals, prorated charges`
4. `e-commerce orders: tracking, delays, missing or wrong items, damaged deliveries, address changes, international shipping`
5. `returns and warranty: return policy, exchanges, warranty claims, repairs, proof of purchase, refund timing`
6. `technical troubleshooting for apps and websites: crashes, slow performance, error codes, sync issues, browser problems, updates`
7. `hardware and devices: setup, Wi-Fi and Bluetooth connectivity, printers, routers, battery, factory reset, firmware`
8. `telecom and internet service: outages, slow speeds, roaming, plan changes, SIM issues, moving service`
9. `privacy and data: data export, account deletion, marketing emails, cookie settings, security notifications`
10. `complaints and escalations: poor previous support, repeated issues, compensation requests, follow-ups on open tickets`

---

## Checks after generating

```bash
# 1. Every line parses and has both fields (invalid lines are reported with line numbers)
uv run python -c "
from pathlib import Path
from moe_pruning.data.calibration import load_calibration_records
records, report = load_calibration_records(Path('data/calibration.jsonl'))
print(report.to_dict())
"

# 2. Duplicates (should print 0)
uv run python -c "
import json
rows = [json.loads(l) for l in open('data/calibration.jsonl') if l.strip()]
print(len(rows) - len({r['human'].strip().lower() for r in rows}), 'duplicate questions')
"
```

Then look through a random sample by eye for repetitive phrasing, real brand names or anything that is not support-related.
