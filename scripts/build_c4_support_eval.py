"""Build the support-domain evaluation subset from C4 by streaming and filtering.

Only the streamed shards are read; C4 is never downloaded in full. Documents
are taken in the (deterministic) stream order of the configured split until
C4_MAX_SAMPLES are accepted or C4_MAX_SCANNED documents have been read.

Use ``--preview N`` to inspect the first N accepted documents without writing
anything.

Limitation: C4 is a general web crawl. The result is a heuristically filtered
subset, not a professionally curated support corpus.
"""

from __future__ import annotations

import json
import logging
import textwrap
from datetime import datetime, timezone

from datasets import load_dataset
from tqdm import tqdm

from moe_pruning.data.c4_support import Deduplicator, FilterStatistics, SupportDocumentFilter
from moe_pruning.utils.cli import bootstrap
from moe_pruning.utils.paths import write_json

logger = logging.getLogger("build_c4_support_eval")


def _add_args(parser):
    parser.add_argument("--preview", type=int, default=0, metavar="N",
                        help="Show the first N accepted documents and exit without writing.")
    parser.add_argument("--preview-chars", type=int, default=600)


def main() -> None:
    args, cfg, _ = bootstrap(__doc__, "build_c4_support_eval", _add_args)
    flt = SupportDocumentFilter(
        keywords=cfg.c4_support_keywords,
        url_keywords=cfg.c4_url_keywords,
        allowed_domains=cfg.c4_allowed_domains,
        min_keyword_matches=cfg.c4_min_keyword_matches,
        min_text_length=cfg.c4_min_text_length,
        max_text_length=cfg.c4_max_text_length,
    )
    target = args.preview if args.preview > 0 else cfg.c4_max_samples
    logger.info(
        "Streaming %s/%s split=%s; target=%d accepted, scan limit=%d",
        cfg.c4_dataset_id, cfg.c4_config, cfg.c4_split, target, cfg.c4_max_scanned,
    )
    stream = load_dataset(cfg.c4_dataset_id, cfg.c4_config, split=cfg.c4_split, streaming=True)

    stats = FilterStatistics()
    dedup = Deduplicator()
    selected: list[dict] = []
    progress = tqdm(total=cfg.c4_max_scanned, desc="scanning C4", unit="doc")
    for row_index, row in enumerate(stream):
        if stats.scanned >= cfg.c4_max_scanned or len(selected) >= target:
            break
        stats.scanned += 1
        progress.update(1)
        text, url = row["text"], row.get("url", "")
        decision = flt.evaluate(text, url)
        if not decision.accepted:
            stats.rejected += 1
            stats.rejection_reasons[decision.reason] += 1
            continue
        if dedup.is_duplicate(text, url):
            stats.duplicates += 1
            stats.rejection_reasons["duplicate"] += 1
            continue
        stats.accepted += 1
        selected.append(
            {
                "text": text,
                "url": url,
                "timestamp": str(row.get("timestamp", "")),
                "filter_score": decision.score,
                "matched_keywords": decision.matched_keywords,
                "keyword_occurrences": decision.keyword_occurrences,
                "url_keywords": decision.url_keywords,
                "c4_row_index": row_index,
            }
        )
        progress.set_postfix(accepted=len(selected))
    progress.close()

    logger.info(
        "Scanned %d | accepted %d | rejected %d | duplicates %d",
        stats.scanned, stats.accepted, stats.rejected, stats.duplicates,
    )
    logger.info("Most common rejection reasons:")
    for reason, count in stats.rejection_reasons.most_common(6):
        logger.info("  %-30s %d", reason, count)

    if args.preview > 0:
        for i, rec in enumerate(selected):
            print("=" * 100)
            print(f"[{i}] score={rec['filter_score']} url={rec['url']}")
            print(f"    keywords={rec['matched_keywords']} url_keywords={rec['url_keywords']}")
            print(textwrap.indent(textwrap.shorten(rec["text"], args.preview_chars), "    "))
        logger.info("Preview only - nothing written.")
        return

    if not selected:
        raise SystemExit(
            "No documents matched the support filter. Increase C4_MAX_SCANNED, lower "
            "C4_MIN_KEYWORD_MATCHES or extend C4_SUPPORT_KEYWORDS."
        )
    if len(selected) < cfg.c4_max_samples:
        logger.warning(
            "Only %d of %d requested documents found within C4_MAX_SCANNED=%d. "
            "Increase C4_MAX_SCANNED (acceptance rate %.3f%%) to collect more.",
            len(selected), cfg.c4_max_samples, cfg.c4_max_scanned,
            100.0 * len(selected) / max(stats.scanned, 1),
        )

    out = cfg.evaluation_dataset_path
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for rec in selected:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    scores = sorted(r["filter_score"] for r in selected)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {"id": cfg.c4_dataset_id, "config": cfg.c4_config, "split": cfg.c4_split, "streaming": True},
        "domain": cfg.c4_domain,
        "sampling": "deterministic stream order of the split; first accepted documents are kept",
        "seed": cfg.seed,
        "filter": {
            "keywords": list(cfg.c4_support_keywords),
            "url_keywords": list(cfg.c4_url_keywords),
            "allowed_domains": list(cfg.c4_allowed_domains),
            "min_keyword_matches": cfg.c4_min_keyword_matches,
            "min_text_length": cfg.c4_min_text_length,
            "max_text_length": cfg.c4_max_text_length,
            "rule": (
                ">= min_keyword_matches distinct text phrases, or URL keyword + >= 1 text phrase; "
                "phrases matched case-insensitively on word boundaries"
            ),
        },
        "requested_samples": cfg.c4_max_samples,
        "max_scanned": cfg.c4_max_scanned,
        "sample_count": len(selected),
        "statistics": stats.to_dict(),
        "filter_score": {"min": scores[0], "median": scores[len(scores) // 2], "max": scores[-1]},
        "limitation": (
            "C4 is a general web crawl; this is a heuristically filtered support-like subset, "
            "not a professionally curated support corpus."
        ),
        "output": str(out),
    }
    manifest_path = out.with_suffix(".manifest.json")
    write_json(manifest_path, manifest)
    logger.info("Wrote %d documents to %s (manifest: %s)", len(selected), out, manifest_path)


if __name__ == "__main__":
    main()
