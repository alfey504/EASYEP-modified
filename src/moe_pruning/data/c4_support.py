"""Heuristic support-domain filtering of C4 documents and loading of the result.

C4 is raw web text. The filter keeps documents that look like customer or
technical support content; it is a heuristic, not a curated support corpus.

Matching rules
--------------
* Text keywords are matched case-insensitively as whole phrases (no match
  inside longer words; any whitespace between the words of a phrase).
* Only the configured multi-word / specific phrases count. Generic words such
  as "help" or "service" alone never match.
* URL keywords are matched as whole tokens of the URL (delimited by
  non-alphanumeric characters), e.g. ``/help/`` or ``support.example.com``.
* A document is accepted when it has at least ``min_keyword_matches`` distinct
  text keywords, or when the URL matches and at least one text keyword matches.
  A URL match alone is never sufficient.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse


def _phrase_pattern(phrase: str) -> re.Pattern:
    words = [re.escape(w) for w in phrase.split()]
    return re.compile(r"(?<![A-Za-z0-9])" + r"\s+".join(words) + r"(?![A-Za-z0-9])", re.IGNORECASE)


@dataclass
class FilterDecision:
    accepted: bool
    reason: str  # "accepted" or the rejection reason
    score: float = 0.0
    matched_keywords: list[str] = field(default_factory=list)
    keyword_occurrences: int = 0
    url_keywords: list[str] = field(default_factory=list)


class SupportDocumentFilter:
    def __init__(
        self,
        keywords: tuple[str, ...],
        url_keywords: tuple[str, ...],
        allowed_domains: tuple[str, ...],
        min_keyword_matches: int,
        min_text_length: int,
        max_text_length: int,
    ):
        self.keywords = [(kw, _phrase_pattern(kw)) for kw in keywords]
        self.url_keywords = [(kw, _phrase_pattern(kw)) for kw in url_keywords]
        self.allowed_domains = tuple(d.lower().lstrip(".") for d in allowed_domains)
        self.min_keyword_matches = min_keyword_matches
        self.min_text_length = min_text_length
        self.max_text_length = max_text_length

    def _domain_allowed(self, url: str) -> bool:
        if not self.allowed_domains:
            return True
        host = (urlparse(url).hostname or "").lower()
        return any(host == d or host.endswith("." + d) for d in self.allowed_domains)

    def evaluate(self, text: str, url: str) -> FilterDecision:
        n = len(text)
        if n < self.min_text_length:
            return FilterDecision(False, "too_short")
        if n > self.max_text_length:
            return FilterDecision(False, "too_long")
        if not self._domain_allowed(url):
            return FilterDecision(False, "domain_not_allowed")

        matched: list[str] = []
        occurrences = 0
        for kw, pattern in self.keywords:
            count = len(pattern.findall(text))
            if count:
                matched.append(kw)
                occurrences += count
        url_matched = [kw for kw, pattern in self.url_keywords if pattern.search(url)]

        n_matched = len(matched)
        accepted = n_matched >= self.min_keyword_matches or (bool(url_matched) and n_matched >= 1)
        # Distinct phrases dominate; repeated mentions and URL evidence add a little.
        score = n_matched + 0.25 * min(occurrences - n_matched, 8) + (1.0 if url_matched else 0.0)
        if accepted:
            reason = "accepted"
        elif n_matched == 0:
            reason = "url_match_only" if url_matched else "no_keyword_match"
        else:
            reason = "insufficient_keyword_matches"
        return FilterDecision(accepted, reason, round(score, 3), matched, occurrences, url_matched)


def normalized_text_hash(text: str) -> str:
    normalized = " ".join(text.lower().split())
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


@dataclass
class FilterStatistics:
    scanned: int = 0
    accepted: int = 0
    rejected: int = 0
    duplicates: int = 0
    rejection_reasons: Counter = field(default_factory=Counter)

    def to_dict(self) -> dict:
        return {
            "scanned": self.scanned,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "duplicates": self.duplicates,
            "rejection_reasons": dict(self.rejection_reasons.most_common()),
        }


class Deduplicator:
    def __init__(self) -> None:
        self._texts: set[str] = set()
        self._urls: set[str] = set()

    def is_duplicate(self, text: str, url: str) -> bool:
        h = normalized_text_hash(text)
        if h in self._texts or (url and url in self._urls):
            return True
        self._texts.add(h)
        if url:
            self._urls.add(url)
        return False


# ----------------------------------------------------------------------
# Loading the generated evaluation dataset
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class EvalDocument:
    example_id: int
    text: str
    url: str
    timestamp: str


def load_eval_documents(path: Path, max_samples: int) -> list[EvalDocument]:
    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation dataset {path} not found. Run `uv run python scripts/build_c4_support_eval.py`."
        )
    docs: list[EvalDocument] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON ({exc.msg})") from exc
            if not isinstance(obj.get("text"), str) or not obj["text"].strip():
                raise ValueError(f"{path}:{line_number}: record has no 'text'")
            docs.append(EvalDocument(len(docs), obj["text"], obj.get("url", ""), str(obj.get("timestamp", ""))))
            if len(docs) >= max_samples:
                break
    if not docs:
        raise ValueError(f"No evaluation documents in {path}")
    return docs
