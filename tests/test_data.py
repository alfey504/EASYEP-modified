import json

import numpy as np
import pytest

from moe_pruning.data.c4_support import Deduplicator, SupportDocumentFilter
from moe_pruning.data.calibration import (
    CalibrationDataError,
    TokenizedPair,
    load_calibration_records,
    split_train_validation,
)
from moe_pruning.data.collators import collate_seq2seq
from moe_pruning.data.span_corruption import corrupt_example, random_spans_noise_mask
from tests.conftest import REAL_MODEL_ID, requires_real_model


class FakeT5Tokenizer:
    """Minimal stand-in exposing what span corruption needs (sentinels = 1000 - i)."""

    eos_token_id = 1
    unk_token_id = 2

    def convert_tokens_to_ids(self, tokens):
        return [1000 - int(t[len("<extra_id_") : -1]) for t in tokens]


# ----------------------------------------------------------------------
# Calibration JSONL
# ----------------------------------------------------------------------
def test_calibration_loader_validates_records(tmp_path):
    path = tmp_path / "calib.jsonl"
    lines = [
        json.dumps({"human": "How do I reset my password?", "response": "Use the reset link."}),
        "",
        "{not json",
        json.dumps({"human": "only human"}),
        json.dumps({"human": "  ", "response": "x"}),
        json.dumps(["a", "list"]),
        json.dumps({"human": "Second", "response": "Answer", "extra": 1}),
    ]
    path.write_text("\n".join(lines) + "\n")
    records, report = load_calibration_records(path)
    assert [r.human for r in records] == ["How do I reset my password?", "Second"]
    assert [r.line_number for r in records] == [1, 7]
    assert report.skipped_blank_lines == 1
    assert report.skipped_malformed_json == 2
    assert report.skipped_missing_fields == 1
    assert report.skipped_empty_fields == 1
    assert any("line 3" in p for p in report.problems)


def test_calibration_max_samples(tmp_path):
    path = tmp_path / "calib.jsonl"
    path.write_text("".join(json.dumps({"human": f"q{i}", "response": f"a{i}"}) + "\n" for i in range(10)))
    records, report = load_calibration_records(path, max_samples=3)
    assert len(records) == 3 and report.truncated_at_max_samples


def test_calibration_errors(tmp_path):
    with pytest.raises(CalibrationDataError, match="not found"):
        load_calibration_records(tmp_path / "missing.jsonl")
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{}\n")
    with pytest.raises(CalibrationDataError, match="No valid"):
        load_calibration_records(bad)


def test_split_is_deterministic_and_disjoint():
    items = list(range(20))
    train1, val1 = split_train_validation(items, 0.2, seed=3)
    train2, val2 = split_train_validation(items, 0.2, seed=3)
    assert (train1, val1) == (train2, val2)
    assert len(val1) == 4 and not set(train1) & set(val1)
    assert sorted(train1 + val1) == items


# ----------------------------------------------------------------------
# Collation and padding masks
# ----------------------------------------------------------------------
def test_collator_right_pads_and_masks():
    items = [TokenizedPair(0, [5, 6, 7, 1], [8, 9, 1]), TokenizedPair(1, [5, 1], [8, 9, 10, 11, 1])]
    batch = collate_seq2seq(items, pad_token_id=0)
    assert batch.input_ids.tolist() == [[5, 6, 7, 1], [5, 1, 0, 0]]
    assert batch.attention_mask.tolist() == [[1, 1, 1, 1], [1, 1, 0, 0]]
    assert batch.labels.tolist() == [[8, 9, 1, -100, -100], [8, 9, 10, 11, 1]]
    masks = batch.token_masks()
    assert masks["encoder"].sum() == 6 and masks["decoder"].sum() == 8


# ----------------------------------------------------------------------
# Span corruption
# ----------------------------------------------------------------------
def test_noise_mask_density_and_determinism():
    m1 = random_spans_noise_mask(200, 0.15, 3.0, np.random.default_rng([42, 7]))
    m2 = random_spans_noise_mask(200, 0.15, 3.0, np.random.default_rng([42, 7]))
    m3 = random_spans_noise_mask(200, 0.15, 3.0, np.random.default_rng([42, 8]))
    assert np.array_equal(m1, m2) and not np.array_equal(m1, m3)
    assert m1.sum() == 30
    spans = int(np.sum(m1[1:] & ~m1[:-1]) + m1[0])
    assert spans == 10  # round(30 / 3)


def test_corruption_is_invertible():
    tok = FakeT5Tokenizer()
    tokens = list(range(10, 110))
    ex = corrupt_example(3, tokens, tok, 0.15, 3.0, seed=42, max_source_length=256, max_target_length=128)
    assert ex.input_ids[-1] == 1 and ex.labels[-1] == 1
    sentinels = {1000 - i for i in range(100)}
    # Rebuild the document by substituting each input sentinel by its target span.
    spans: dict[int, list[int]] = {}
    current = None
    for t in ex.labels[:-1]:
        if t in sentinels:
            current = t
            spans[current] = []
        else:
            spans[current].append(t)
    rebuilt = []
    for t in ex.input_ids[:-1]:
        rebuilt.extend(spans[t] if t in sentinels else [t])
    assert rebuilt == tokens
    assert sum(ex.noise_mask) == 15


def test_corruption_independent_of_other_examples():
    tok = FakeT5Tokenizer()
    a = corrupt_example(5, list(range(10, 90)), tok, 0.15, 3.0, 42, 256, 128)
    b = corrupt_example(5, list(range(10, 90)), tok, 0.15, 3.0, 42, 256, 128)
    assert a == b


def test_corruption_respects_lengths():
    tok = FakeT5Tokenizer()
    ex = corrupt_example(0, list(range(10, 600)), tok, 0.15, 3.0, 42, max_source_length=64, max_target_length=16)
    assert len(ex.source_ids) <= 63 and len(ex.labels) <= 16


# ----------------------------------------------------------------------
# C4 support filter
# ----------------------------------------------------------------------
def _filter(**kw):
    params = dict(
        keywords=("customer support", "help center", "FAQ", "refund policy", "error code"),
        url_keywords=("support", "help", "faq", "knowledge-base"),
        allowed_domains=(),
        min_keyword_matches=2,
        min_text_length=20,
        max_text_length=10_000,
    )
    params.update(kw)
    return SupportDocumentFilter(**params)


def test_filter_requires_specific_phrases():
    f = _filter()
    generic = "We are here to help. Our service team will help you with any service you need, always."
    assert f.evaluate(generic, "https://example.com/about").reason == "no_keyword_match"
    good = "Contact customer support or read the FAQ before asking about our refund policy."
    d = f.evaluate(good, "https://example.com/page")
    assert d.accepted and set(d.matched_keywords) == {"customer support", "FAQ", "refund policy"}


def test_filter_word_boundaries_and_whitespace():
    f = _filter()
    assert f.evaluate("FAQs and FAQing are not FAQ matches?? padding text here", "").matched_keywords == ["FAQ"]
    d = f.evaluate("Visit the help\n  center, then file an error code report. padding", "")
    assert set(d.matched_keywords) == {"help center", "error code"}


def test_filter_url_rules():
    f = _filter()
    one = "Please read the FAQ before writing to us. Some more padding text."
    assert f.evaluate(one, "https://example.com/help/article-1").accepted
    assert not f.evaluate(one, "https://helpful.example.com/x").accepted  # 'helpful' != 'help'
    url_only = "Nothing relevant in this text at all, just words and more words."
    assert f.evaluate(url_only, "https://support.example.com/a").reason == "url_match_only"


def test_filter_length_and_domain():
    f = _filter(allowed_domains=("example.com",))
    assert f.evaluate("short", "").reason == "too_short"
    assert f.evaluate("x" * 20_000, "").reason == "too_long"
    text = "customer support and FAQ are here for you, plenty of text."
    assert f.evaluate(text, "https://other.org/help").reason == "domain_not_allowed"
    assert f.evaluate(text, "https://support.example.com/help").accepted


def test_deduplicator_normalises_text():
    d = Deduplicator()
    assert not d.is_duplicate("Hello   World", "https://a")
    assert d.is_duplicate("hello world", "https://b")
    assert d.is_duplicate("different", "https://a")


@pytest.mark.integration
@requires_real_model
def test_real_t5_tokenizer_sentinels_and_teacher_forcing():
    from transformers import AutoTokenizer

    from moe_pruning.data.calibration import CalibrationRecord, tokenize_calibration
    from moe_pruning.data.span_corruption import sentinel_ids

    tok = AutoTokenizer.from_pretrained(REAL_MODEL_ID)
    ids = sentinel_ids(tok)
    assert ids[0] == tok.convert_tokens_to_ids("<extra_id_0>") and len(set(ids)) == 100
    text = "Customer support can reset your password from the help center in a few minutes. " * 3
    ex = corrupt_example(0, tok(text, add_special_tokens=False).input_ids, tok, 0.15, 3.0, 42, 256, 128)
    assert ex.input_ids[-1] == tok.eos_token_id and ex.labels[0] == ids[0]
    assert "<extra_id_0>" in tok.decode(ex.input_ids)

    pairs = tokenize_calibration([CalibrationRecord("How do I reset it?", "Use the link.", 1)], tok, 256, 128)
    assert pairs[0].input_ids[-1] == tok.eos_token_id and pairs[0].labels[-1] == tok.eos_token_id
