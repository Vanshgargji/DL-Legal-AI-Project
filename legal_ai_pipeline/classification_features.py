"""Feature extraction for legal issue classification."""

from __future__ import annotations

import re
from collections import Counter

from .classification_config import ClassificationConfig
from .classification_schema import LABEL_SCHEMA
from .io_utils import ensure_dir, read_jsonl, write_json, write_jsonl
from .text import tokenize, top_terms

REF_RE = re.compile(r"\b(article|section|rule|act)\s+[0-9a-zA-Z()-]+", re.IGNORECASE)


def legal_keyword_scores(text: str) -> dict[str, float]:
    tokens = tokenize(text, remove_stopwords=True)
    token_counts = Counter(tokens)
    lower = " ".join(tokens)
    scores = {}
    for label, meta in LABEL_SCHEMA.items():
        if label == "other":
            continue
        score = 0.0
        for kw in meta["keywords"]:
            kw_tokens = tokenize(kw, remove_stopwords=True)
            if len(kw_tokens) == 1:
                score += token_counts.get(kw_tokens[0], 0)
            elif " ".join(kw_tokens) in lower:
                score += 2.5
        scores[label] = score
    return scores


def extract_features(item: dict, item_type: str) -> dict:
    text = (item.get("title", "") + " " + item["text"]).strip()
    tokens = tokenize(text, remove_stopwords=True)
    refs = REF_RE.findall(text)
    keyword_scores = legal_keyword_scores(text)
    return {
        "id": item.get("doc_id") or item.get("query_id"),
        "item_type": item_type,
        "token_count": len(tokens),
        "top_terms": top_terms(text, limit=20),
        "reference_count": len(refs),
        "keyword_scores": keyword_scores,
    }


def extract_all_features(config: ClassificationConfig) -> dict:
    ensure_dir(config.features_dir)
    cases = read_jsonl(config.processed_dir / "cases.jsonl")
    statutes = read_jsonl(config.processed_dir / "statutes.jsonl")
    queries = read_jsonl(config.processed_dir / "queries.jsonl")

    case_features = [extract_features(doc, "case") for doc in cases]
    statute_features = [extract_features(doc, "statute") for doc in statutes]
    query_features = [extract_features({"query_id": q["query_id"], "text": q["text"]}, "query") for q in queries]
    write_jsonl(config.features_dir / "case_features.jsonl", case_features)
    write_jsonl(config.features_dir / "statute_features.jsonl", statute_features)
    write_jsonl(config.features_dir / "query_features.jsonl", query_features)
    all_terms = Counter()
    for row in case_features + statute_features + query_features:
        all_terms.update(row["top_terms"])
    vocabulary = {"top_terms": [term for term, _ in all_terms.most_common(5000)]}
    write_json(config.features_dir / "vocabulary.json", vocabulary)
    return {
        "case_features": len(case_features),
        "statute_features": len(statute_features),
        "query_features": len(query_features),
        "vocabulary_terms": len(vocabulary["top_terms"]),
    }
