"""Text utilities shared across retrieval, expansion, and RAG modules."""

from __future__ import annotations

import html
import re
from collections import Counter


TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")
SPACE_RE = re.compile(r"\s+")

STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "with",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
}

LEGAL_TERMS = {
    "act",
    "appeal",
    "appellant",
    "article",
    "authority",
    "bench",
    "case",
    "civil",
    "claim",
    "clause",
    "constitution",
    "contract",
    "conviction",
    "court",
    "criminal",
    "decree",
    "defendant",
    "evidence",
    "government",
    "high",
    "judge",
    "judgment",
    "jurisdiction",
    "law",
    "legal",
    "liability",
    "order",
    "petition",
    "petitioner",
    "plaintiff",
    "power",
    "proceeding",
    "property",
    "relief",
    "respondent",
    "right",
    "rule",
    "section",
    "sentence",
    "statute",
    "suit",
    "supreme",
    "tribunal",
    "writ",
}


def normalize_text(text: str) -> str:
    """Conservative cleanup that preserves legal content and identifiers."""
    text = html.unescape(text or "")
    text = text.replace("\ufeff", " ")
    text = text.replace("\r", "\n")
    text = SPACE_RE.sub(" ", text)
    return text.strip()


def tokenize(text: str, *, remove_stopwords: bool = False) -> list[str]:
    tokens = [m.group(0).lower() for m in TOKEN_RE.finditer(text or "")]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return tokens


def simple_stem(token: str) -> str:
    token = token.lower()
    for suffix in ("ments", "ment", "ingly", "edly", "ing", "ies", "ied", "ed", "es", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            if suffix in {"ies", "ied"}:
                return token[: -len(suffix)] + "y"
            return token[: -len(suffix)]
    return token


def token_count(text: str) -> int:
    return len(tokenize(text))


def top_terms(text: str, limit: int = 12) -> list[str]:
    counts = Counter(tokenize(text, remove_stopwords=True))
    scored = []
    for term, freq in counts.items():
        legal_bonus = 2 if term in LEGAL_TERMS else 0
        scored.append((freq + legal_bonus, len(term), term))
    scored.sort(reverse=True)
    return [term for _, _, term in scored[:limit]]


def snippet_for_query(text: str, query: str, max_chars: int = 650) -> str:
    clean = normalize_text(text)
    if len(clean) <= max_chars:
        return clean
    query_terms = set(tokenize(query, remove_stopwords=True))
    best_start = 0
    best_score = -1
    window = max_chars
    step = max(80, max_chars // 4)
    for start in range(0, max(1, len(clean) - window), step):
        chunk = clean[start : start + window]
        score = len(query_terms.intersection(tokenize(chunk, remove_stopwords=True)))
        if score > best_score:
            best_score = score
            best_start = start
    snippet = clean[best_start : best_start + max_chars].strip()
    if best_start > 0:
        snippet = "..." + snippet
    if best_start + max_chars < len(clean):
        snippet += "..."
    return snippet
