"""Core text segmentation and sentence scoring for Pipeline 3."""

from __future__ import annotations

import re
from collections import Counter

from .text import LEGAL_TERMS, normalize_text, tokenize, top_terms

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")
REF_RE = re.compile(r"\b(article|section|rule|act)\s+[0-9a-zA-Z().-]+", re.IGNORECASE)

SECTION_HINTS = {
    "facts": ("fact", "background", "incident", "case of the prosecution", "briefly stated"),
    "legal_issues": ("issue", "question", "point for consideration", "whether", "contention"),
    "arguments": ("submitted", "contended", "argued", "learned counsel", "submission"),
    "reasoning": ("held", "observed", "considered", "therefore", "in view of", "we find"),
    "holding": ("appeal allowed", "appeal dismissed", "petition dismissed", "order", "disposed", "conviction"),
}


def split_sentences(text: str) -> list[str]:
    clean = normalize_text(text)
    if not clean:
        return []
    rough = SENTENCE_RE.split(clean)
    sentences = []
    for sentence in rough:
        sentence = sentence.strip()
        if len(sentence) < 25:
            continue
        if len(sentence) > 1200:
            parts = [sentence[i : i + 900].strip() for i in range(0, len(sentence), 900)]
            sentences.extend(part for part in parts if len(part) >= 25)
        else:
            sentences.append(sentence)
    return sentences


def detect_section(sentence: str, position_ratio: float) -> str:
    lower = sentence.lower()
    for section, hints in SECTION_HINTS.items():
        if any(hint in lower for hint in hints):
            return section
    if position_ratio < 0.22:
        return "facts"
    if position_ratio > 0.82:
        return "holding"
    return "reasoning"


def segment_document(doc: dict, doc_type: str, max_chunk_sentences: int = 12) -> dict:
    sentences = split_sentences(doc["text"])
    chunks = []
    total = max(1, len(sentences))
    for i in range(0, len(sentences), max_chunk_sentences):
        chunk_sentences = sentences[i : i + max_chunk_sentences]
        chunk_text = " ".join(chunk_sentences)
        midpoint = min(total - 1, i + len(chunk_sentences) // 2) / total
        section = detect_section(chunk_text, midpoint)
        chunks.append(
            {
                "chunk_id": f"{doc['doc_id']}_chunk_{len(chunks)+1}",
                "doc_id": doc["doc_id"],
                "doc_type": doc_type,
                "chunk_index": len(chunks) + 1,
                "section": section,
                "token_count": len(tokenize(chunk_text)),
                "text": chunk_text,
            }
        )
    return {
        "doc_id": doc["doc_id"],
        "doc_type": doc_type,
        "sentence_count": len(sentences),
        "chunk_count": len(chunks),
        "chunks": chunks,
    }


def sentence_scores(text: str, query: str | None = None) -> list[tuple[int, str, float, str]]:
    sentences = split_sentences(text)
    if not sentences:
        return []
    doc_terms = set(top_terms(text, limit=24))
    query_terms = set(tokenize(query or "", remove_stopwords=True))
    scores = []
    total = max(1, len(sentences))
    for idx, sentence in enumerate(sentences):
        terms = tokenize(sentence, remove_stopwords=True)
        counts = Counter(terms)
        legal_overlap = sum(counts[t] for t in LEGAL_TERMS if t in counts)
        top_overlap = len(set(terms) & doc_terms)
        query_overlap = len(set(terms) & query_terms)
        ref_bonus = 2.0 if REF_RE.search(sentence) else 0.0
        lower = sentence.lower()
        outcome_bonus = 3.0 if any(phrase in lower for phrase in ("appeal allowed", "appeal dismissed", "petition dismissed", "held that", "we hold")) else 0.0
        position_bonus = 1.5 if idx < 3 or idx > total - 5 else 0.0
        length_penalty = 0.0 if 12 <= len(terms) <= 90 else -1.0
        score = legal_overlap * 0.25 + top_overlap * 0.8 + query_overlap * 1.8 + ref_bonus + outcome_bonus + position_bonus + length_penalty
        scores.append((idx, sentence, score, detect_section(sentence, idx / total)))
    return scores


def select_section_sentences(text: str, section: str, *, query: str | None = None, limit: int = 2) -> list[str]:
    scored = [row for row in sentence_scores(text, query=query) if row[3] == section]
    if not scored:
        scored = sentence_scores(text, query=query)
    ranked = sorted(scored, key=lambda row: (-row[2], row[0]))[:limit]
    return [sentence for _, sentence, _, _ in sorted(ranked, key=lambda row: row[0])]


def extractive_summary(text: str, *, query: str | None = None, limit: int = 8) -> list[str]:
    ranked = sorted(sentence_scores(text, query=query), key=lambda row: (-row[2], row[0]))[:limit]
    return [sentence for _, sentence, _, _ in sorted(ranked, key=lambda row: row[0])]
