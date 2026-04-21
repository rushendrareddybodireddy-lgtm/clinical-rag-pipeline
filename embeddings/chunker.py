"""
Clinical note chunking for vector embedding.

Clinical notes in MIMIC are long-form free text — discharge summaries can
exceed 5,000 tokens.  This module splits them into overlapping chunks
suitable for embedding while preserving enough context for the RAG retriever
to return coherent, clinically meaningful passages.

Strategy: sentence-boundary-aware sliding window.
  - Target chunk size: 400 tokens (~1,600 chars)
  - Overlap: 80 tokens (~320 chars) to avoid context loss at boundaries
  - We do NOT use recursive character splitting — clinical notes have
    distinct section headers (e.g. "Chief Complaint:", "Assessment:") that
    we use as natural split points first.
"""

from __future__ import annotations

import re
from typing import Any

# Clinical note section headers common in MIMIC NOTEEVENTS
SECTION_HEADERS = re.compile(
    r"(?m)^("
    r"Chief Complaint|History of Present Illness|Past Medical History|"
    r"Social History|Family History|Review of Systems|Physical Exam|"
    r"Assessment( and Plan)?|Plan|Medications|Allergies|"
    r"Discharge (Diagnosis|Condition|Instructions|Medications)|"
    r"Laboratory Data|Imaging|Procedures|Impression"
    r")\s*:",
    re.IGNORECASE,
)

# Very rough chars-per-token estimate for clinical English
CHARS_PER_TOKEN = 4
CHUNK_TOKENS    = 400
OVERLAP_TOKENS  = 80

CHUNK_SIZE_CHARS    = CHUNK_TOKENS * CHARS_PER_TOKEN    # 1600
OVERLAP_SIZE_CHARS  = OVERLAP_TOKENS * CHARS_PER_TOKEN  # 320


def _split_on_sections(text: str) -> list[str]:
    """Split note text on section headers, keeping headers with their content."""
    splits = SECTION_HEADERS.split(text)
    sections: list[str] = []

    i = 0
    while i < len(splits):
        segment = splits[i].strip()
        if not segment:
            i += 1
            continue
        # If next element looks like a header label, merge them
        if i + 1 < len(splits) and SECTION_HEADERS.match(splits[i + 1] + ":"):
            i += 1
            continue
        sections.append(segment)
        i += 1

    return sections if sections else [text]


def _sliding_window(text: str) -> list[str]:
    """Fallback: fixed-size sliding window when section splitting isn't useful."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE_CHARS
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - OVERLAP_SIZE_CHARS   # overlap
    return chunks


def chunk_note(text: str, note_metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """
    Split a clinical note into embeddable chunks.

    Returns a list of dicts:
        {
            "text":       str,   # chunk text
            "chunk_index": int,  # position within this note
            "char_start":  int,  # char offset in original text
            **note_metadata,     # hadm_id, icustay_id, etc.
        }
    """
    if not text or not text.strip():
        return []

    meta = note_metadata or {}

    # Step 1: section-level split
    sections = _split_on_sections(text)

    chunks: list[dict[str, Any]] = []
    char_offset = 0
    chunk_idx   = 0

    for section in sections:
        if len(section) <= CHUNK_SIZE_CHARS:
            if section.strip():
                chunks.append({
                    "text":        section.strip(),
                    "chunk_index": chunk_idx,
                    "char_start":  char_offset,
                    **meta,
                })
                chunk_idx += 1
        else:
            # Section is too big — apply sliding window within it
            sub_chunks = _sliding_window(section)
            for sub in sub_chunks:
                chunks.append({
                    "text":        sub,
                    "chunk_index": chunk_idx,
                    "char_start":  char_offset,
                    **meta,
                })
                chunk_idx += 1
        char_offset += len(section)

    return chunks
