"""
Prompt templates for the clinical RAG endpoint.

We keep templates here rather than inline so they're easy to iterate on
and A/B test without touching the API logic.  Each template is a function
that takes structured data and returns a ready-to-send messages list.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any


SYSTEM_PROMPT = """\
You are a critical care clinical decision support assistant.  Your role is to
analyse ICU patient data and identify patients at elevated risk of sepsis
within the next 6 hours.

Guidelines:
- Base your assessment on the provided SOFA scores and retrieved clinical notes.
- Be concise and clinician-friendly: use medical terminology, avoid hedging.
- For each high-risk patient, give a brief rationale (2-3 sentences) referencing
  specific lab values, vital trends, or note findings.
- Rank patients from highest to lowest risk.
- If data is insufficient to assess a patient, say so clearly rather than guessing.
- Do not fabricate clinical details not present in the provided context.
"""


def build_sepsis_alert_messages(
    query: str,
    alert_rows: list[dict[str, Any]],
    note_excerpts: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """
    Build the messages list for the sepsis risk query.

    Args:
        query:        Clinician's natural language question
        alert_rows:   List of patient alert dicts from the `sepsis_alerts` table
        note_excerpts: Retrieved clinical note chunks for context
    """
    # Format patient data block
    patient_blocks = []
    for row in alert_rows:
        block = (
            f"Patient ICU Stay: {row['icustay_id']}\n"
            f"  SOFA Total: {row['sofa_total']} ({row['risk_tier'].upper()} risk)\n"
            f"  Sub-scores — Respiratory: {row['sofa_resp']}, Coagulation: {row['sofa_coag']}, "
            f"Liver: {row['sofa_liver']}, Cardiovascular: {row['sofa_cardio']}, "
            f"CNS: {row['sofa_cns']}, Renal: {row['sofa_renal']}\n"
            f"  Score window end: {row['score_window_end']}"
        )
        patient_blocks.append(block)

    patients_section = "\n\n".join(patient_blocks) if patient_blocks else "No patients currently flagged."

    # Format retrieved notes
    notes_lines = []
    for i, note in enumerate(note_excerpts[:8], 1):   # cap at 8 excerpts to stay within context
        notes_lines.append(
            f"[Excerpt {i} — ICU Stay {note.get('icustay_id', 'unknown')}, "
            f"Category: {note.get('note_category', 'unknown')}]\n{note['chunk_text'][:600]}"
        )

    notes_section = (
        "\n\n---\n\n".join(notes_lines)
        if notes_lines
        else "No clinical notes retrieved."
    )

    user_content = f"""
Clinician Query: {query}

Retrieved Timestamp: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}

─── SOFA Alert Data ───────────────────────────────────────────────
{patients_section}

─── Retrieved Clinical Notes ──────────────────────────────────────
{notes_section}

─── Task ───────────────────────────────────────────────────────────
Based on the above data, provide a ranked sepsis risk assessment.
For each patient, include:
1. Risk level and primary clinical drivers
2. Organ systems of greatest concern
3. Recommended immediate monitoring actions

Format your response as a structured list, one entry per patient.
""".strip()

    return [
        {"role": "user", "content": user_content},
    ]
