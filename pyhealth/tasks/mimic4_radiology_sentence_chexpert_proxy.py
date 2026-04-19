"""MIMIC-CXR radiology sentence task with CheXpert-derived proxy labels.

This task is a *pragmatic proxy* for paper-style sentence abnormality labels when
true sentence-level annotations are unavailable: it uses MIMIC-IV radiology note
text split into sentences, and derives a binary document-level abnormality label
from aggregated CheXpert columns on the same patient.

Notes:
    - This is not identical to the paper's teacher-labeled sentences.
    - It is intended for reproducible PyHealth pipelines on credentialed MIMIC data
      (or small subsets for compute limits).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from .base_task import BaseTask

_CHEXPERT_ATTRS = (
    "atelectasis",
    "cardiomegaly",
    "consolidation",
    "edema",
    "enlarged cardiomediastinum",
    "fracture",
    "lung lesion",
    "lung opacity",
    "pleural effusion",
    "pleural other",
    "pneumonia",
    "pneumothorax",
    "support devices",
)


def _split_sentences(text: str) -> List[str]:
    chunks = re.split(r"(?<=[.!?])\s+", str(text).strip())
    return [c.strip() for c in chunks if c.strip()]


def _hash_bow_vector(text: str, dim: int = 128) -> List[float]:
    """Very small bag-of-words hashing projection (deterministic, no deps)."""
    vec = [0.0] * dim
    for token in re.findall(r"[A-Za-z0-9]+", text.lower()):
        idx = hash(token) % dim
        vec[idx] += 1.0
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return [v / norm for v in vec]


def _patient_abnormal_from_chexpert(patient: Any) -> int:
    events = patient.get_events(event_type="chexpert")
    for ev in events:
        for attr in _CHEXPERT_ATTRS:
            val = ev.get(attr, 0)
            try:
                v = float(val)
            except (TypeError, ValueError):
                continue
            if v == 1.0 or v == -1.0:
                return 1
    return 0


class MIMIC4RadiologySentenceCheXpertProxy(BaseTask):
    """Radiology sentences with CheXpert-derived patient-level abnormality label."""

    task_name: str = "MIMIC4RadiologySentenceCheXpertProxy"
    input_schema: Dict[str, str] = {"sentence_bow": "tensor"}
    output_schema: Dict[str, str] = {"label": "binary"}

    def __init__(self, bow_dim: int = 128) -> None:
        super().__init__()
        self.bow_dim = bow_dim

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        label = _patient_abnormal_from_chexpert(patient)
        samples: List[Dict[str, Any]] = []
        for note in patient.get_events(event_type="radiology"):
            text = str(note.get("text", "")).strip()
            if not text:
                continue
            for sent in _split_sentences(text):
                samples.append(
                    {
                        "patient_id": patient.patient_id,
                        "visit_id": str(note.get("hadm_id", "")),
                        "sentence_bow": [_hash_bow_vector(sent, self.bow_dim)],
                        "label": label,
                    }
                )
        return samples
