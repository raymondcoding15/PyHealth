import unittest
from datetime import datetime

from pyhealth.tasks.mimic4_radiology_sentence_chexpert_proxy import (
    MIMIC4RadiologySentenceCheXpertProxy,
)


class _Evt:
    def __init__(self, event_type: str, **attrs):
        self.event_type = event_type
        self.timestamp = datetime(2020, 1, 1)
        self._attrs = attrs

    def get(self, key, default=None):
        return self._attrs.get(key, default)


class _Patient:
    def __init__(self, patient_id: str, events):
        self.patient_id = patient_id
        self._events = events

    def get_events(self, event_type=None, **kwargs):
        _ = kwargs
        return [e for e in self._events if e.event_type == event_type]


class TestMIMIC4RadiologySentenceCheXpertProxy(unittest.TestCase):
    def test_splits_sentences_and_labels(self):
        task = MIMIC4RadiologySentenceCheXpertProxy(bow_dim=16)
        patient = _Patient(
            "p1",
            [
                _Evt("chexpert", **{"lung opacity": 1.0}),
                _Evt(
                    "radiology",
                    text="Sentence one. Sentence two!",
                    hadm_id="1",
                ),
            ],
        )
        samples = task(patient)
        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0]["label"], 1)
        self.assertEqual(len(samples[0]["sentence_bow"][0]), 16)

    def test_normal_when_no_positive_findings(self):
        task = MIMIC4RadiologySentenceCheXpertProxy(bow_dim=8)
        patient = _Patient(
            "p2",
            [
                _Evt("chexpert", **{"no finding": 1.0}),
                _Evt("radiology", text="Normal study.", hadm_id="2"),
            ],
        )
        samples = task(patient)
        self.assertEqual(samples[0]["label"], 0)


if __name__ == "__main__":
    unittest.main()
