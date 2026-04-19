"""Train ``ContrastiveMLP`` on MIMIC-IV radiology sentences with CheXpert proxy labels.

This script uses real MIMIC tables available through ``MIMIC4Dataset``:

- ``note_tables=["radiology"]`` for radiology report text
- ``cxr_tables=["chexpert"]`` for weak binary labels aggregated at patient level

It is a *proxy* setup for compute/reproducibility: labels are not sentence-level
teacher labels from the paper, but they are derived from real CheXpert columns.

Example (paths are placeholders; use your local PhysioNet download roots):

.. code-block:: bash

    python examples/mimic4_radiology_sentence_abnormality_contrastive_mlp.py \\
      --note-root /path/to/mimic-iv-note/2.2 \\
      --cxr-root /path/to/mimic-cxr/2.0.0 \\
      --cache-dir /tmp/mimic_cache \\
      --epochs 2

Ablation:
    Sweeps ``contrastive_weight`` and ``hidden_dim`` and prints validation metrics.
"""

from __future__ import annotations

import argparse
from typing import Dict

from pyhealth.datasets import MIMIC4Dataset, get_dataloader, split_by_patient
from pyhealth.models import ContrastiveMLP
from pyhealth.tasks.mimic4_radiology_sentence_chexpert_proxy import (
    MIMIC4RadiologySentenceCheXpertProxy,
)
from pyhealth.trainer import Trainer


def run_once(sample_dataset, hidden_dim: int, contrastive_weight: float, epochs: int) -> Dict[str, float]:
    train_ds, val_ds, _ = split_by_patient(sample_dataset, [0.7, 0.2, 0.1])
    train_loader = get_dataloader(train_ds, batch_size=32, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=32, shuffle=False)

    model = ContrastiveMLP(
        dataset=sample_dataset,
        hidden_dim=hidden_dim,
        contrastive_weight=contrastive_weight,
    )
    trainer = Trainer(
        model=model,
        metrics=["accuracy", "roc_auc"],
        device="cpu",
        enable_logging=False,
    )
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=epochs,
        monitor="roc_auc",
        optimizer_params={"lr": 1e-3},
    )
    return trainer.evaluate(val_loader)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--note-root", required=True, help="Path to MIMIC-IV Note root (contains note/).")
    parser.add_argument("--cxr-root", required=True, help="Path to MIMIC-CXR root (contains chexpert table).")
    parser.add_argument("--cache-dir", default=None, help="Optional LitData cache directory.")
    parser.add_argument("--dev", action="store_true", help="Use dev=True for faster subset processing.")
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()

    base = MIMIC4Dataset(
        note_root=args.note_root,
        cxr_root=args.cxr_root,
        note_tables=["radiology"],
        cxr_tables=["chexpert"],
        cache_dir=args.cache_dir,
        dev=args.dev,
        num_workers=1,
    )

    task = MIMIC4RadiologySentenceCheXpertProxy(bow_dim=128)
    sample_dataset = base.set_task(task, num_workers=1)

    settings = [
        {"hidden_dim": 64, "contrastive_weight": 0.0},
        {"hidden_dim": 64, "contrastive_weight": 0.1},
        {"hidden_dim": 128, "contrastive_weight": 0.1},
    ]
    print(f"Total samples: {len(sample_dataset)}")
    for s in settings:
        metrics = run_once(sample_dataset, epochs=args.epochs, **s)
        print(f"setting={s} -> {metrics}")


if __name__ == "__main__":
    main()
