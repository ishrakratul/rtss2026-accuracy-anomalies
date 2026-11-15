"""
calibration.py

Builds a `Calibration` object from raw per-classifier outputs (confidence,
correctness, inference time), following the paper's calibration procedure
(Section I and Section III of *Accuracy Anomalies in Classifier Cascades*).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Map of (label, column-prefix-in-CSV) for the six classifiers used in the paper.
ALL_CLASSIFIERS = [
    ("K1", "AlexNet"),
    ("K2", "EfficientNet"),
    ("K3", "VGG16"),
    ("K4", "ResNet18"),
    ("K5", "ConvNeXt"),
    ("K6", "Swin_Transformer"),
]


# ---------------------------------------------------------------------------
# Subset-mask helpers
# ---------------------------------------------------------------------------
def members(mask: int, labels):
    """Return the labels whose bit is set in `mask`."""
    return [k for i, k in enumerate(labels) if mask & (1 << i)]


def remove(mask: int, label: str, labels):
    """Drop `label` from the mask."""
    return mask & ~(1 << labels.index(label))


def subsets_of_size(n: int, size: int):
    """All bitmasks over n bits with popcount == size."""
    return [m for m in range(1 << n) if bin(m).count("1") == size]


def fmt_set(mask: int, labels):
    ms = members(mask, labels)
    return "{" + ", ".join(ms) + "}" if ms else "\u2205"


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
@dataclass
class Calibration:
    labels: list           # e.g. ["K1","K2",...]
    models: list            # CSV column prefixes
    Ci: dict                # WORST-CASE execution time (max over samples)
    C_bar: dict              # AVERAGE-CASE execution time (mean over samples)
    H: dict                  # confidence thresholds
    status: dict              # label -> ["correct"|"incorrect"|"IDK"] array
    PrCond: dict               # (Ki, result, S_mask) -> prob
    PrIDK: dict                 # S_mask -> P(all in S are IDK)
    n_samples: int               # for diagnostics


def _bool(series: pd.Series) -> np.ndarray:
    """Coerce the `*_correct` column to bool."""
    if series.dtype == bool:
        return series.to_numpy()
    return series.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    ).astype(bool).to_numpy()


def _threshold_for_psi(conf: np.ndarray, correct: np.ndarray, psi: float):
    """Lowest H such that mean(correct[conf >= H]) >= psi.

    Returns (H, achieved_accuracy). If no such H exists, returns the H
    with the highest achievable accuracy (and the caller can see the
    violation).
    """
    best_H, best_acc = None, -np.inf
    for H in np.sort(np.unique(conf)):
        accepted = conf >= H
        if accepted.sum() == 0:
            continue
        acc = correct[accepted].mean()
        if acc > best_acc:
            best_H, best_acc = H, acc
        if acc >= psi:
            return float(H), float(acc)
    return float(best_H), float(best_acc)


def build_calibration(df: pd.DataFrame,
                       classifiers: list,
                       psi: float) -> Calibration:
    """Build everything (thresholds, status, PrCond, PrIDK) for one (psi, K) pair."""
    labels = [k for k, _ in classifiers]
    models = [m for _, m in classifiers]
    n = len(df)
    nK = len(labels)

    # --- Per-classifier execution times ----------------------------------
    # Ci    = WORST-CASE  (max of per-sample inference times) -- used for the
    #         hard-deadline constraint casWCET <= D.
    # C_bar = AVERAGE-CASE (mean of per-sample inference times) -- used in the
    #         expected-cost objective (Algs 2 line 11/25, Alg 7 line 9/11/18)
    #         and in Eq. 8's expected classification time.
    Ci = {k: float(df[f"{m}_inference_time"].max()) for k, m in classifiers}
    C_bar = {k: float(df[f"{m}_inference_time"].mean()) for k, m in classifiers}

    # --- Confidence thresholds and per-sample status ---------------------
    H, status = {}, {}
    for k, m in classifiers:
        conf = df[f"{m}_confidence"].to_numpy(dtype=float)
        correct = _bool(df[f"{m}_correct"])
        H[k], _ = _threshold_for_psi(conf, correct, psi)
        # conf >= H -> "correct" or "incorrect"; conf < H -> "IDK"
        status[k] = np.where(
            conf < H[k], "IDK",
            np.where(correct, "correct", "incorrect")
        )

    # --- PrIDK[S] = fraction of rows where every Kj in S is IDK ----------
    # PrIDK[empty] = 1.0 (vacuous).
    PrIDK = {0: 1.0}
    for mask in range(1, 1 << nK):
        ms = members(mask, labels)
        idk = np.logical_and.reduce([status[k] == "IDK" for k in ms])
        PrIDK[mask] = float(idk.sum()) / n

    # --- PrCond[(Ki, result, S)] = P(Ki = result | all Kj in S are IDK) --
    # When Ki in S, define PrCond := IDK=1.0 (paper, Table IV footnote).
    # When the conditioning set has 0 matching rows, also default to IDK=1.0.
    PrCond = {}
    for ki in labels:
        ki_status = status[ki]
        ki_bit = 1 << labels.index(ki)
        for mask in range(1 << nK):
            if mask & ki_bit:
                PrCond[(ki, "correct", mask)] = 0.0
                PrCond[(ki, "incorrect", mask)] = 0.0
                PrCond[(ki, "IDK", mask)] = 1.0
                continue

            ms = members(mask, labels)
            if not ms:
                cond = np.ones(n, dtype=bool)
            else:
                cond = np.logical_and.reduce([status[k] == "IDK" for k in ms])

            denom = int(cond.sum())
            if denom == 0:
                PrCond[(ki, "correct", mask)] = 0.0
                PrCond[(ki, "incorrect", mask)] = 0.0
                PrCond[(ki, "IDK", mask)] = 1.0
            else:
                PrCond[(ki, "correct", mask)] = float((cond & (ki_status == "correct")).sum()) / denom
                PrCond[(ki, "incorrect", mask)] = float((cond & (ki_status == "incorrect")).sum()) / denom
                PrCond[(ki, "IDK", mask)] = float((cond & (ki_status == "IDK")).sum()) / denom

    return Calibration(
        labels=labels, models=models, Ci=Ci, C_bar=C_bar, H=H, status=status,
        PrCond=PrCond, PrIDK=PrIDK, n_samples=n,
    )
