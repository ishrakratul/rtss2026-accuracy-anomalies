"""
test_toy_example.py

Correctness check of the calibration threshold rule and cascade accuracy
computation, against the paper's hand-worked 10-sample toy example (Table I,
Section II, Example 2). This does NOT require the real CSV or any trained
models -- it is a fast, self-contained sanity check.

This test calls src/cascade/calibration.py's build_calibration (the exact
function used to produce every table/figure elsewhere in this artifact) on
a tiny in-memory DataFrame shaped like the toy example, rather than
reimplementing the threshold rule locally -- so a bug in the real
calibration code fails this test too.

Paper claims (Example 2 / Section II, Observations 1 and 2):
    psi=0.65 -> H1=0.79, H2=0.66, cascade K1->K2 accuracy=0.6   (< psi, Obs. 1)
    psi=0.70 -> H1=0.88, H2=0.66, cascade K1->K2 accuracy=0.5   (< 0.6, Obs. 2)

Run with:
    python -m pytest test_toy_example.py -v
or:
    python test_toy_example.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.cascade.calibration import build_calibration

# Paper Table I: (K1_correct, K1_conf, K2_correct, K2_conf), sorted by confidence desc.
TABLE_I = [
    (True, 0.95, True, 0.99),
    (True, 0.92, True, 0.97),
    (True, 0.91, True, 0.93),
    (False, 0.91, True, 0.93),
    (True, 0.88, True, 0.86),
    (True, 0.88, True, 0.86),
    (False, 0.88, True, 0.86),
    (True, 0.79, False, 0.72),
    (False, 0.79, False, 0.72),
    (False, 0.73, False, 0.66),
]

# build_calibration reads CSV-style columns named "{model}_confidence" /
# "{model}_correct" / "{model}_inference_time"; inference_time isn't part of
# the toy example so it's filled with zeros (unused by threshold/status logic).
_TOY_CLASSIFIERS = [("K1", "ToyM1"), ("K2", "ToyM2")]
_TOY_DF = pd.DataFrame({
    "ToyM1_confidence": [r[1] for r in TABLE_I],
    "ToyM1_correct": [r[0] for r in TABLE_I],
    "ToyM1_inference_time": [0.0] * len(TABLE_I),
    "ToyM2_confidence": [r[3] for r in TABLE_I],
    "ToyM2_correct": [r[2] for r in TABLE_I],
    "ToyM2_inference_time": [0.0] * len(TABLE_I),
})


def cascade_accuracy(k1_status, k2_status):
    """Cascade K1 -> K2: use K1's outcome unless it's IDK, then use K2's.
    This combinator is the toy example's own cascade semantics (paper
    Example 2); it has no standalone equivalent in src/cascade since the
    real algorithms operate on probabilistic Calibration objects rather
    than replaying a single fixed cascade over raw per-sample status."""
    n = len(k1_status)
    correct = sum(
        1 for i in range(n)
        if (k1_status[i] if k1_status[i] != "IDK" else k2_status[i]) == "correct"
    )
    return correct / n


def test_psi_065():
    cal = build_calibration(_TOY_DF, _TOY_CLASSIFIERS, psi=0.65)
    assert abs(cal.H["K1"] - 0.79) < 1e-9, f"expected H1=0.79, got {cal.H['K1']}"
    assert abs(cal.H["K2"] - 0.66) < 1e-9, f"expected H2=0.66, got {cal.H['K2']}"

    acc = cascade_accuracy(cal.status["K1"], cal.status["K2"])
    assert abs(acc - 0.6) < 1e-9, f"expected cascade accuracy 0.6, got {acc}"


def test_psi_070_observation_2():
    """Paper Observation 2: raising psi from 0.65 to 0.70 DECREASES cascade
    accuracy (0.6 -> 0.5), even though it raises each classifier's own
    success accuracy."""
    cal = build_calibration(_TOY_DF, _TOY_CLASSIFIERS, psi=0.70)
    assert abs(cal.H["K1"] - 0.88) < 1e-9, f"expected H1=0.88, got {cal.H['K1']}"
    assert abs(cal.H["K2"] - 0.66) < 1e-9, f"expected H2=0.66, got {cal.H['K2']}"

    acc = cascade_accuracy(cal.status["K1"], cal.status["K2"])
    assert abs(acc - 0.5) < 1e-9, f"expected cascade accuracy 0.5, got {acc}"

    # Observation 2 itself: accuracy at psi=0.70 must be LOWER than at psi=0.65.
    cal_065 = build_calibration(_TOY_DF, _TOY_CLASSIFIERS, psi=0.65)
    acc_065 = cascade_accuracy(cal_065.status["K1"], cal_065.status["K2"])
    assert acc < acc_065, "Observation 2 violated: raising psi should decrease cascade accuracy here"


if __name__ == "__main__":
    test_psi_065()
    test_psi_070_observation_2()
    print("All toy-example tests passed.")
    print("psi=0.65: H1=0.79  H2=0.66  cascade accuracy=0.6000  (matches paper Table II)")
    print("psi=0.70: H1=0.88  H2=0.66  cascade accuracy=0.5000  (matches paper Table III)")
