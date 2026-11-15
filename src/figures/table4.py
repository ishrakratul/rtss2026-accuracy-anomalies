"""
table4.py

Reproduces paper Table IV: an example PrCond(Ki, S, RES) table for |K|=3,
using K1=AlexNet, K2=ResNet18, K3=ConvNeXt (as named in the paper text,
Section III) computed from the real Tiny ImageNet calibration data.

Reuses src/cascade/calibration.py's build_calibration -- the same
calibration machinery verified against paper Table V/VI/VII -- restricted
to the 3-classifier subset and formatted exactly as paper Table IV
(Row | Ki | S | correct | incorrect | IDK, with S enumerated in the
paper's canonical order: empty, {K1}, {K2}, {K1,K2}, {K3}, {K1,K3},
{K2,K3}, {K1,K2,K3}).

Usage:
    python table4.py --data ../../data/all_6_models_inference_results_RTSS.csv \
        --outdir ../../outputs/tables --psi 0.80
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.cascade import build_calibration, members
from src.utils import print_title

TABLE4_CLASSIFIERS = [("K1", "AlexNet"), ("K2", "ResNet18"), ("K3", "ConvNeXt")]


def build_table4(cal, labels):
    """Row order matches paper Table IV exactly: for each Ki (K1, K2, K3),
    enumerate S in canonical binary order over the OTHER two classifiers
    only (paper Table IV footnote: rows with Ki in S are shown but carry
    no information, PrCond(Ki,S,IDK)=1 vacuously) -- but the paper's own
    table *does* include those rows for completeness, so we do too, in
    the same order the paper's binary enumeration implies:
    000, 100, 010, 110, 001, 101, 011, 111 (bit i = whether Ki is in S)."""
    nK = len(labels)
    rows = []
    row_id = 0
    for ki in labels:
        for mask in range(1 << nK):
            S_members = members(mask, labels)
            S_label = "{" + ", ".join(S_members) + "}" if S_members else "\u2205"
            pc = cal.PrCond[(ki, "correct", mask)]
            pw = cal.PrCond[(ki, "incorrect", mask)]
            pi = cal.PrCond[(ki, "IDK", mask)]
            rows.append({
                "Row": row_id, "Ki": ki, "S": S_label,
                "correct": round(pc, 4), "incorrect": round(pw, 4), "IDK": round(pi, 4),
            })
            row_id += 1
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../../data/all_6_models_inference_results_RTSS.csv")
    ap.add_argument("--outdir", default="../../outputs/tables")
    ap.add_argument("--psi", type=float, default=0.80,
                     help="Precision threshold used to calibrate H_i for K1/K2/K3 (paper Section I).")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.data)

    cal = build_calibration(df, TABLE4_CLASSIFIERS, args.psi)
    labels = cal.labels

    print_title(f"Table IV: PrCond(Ki, S, RES) example, |K|=3  (K1=AlexNet, K2=ResNet18, K3=ConvNeXt, psi={args.psi})")
    print(f"Confidence thresholds: " + ", ".join(f"H({k})={cal.H[k]:.4f}" for k in labels))
    print()

    table4 = build_table4(cal, labels)
    print(table4.to_string(index=False))

    out_path = os.path.join(args.outdir, "table4_prcond_example.csv")
    table4.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
