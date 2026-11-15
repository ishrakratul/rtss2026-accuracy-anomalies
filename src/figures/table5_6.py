"""
table5_6.py

Reproduces paper Table V (expected accuracy vs psi) and Table VI (expected
classification time vs psi), D = 150 ms, for Alg1, Alg2 (alpha in
{0.70, 0.75, 0.80}), and accObl.

Usage:
    python table5_6.py --data ../../data/all_6_models_inference_results_RTSS.csv \
        --outdir ../../outputs/tables
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.cascade import ALL_CLASSIFIERS, build_calibration, algorithm1, algorithm2, algorithm7, brute_force_search
from src.utils import print_title


def generate(df: pd.DataFrame, D_fixed: float = 0.150, k_det: str = "K6"):
    PSI_SWEEP = [0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    ALPHA_SWEEP = [0.70, 0.75, 0.80]
    BRTFRC_COLS = ["brtFrc (max accuracy)"] + [f"brtFrc (accuracy >= {a:.2f})" for a in ALPHA_SWEEP]

    acc_rows, time_rows = [], []
    cascade_details = []

    for psi in PSI_SWEEP:
        cal = build_calibration(df, ALL_CLASSIFIERS, psi)

        r1 = algorithm1(cal, D_fixed)
        r7 = algorithm7(cal, D_fixed, k_det)

        acc_row = {"psi": psi}
        time_row = {"psi": psi}

        acc_row["Alg1"] = r1["expected_accuracy"] if r1["Cas"] else None
        time_row["Alg1"] = r1["expected_time"] * 1000 if r1["Cas"] else None
        cascade_details.append((psi, "Alg1", r1, cal))

        for a in ALPHA_SWEEP:
            r2 = algorithm2(cal, D_fixed, a)
            acc_row[f"Alg2 a={a}"] = r2["expected_accuracy"] if r2["Cas"] else None
            time_row[f"Alg2 a={a}"] = r2["expected_time"] * 1000 if r2["Cas"] else None
            cascade_details.append((psi, f"Alg2 a={a}", r2, cal))

        acc_row["accObl"] = r7["expected_accuracy"] if r7["Cas"] else None
        time_row["accObl"] = r7["expected_time"] * 1000 if r7["Cas"] else None
        cascade_details.append((psi, "accObl", r7, cal))

        # Brute-force cross-check (paper's optimality claim for Alg1, and the
        # best-time-subject-to-accuracy-threshold comparison for Alg2) --
        # previously only reported in Table VII; included here too so Tables
        # V/VI carry all nine algorithm settings the paper compares.
        best_max_acc, best_by_threshold = brute_force_search(cal, D_fixed, ALPHA_SWEEP)
        acc_row["brtFrc (max accuracy)"] = best_max_acc["expected_accuracy"] if best_max_acc["Cas"] else None
        time_row["brtFrc (max accuracy)"] = best_max_acc["expected_time"] * 1000 if best_max_acc["Cas"] else None
        cascade_details.append((psi, "brtFrc (max accuracy)", best_max_acc, cal))

        for a in ALPHA_SWEEP:
            res = best_by_threshold[a]
            col = f"brtFrc (accuracy >= {a:.2f})"
            acc_row[col] = res["expected_accuracy"] if res["Cas"] else None
            time_row[col] = res["expected_time"] * 1000 if res["Cas"] else None
            cascade_details.append((psi, col, res, cal))

        acc_rows.append(acc_row)
        time_rows.append(time_row)

    cols = ["Alg1"] + [f"Alg2 a={a}" for a in ALPHA_SWEEP] + ["accObl"] + BRTFRC_COLS
    table_acc = pd.DataFrame(acc_rows).set_index("psi")[cols]
    table_time = pd.DataFrame(time_rows).set_index("psi")[cols]
    return table_acc, table_time, cascade_details


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../../data/all_6_models_inference_results_RTSS.csv")
    ap.add_argument("--outdir", default="../../outputs/tables")
    ap.add_argument("--deadline_ms", type=float, default=150.0)
    ap.add_argument("--k_det", default="K6")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.data)

    table_acc, table_time, details = generate(df, D_fixed=args.deadline_ms / 1000.0, k_det=args.k_det)

    print_title(f"Table V: Expected accuracy  (D = {args.deadline_ms:.0f} ms, paper Eq. 9)")
    print(table_acc.round(4).to_string())
    print()
    print_title(f"Table VI: Expected classification time, ms  (D = {args.deadline_ms:.0f} ms, paper Eq. 8)")
    print(table_time.round(3).to_string())

    table_acc.round(4).to_csv(os.path.join(args.outdir, "table5_expected_accuracy.csv"))
    table_time.round(3).to_csv(os.path.join(args.outdir, "table6_expected_time_ms.csv"))
    print(f"\nSaved: {args.outdir}/table5_expected_accuracy.csv")
    print(f"Saved: {args.outdir}/table6_expected_time_ms.csv")


if __name__ == "__main__":
    main()
