"""
table7.py

Reproduces paper Table VII in full: synthesized cascades (Alg1, Alg2 at
alpha in {0.70, 0.75, 0.80}, accObl, and brute-force baselines) at
psi = 0.85, D = 150 ms -- including BOTH the Expected Accuracy/Time
columns (from the calibration's PrCond/PrIDK tables) AND the Observed
Accuracy/Time columns the paper reports (measured by re-executing each
synthesized cascade on 2,000 randomly sampled images, paper Section V-B).

Also reproduces the stage-accuracy case study (paper Definition 4) that
explains why Alg2(alpha=0.75) does not select K4 -> K5 -> K1 -> K6 even
though its overall expected accuracy exceeds 0.75.

Usage:
    python table7.py --data ../../data/all_6_models_inference_results_RTSS.csv \
        --outdir ../../outputs/tables
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.cascade import (
    ALL_CLASSIFIERS, build_calibration, algorithm1, algorithm2, algorithm7,
    brute_force_search, print_cascade, stage_accuracy_table, observed_metrics,
)
from src.utils import print_title, print_subtitle

ALPHA_SWEEP = (0.70, 0.75, 0.80)


def add_row(rows, name, r, cal, df, n_sample, seed, stratify):
    if r["Cas"] is None:
        rows.append({"algorithm": name, "cascade": None,
                     "expected_accuracy": None, "expected_time_ms": None, "wcet_ms": None,
                     "observed_accuracy": None, "observed_time_ms": None})
        return
    obs_acc, obs_t = observed_metrics(r["Cas"], cal, df, n_sample=n_sample, seed=seed, stratify_by_class=stratify)
    rows.append({
        "algorithm": name,
        "cascade": " -> ".join(r["Cas"]),
        "expected_accuracy": round(r["expected_accuracy"], 4),
        "expected_time_ms": round(r["expected_time"] * 1000, 3),
        "wcet_ms": round(r["wcet"] * 1000, 3),
        "observed_accuracy": round(obs_acc, 4),
        "observed_time_ms": round(obs_t * 1000, 3),
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../../data/all_6_models_inference_results_RTSS.csv")
    ap.add_argument("--outdir", default="../../outputs/tables")
    ap.add_argument("--psi", type=float, default=0.85)
    ap.add_argument("--deadline_ms", type=float, default=150.0)
    ap.add_argument("--k_det", default="K6")
    ap.add_argument("--n_sample", type=int, default=2000,
                     help="Number of randomly re-sampled images for Observed Accuracy/Time (paper Section V-B).")
    ap.add_argument("--seed", type=int, default=0,
                     help="RNG seed for the 2000-sample re-execution. The paper does not fix/report a seed, "
                          "so exact Observed values will differ run-to-run; Expected values are exact regardless.")
    ap.add_argument("--no_stratify", action="store_true",
                     help="Disable per-class stratified sampling for Observed Accuracy/Time (default draws "
                          "exactly n_sample // n_classes images from EACH of the 200 Tiny ImageNet classes, "
                          "since 2000/200=10 exactly; pass this flag for plain uniform sampling instead).")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.data)
    D = args.deadline_ms / 1000.0

    cal = build_calibration(df, ALL_CLASSIFIERS, args.psi)

    print_title(f"Table VII: Synthesized cascades at psi={args.psi}, D={args.deadline_ms:.0f} ms")
    print("Per-classifier worst-case Ci and average-case C_bar (ms):")
    for ki in cal.labels:
        print(f"  {ki}: Ci = {cal.Ci[ki]*1000:7.3f}   C_bar = {cal.C_bar[ki]*1000:7.3f}   H = {cal.H[ki]:.4f}")
    print()

    rows = []

    r1 = algorithm1(cal, D)
    print_cascade("Alg1", r1, cal)
    print()
    add_row(rows, "Alg1", r1, cal, df, args.n_sample, args.seed, not args.no_stratify)

    r7 = algorithm7(cal, D, args.k_det)
    print_cascade("accObl", r7, cal)
    print()
    add_row(rows, "accObl", r7, cal, df, args.n_sample, args.seed, not args.no_stratify)

    r2_by_alpha = {}
    for a in ALPHA_SWEEP:
        r2 = algorithm2(cal, D, a)
        r2_by_alpha[a] = r2
        print_cascade(f"Alg2 (alpha={a})", r2, cal)
        print()
        add_row(rows, f"Alg2 (alpha={a})", r2, cal, df, args.n_sample, args.seed, not args.no_stratify)

    # Brute-force cross-check (max accuracy, and best-time subject to accuracy>=threshold)
    best_max_acc, best_by_threshold = brute_force_search(cal, D, ALPHA_SWEEP)

    print("=" * 80)
    print_subtitle(f"Brute-force cross-check at psi = {args.psi} (D = {args.deadline_ms:.0f} ms)")
    print("=" * 80)
    print_cascade("brtFrc (max accuracy)", best_max_acc, cal)
    print()
    add_row(rows, "brtFrc (max accuracy)", best_max_acc, cal, df, args.n_sample, args.seed, not args.no_stratify)

    for t in ALPHA_SWEEP:
        res = best_by_threshold[t]
        name = f"brtFrc (accuracy >= {t:.2f})"
        if res["Cas"] is None:
            print(f"{name}: infeasible")
            rows.append({"algorithm": name, "cascade": None,
                         "expected_accuracy": None, "expected_time_ms": None, "wcet_ms": None,
                         "observed_accuracy": None, "observed_time_ms": None})
            continue
        print_cascade(name, res, cal)
        print()
        add_row(rows, name, res, cal, df, args.n_sample, args.seed, not args.no_stratify)

    table7_df = pd.DataFrame(rows)
    print_title("Full Table VII (Expected + Observed, paper columns)")
    print(table7_df.to_string(index=False))

    out_path = os.path.join(args.outdir, "table7_synthesized_cascades.csv")
    table7_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # Stage-accuracy breakdown for the case discussed in the paper text
    # (Section VI / case study): K4 -> K5 -> K1 -> K6 has overall accuracy
    # > 0.75 but Alg2(alpha=0.75) does NOT select it, because its stage
    # accuracy at K5 and K1 falls below 0.75.
    print()
    print_subtitle("Stage-accuracy breakdown: K4 -> K5 -> K1 -> K6  (psi=0.85)")
    tbl = stage_accuracy_table(["K4", "K5", "K1", "K6"], cal)
    print(tbl.to_string(index=False))
    tbl_path = os.path.join(args.outdir, "table7_stage_accuracy_case_study.csv")
    tbl.to_csv(tbl_path, index=False)
    print(f"Saved: {tbl_path}")


if __name__ == "__main__":
    main()
