"""
fig3_runtime_complexity.py

Reproduces paper Fig. 3: average measured runtime of Alg1, Alg2, accObl, and
a brute-force n! baseline, as the number of classifiers n = |K| increases.

This uses synthetic ("dummy-probability") problem instances rather than the
real 6-classifier calibration, because the figure's purpose is to show
worst-case *algorithmic* runtime scaling as n grows well past 6. No data-dependent shortcuts are taken: every loop iteration in each algorithm's pseudocode always executes,
so the measured runtime reflects the O(n*2^n) (DP algorithms) / O(n!)
(brute force) worst case.

Usage:
    python fig3_runtime_complexity.py --outdir ../../outputs/figures
    python fig3_runtime_complexity.py --outdir ../../outputs/figures --quick
        # quick mode: smaller N range and fewer repeats, for fast smoke-testing
"""
import argparse
import gc
import itertools
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, LogLocator, NullFormatter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils import print_title

DUMMY_SEED = 12345

# int.bit_count() was added in Python 3.10; fall back to a bin()-based
# popcount on 3.9 so this benchmark runs unmodified on either version.
if sys.version_info >= (3, 10):
    def _popcount(x: int) -> int:
        return x.bit_count()
else:
    def _popcount(x: int) -> int:
        return bin(x).count("1")


# ---------------------------------------------------------------------------
# Dummy-probability problem generation
# ---------------------------------------------------------------------------
@dataclass
class DummyProblem:
    n: int
    C: np.ndarray
    p_idk: np.ndarray
    p_correct: np.ndarray
    p_wrong: np.ndarray
    acc_if_answer: np.ndarray
    PrIDK: np.ndarray
    subset_cost: np.ndarray
    D: float


def _build_subset_arrays(C, p_idk):
    n = len(C)
    m_count = 1 << n
    PrIDK = np.ones(m_count, dtype=np.float64)
    subset_cost = np.zeros(m_count, dtype=np.float64)
    for mask in range(1, m_count):
        lsb = mask & -mask
        i = lsb.bit_length() - 1
        prev = mask ^ lsb
        PrIDK[mask] = PrIDK[prev] * p_idk[i]
        subset_cost[mask] = subset_cost[prev] + C[i]
    return PrIDK, subset_cost


def make_dummy_problem(n, seed=DUMMY_SEED):
    rng = np.random.default_rng(seed + 1009 * n)
    C = rng.uniform(0.001, 0.010, size=n)
    p_idk = rng.uniform(0.05, 0.30, size=n)
    acc_if_answer = rng.uniform(0.60, 0.95, size=n)
    p_correct = (1.0 - p_idk) * acc_if_answer
    p_wrong = (1.0 - p_idk) * (1.0 - acc_if_answer)
    PrIDK, subset_cost = _build_subset_arrays(C, p_idk)
    D = float(C.sum() + 1.0)  # every subset feasible -> deadline never shortcuts loops
    return DummyProblem(n=n, C=C, p_idk=p_idk, p_correct=p_correct, p_wrong=p_wrong,
                         acc_if_answer=acc_if_answer, PrIDK=PrIDK, subset_cost=subset_cost, D=D)


def expected_metrics_for_order(order, problem):
    prefix_mask = 0
    expected_time = 0.0
    expected_accuracy = 0.0
    for i in order:
        reach_prob = problem.PrIDK[prefix_mask]
        expected_time += problem.C[i] * reach_prob
        expected_accuracy += problem.p_correct[i] * reach_prob
        prefix_mask |= (1 << i)
    return expected_time, expected_accuracy


# ---------------------------------------------------------------------------
# Worst-case-path implementations (every loop iteration always executes --
# no early exits, no infeasibility skips, no deadline pruning)
# ---------------------------------------------------------------------------
def run_algorithm1_dummy(problem):
    n = problem.n
    m_count = 1 << n
    acc = np.zeros(m_count, dtype=np.float64)
    last = np.full(m_count, -1, dtype=np.int16)

    for subset_size in range(1, n + 1):
        for S in range(m_count):
            if _popcount(S) != subset_size:
                continue
            best_acc, best_i = 0.0, -1
            m = S
            while m:
                lsb = m & -m
                i = lsb.bit_length() - 1
                S_prev = S ^ lsb
                this_acc = acc[S_prev] + problem.p_correct[i] * problem.PrIDK[S_prev]
                if this_acc > best_acc:
                    best_acc, best_i = this_acc, i
                m ^= lsb
            acc[S] = best_acc
            last[S] = best_i

    best_S, best_acc = 0, -np.inf
    for S in range(m_count):
        _casWCET = problem.subset_cost[S]  # computed but not used to skip (worst-case path)
        if acc[S] > best_acc:
            best_acc, best_S = acc[S], S

    order = []
    S = best_S
    while S:
        i = int(last[S])
        order.append(i)
        S ^= (1 << i)
    order.reverse()

    expected_time, expected_accuracy = expected_metrics_for_order(order, problem)
    return {"order": order, "expected_time": expected_time, "expected_accuracy": expected_accuracy}


def run_algorithm2_dummy(problem, alpha=0.60):
    n = problem.n
    m_count = 1 << n
    K_mask = m_count - 1
    pre_cost = np.full(m_count, np.inf, dtype=np.float64)
    pre_last = np.full(m_count, -1, dtype=np.int16)
    pre_cost[0] = 0.0

    for subset_size in range(1, n):
        for S in range(m_count):
            if _popcount(S) != subset_size:
                continue
            best_cost, best_i = np.inf, -1
            m = S
            while m:
                lsb = m & -m
                i = lsb.bit_length() - 1
                S_prev = S ^ lsb
                _stage_acc = problem.acc_if_answer[i]  # computed, not gated (worst-case path)
                this_cost = pre_cost[S_prev] + problem.C[i] * problem.PrIDK[S_prev]
                if this_cost < best_cost:
                    best_cost, best_i = this_cost, i
                m ^= lsb
            pre_cost[S] = best_cost
            pre_last[S] = best_i

    cas_cost, cas_last, cas_pre_set = np.inf, -1, 0
    for S in range(m_count):
        if S == K_mask:
            continue
        preCasWCET = problem.subset_cost[S]
        pre_idk_S = problem.PrIDK[S]
        pre_cost_S = pre_cost[S]
        m_not_S = K_mask ^ S
        while m_not_S:
            lsb = m_not_S & -m_not_S
            i = lsb.bit_length() - 1
            _casWCET = preCasWCET + problem.C[i]
            _terminal_stage_acc = problem.acc_if_answer[i]
            this_cost = pre_cost_S + problem.C[i] * pre_idk_S
            if this_cost < cas_cost:
                cas_cost, cas_last, cas_pre_set = this_cost, i, S
            m_not_S ^= lsb

    if cas_last < 0 or not np.isfinite(cas_cost):
        return {"order": [], "expected_time": np.nan, "expected_accuracy": np.nan}

    order = [int(cas_last)]
    S = cas_pre_set
    while S:
        i = int(pre_last[S])
        if i < 0:
            break
        order.append(i)
        S ^= (1 << i)
    order.reverse()
    expected_time, expected_accuracy = expected_metrics_for_order(order, problem)
    return {"order": order, "expected_time": expected_time, "expected_accuracy": expected_accuracy}


def run_algorithm7_dummy(problem):
    n = problem.n
    m_count = 1 << n
    C_det = float(problem.C.mean())
    det_accuracy = 0.90

    cost = np.full(m_count, np.inf, dtype=np.float64)
    last = np.full(m_count, -1, dtype=np.int16)
    cost[0] = 0.0

    for subset_size in range(1, n + 1):
        for S in range(m_count):
            if _popcount(S) != subset_size:
                continue
            best_cost, best_i = np.inf, -1
            m = S
            while m:
                lsb = m & -m
                i = lsb.bit_length() - 1
                S_prev = S ^ lsb
                this_cost = cost[S_prev] + problem.C[i] * problem.PrIDK[S_prev]
                if this_cost < best_cost:
                    best_cost, best_i = this_cost, i
                m ^= lsb
            cost[S] = best_cost
            last[S] = best_i

    best_S, best_cost = 0, np.inf
    for S in range(m_count):
        _candidate_wcet = problem.subset_cost[S] + C_det
        this_cost = cost[S] + C_det * problem.PrIDK[S]
        if this_cost < best_cost:
            best_cost, best_S = this_cost, S

    prefix_order = []
    S = best_S
    while S:
        i = int(last[S])
        prefix_order.append(i)
        S ^= (1 << i)
    prefix_order.reverse()

    prefix_mask, expected_time, expected_accuracy = 0, 0.0, 0.0
    for i in prefix_order:
        reach_prob = problem.PrIDK[prefix_mask]
        expected_time += problem.C[i] * reach_prob
        expected_accuracy += problem.p_correct[i] * reach_prob
        prefix_mask |= (1 << i)
    expected_time += C_det * problem.PrIDK[prefix_mask]
    expected_accuracy += det_accuracy * problem.PrIDK[prefix_mask]

    return {"order": prefix_order + ["K_det"], "expected_time": expected_time, "expected_accuracy": expected_accuracy}


def run_bruteforce_n_factorial_dummy(problem):
    n = problem.n
    best_order, best_accuracy, best_time = None, -np.inf, np.inf
    permutations_checked = 0
    for order in itertools.permutations(range(n)):
        permutations_checked += 1
        expected_time, expected_accuracy = expected_metrics_for_order(order, problem)
        if (expected_accuracy > best_accuracy) or (
            np.isclose(expected_accuracy, best_accuracy) and expected_time < best_time
        ):
            best_accuracy, best_time, best_order = expected_accuracy, expected_time, order
    return {"order": list(best_order), "expected_time": best_time,
            "expected_accuracy": best_accuracy, "permutations_checked": permutations_checked}


# ---------------------------------------------------------------------------
# Timing harness
# ---------------------------------------------------------------------------
def average_processing_time_ms(fn, repeats, warmup_runs):
    for _ in range(warmup_runs):
        fn()
    times_ms, last_output = [], None
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeats):
            t0 = time.perf_counter_ns()
            last_output = fn()
            t1 = time.perf_counter_ns()
            times_ms.append((t1 - t0) / 1_000_000.0)
    finally:
        if gc_was_enabled:
            gc.enable()
    return last_output, {
        "avg_ms": float(np.mean(times_ms)), "std_ms": float(np.std(times_ms, ddof=0)),
        "min_ms": float(np.min(times_ms)), "max_ms": float(np.max(times_ms)), "repeats": repeats,
    }


def _skip_row(algorithm_name, n, reason):
    return {"algorithm": algorithm_name, "n": n, "avg_processing_time_ms": np.nan,
            "std_ms": np.nan, "min_ms": np.nan, "max_ms": np.nan, "repeats": 0,
            "expected_classification_time": np.nan, "expected_accuracy": np.nan,
            "selected_order": None, "skipped": True, "skip_reason": reason}


def benchmark_algorithm(algorithm_name, runner, n_values, max_n, repeats, warmup_runs, stop_after_avg_seconds):
    rows = []
    stopped_after_slow_point = False
    for n in n_values:
        if n > max_n:
            rows.append(_skip_row(algorithm_name, n, f"skipped because n>{max_n}; too expensive"))
            continue
        if stopped_after_slow_point:
            rows.append(_skip_row(algorithm_name, n, f"skipped: previous point exceeded {stop_after_avg_seconds}s"))
            continue
        problem = make_dummy_problem(n)
        output, stats = average_processing_time_ms(lambda: runner(problem), repeats=repeats, warmup_runs=warmup_runs)
        avg_seconds = stats["avg_ms"] / 1000.0
        if stop_after_avg_seconds is not None and avg_seconds > stop_after_avg_seconds:
            stopped_after_slow_point = True
        rows.append({
            "algorithm": algorithm_name, "n": n,
            "avg_processing_time_ms": stats["avg_ms"], "std_ms": stats["std_ms"],
            "min_ms": stats["min_ms"], "max_ms": stats["max_ms"], "repeats": stats["repeats"],
            "expected_classification_time": output.get("expected_time", np.nan),
            "expected_accuracy": output.get("expected_accuracy", np.nan),
            "selected_order": str(output.get("order", None)), "skipped": False, "skip_reason": "",
        })
        print(f"{algorithm_name:18s} n={n:2d}: avg={stats['avg_ms']:.6f} ms over {stats['repeats']} runs")
    return rows


def make_plot(timing_df, n_values, outpath, max_dp_n):
    plot_order = ["Algorithm 1", "Algorithm 2", "Algorithm 7", "Brute force n!"]
    DISPLAY_LABEL = {"Algorithm 1": "Alg1", "Algorithm 2": "Alg2", "Algorithm 7": "accObl", "Brute force n!": "brtFrc"}
    LINE_COLORS = {"Algorithm 1": "#3A6EA5", "Algorithm 2": "#B22222", "Algorithm 7": "#2E7D32", "Brute force n!": "#111111"}
    LINE_MARKERS = {"Algorithm 1": "o", "Algorithm 2": "s", "Algorithm 7": "X", "Brute force n!": "^"}
    GRID_COLOR = "#E8E8E8"

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 22, "font.weight": "bold",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.facecolor": "#FFFFFF", "figure.facecolor": "#FFFFFF",
        "axes.edgecolor": "#BBBBBB", "axes.linewidth": 0.9,
        "xtick.major.size": 0, "ytick.major.size": 4,
        "xtick.major.pad": 6, "ytick.major.pad": 6,
    })

    fig, ax = plt.subplots(figsize=(14, 10), dpi=150)
    for algorithm_name in plot_order:
        d = timing_df[(timing_df["algorithm"] == algorithm_name) & (~timing_df["skipped"])].copy()
        if len(d) == 0:
            continue
        d = d.sort_values("n")
        ax.plot(d["n"], d["avg_processing_time_ms"], marker=LINE_MARKERS[algorithm_name],
                markersize=18, markerfacecolor=LINE_COLORS[algorithm_name], markeredgecolor="white",
                markeredgewidth=0.8, linewidth=2.6, color=LINE_COLORS[algorithm_name],
                label=DISPLAY_LABEL[algorithm_name], zorder=3)

    ax.set_xlabel(r"Number of Classifiers, $n = |K|$", fontweight="bold", labelpad=10)
    ax.set_ylabel("Average Runtime (ms)", fontweight="bold", labelpad=10)
    # Cap the x-axis at the largest n that actually has DP-algorithm data
    # (n_values may extend further, but Alg1/Alg2/Alg7 are skipped beyond
    # max_dp_n -- stretching the axis to n_values' max leaves the right
    # portion of the plot empty, as in the paper's own Fig. 3 which caps
    # around n=18).
    x_max = min(max(n_values), max_dp_n + 2)
    ax.set_xlim(min(n_values), x_max)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=range(2, 10)))
    ax.yaxis.set_minor_formatter(NullFormatter())

    ax.grid(axis="y", which="major", color=GRID_COLOR, linewidth=0.9, zorder=0)
    ax.grid(axis="y", which="minor", color=GRID_COLOR, linewidth=0.45, alpha=0.55, zorder=0)
    ax.grid(axis="x", which="major", color=GRID_COLOR, linewidth=0.6, alpha=0.65, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)
    for tick in ax.get_xticklabels():
        tick.set_fontweight("bold")
    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")

    legend = ax.legend(frameon=True, loc="lower right", borderpad=0.8, handlelength=2.4, labelspacing=0.6)
    legend.get_frame().set_facecolor("#FFFFFF")
    legend.get_frame().set_edgecolor("#BBBBBB")
    legend.get_frame().set_linewidth(0.8)
    for text in legend.get_texts():
        text.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#BBBBBB")
        spine.set_linewidth(0.9)

    plt.tight_layout(pad=1.5)
    plt.savefig(outpath, bbox_inches="tight", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="../../outputs/figures")
    ap.add_argument("--quick", action="store_true",
                     help="Reduced N range / repeats for a fast smoke test (minutes, not hours).")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.quick:
        N_VALUES = list(range(2, 13))
        MAX_DP_N = 12
        MAX_BRUTE_FORCE_N = 7
        REPEATS, WARMUP = 3, 2
        STOP_AFTER_AVG_SECONDS = 10.0
    else:
        N_VALUES = list(range(2, 31))
        MAX_DP_N = 16
        MAX_BRUTE_FORCE_N = 9
        REPEATS, WARMUP = 10, 5
        STOP_AFTER_AVG_SECONDS = 100000.0

    print_title(f"Fig. 3: Runtime complexity experiment ({'quick' if args.quick else 'full'} mode)")
    print(f"n values : {N_VALUES[0]}..{N_VALUES[-1]}   MAX_DP_N={MAX_DP_N}   MAX_BRUTE_FORCE_N={MAX_BRUTE_FORCE_N}")

    all_rows = []
    all_rows += benchmark_algorithm("Algorithm 1", run_algorithm1_dummy, N_VALUES, MAX_DP_N, REPEATS, WARMUP, STOP_AFTER_AVG_SECONDS)
    all_rows += benchmark_algorithm("Algorithm 2", run_algorithm2_dummy, N_VALUES, MAX_DP_N, REPEATS, WARMUP, STOP_AFTER_AVG_SECONDS)
    all_rows += benchmark_algorithm("Algorithm 7", run_algorithm7_dummy, N_VALUES, MAX_DP_N, REPEATS, WARMUP, STOP_AFTER_AVG_SECONDS)
    all_rows += benchmark_algorithm("Brute force n!", run_bruteforce_n_factorial_dummy, N_VALUES, MAX_BRUTE_FORCE_N, REPEATS, WARMUP, STOP_AFTER_AVG_SECONDS)

    timing_df = pd.DataFrame(all_rows)
    csv_path = os.path.join(args.outdir, "fig3_runtime_timing.csv")
    timing_df.to_csv(csv_path, index=False)
    print(f"\nSaved timing table: {csv_path}")

    plot_path = os.path.join(args.outdir, "fig3_runtime_complexity.png")
    make_plot(timing_df, N_VALUES, plot_path, MAX_DP_N)
    print(f"Saved figure: {plot_path}")


if __name__ == "__main__":
    main()
