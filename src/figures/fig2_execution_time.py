"""
fig2_execution_time.py

Reproduces paper Fig. 2: distribution of per-sample execution times for the
six classifiers, with mean (red line) and worst-case (black star) annotated.

Usage:
    python fig2_execution_time.py --data ../../data/all_6_models_inference_results_RTSS.csv \
        --outdir ../../outputs/figures
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils import print_title

MODELS_AND_LABELS = [
    ("K1", "AlexNet"),
    ("K2", "EfficientNet"),
    ("K3", "VGG16"),
    ("K4", "ResNet18"),
    ("K5", "ConvNeXt"),
    ("K6", "Swin\nTransformer"),
]
ORIG_COL_NAMES = {"SwinTransformer": "Swin_Transformer"}

FIG_WIDTH, FIG_HEIGHT = 3.5, 3.7

BOX_COLOR = "#3A6EA5"
BOX_FACE = "#DCE9F7"
MEAN_COLOR = "#C62828"
STAR_COLOR = "#111111"
WHISKER_COLOR = "#666666"
GRID_COLOR = "#E3E3E3"
TEXT_COLOR = "#111111"

LABEL_SIZE = TICK_SIZE = KI_SIZE = MODEL_SIZE_ = LEGEND_SIZE = 6
ANNOT_SIZE = 5.5

AXIS_LW = 0.6
BOX_LW = 0.85
MEAN_LW = 1.0
GRID_LW = 0.35
BOX_WIDTH = 0.42
STAR_MS = 7.0


def _as_bool(series: pd.Series) -> np.ndarray:
    """Coerce a `*_correct` column to bool (same rule as calibration.py)."""
    if series.dtype == bool:
        return series.to_numpy()
    return series.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    ).astype(bool).to_numpy()


def make_figure(df: pd.DataFrame, outdir: str):
    execution_time_data = []
    summary_rows = []

    for Ki, model in MODELS_AND_LABELS:
        col_model = model.replace("\n", "")
        col_model = ORIG_COL_NAMES.get(col_model, col_model)
        time_col = f"{col_model}_inference_time"
        if time_col not in df.columns:
            raise ValueError(f"Missing column: {time_col}")

        times_ms = df[time_col].dropna().astype(float).to_numpy() * 1000.0
        execution_time_data.append(times_ms)
        summary_rows.append({
            "Ki": Ki, "Model": model,
            "Mean": np.mean(times_ms),
            "P99": np.percentile(times_ms, 99),
            "Ci": np.max(times_ms),
            "Accuracy": float(_as_bool(df[f"{col_model}_correct"]).mean()),
        })

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.weight": "bold",
        "mathtext.fontset": "dejavusans",
        "font.size": LABEL_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "axes.labelweight": "bold",
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "axes.linewidth": AXIS_LW,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=300)
    positions = np.arange(1, len(MODELS_AND_LABELS) + 1)

    fig.subplots_adjust(left=0.20, right=0.985, top=0.985, bottom=0.40)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bp = ax.boxplot(
        execution_time_data, positions=positions, widths=BOX_WIDTH,
        patch_artist=True, showfliers=False, showmeans=True, meanline=True,
        medianprops=dict(color="#333333", linewidth=BOX_LW),
        meanprops=dict(color=MEAN_COLOR, linewidth=MEAN_LW, linestyle="-"),
        whiskerprops=dict(color=WHISKER_COLOR, linewidth=BOX_LW),
        capprops=dict(color=WHISKER_COLOR, linewidth=BOX_LW),
        boxprops=dict(color=BOX_COLOR, linewidth=BOX_LW),
        zorder=3,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(BOX_FACE)
        patch.set_edgecolor(BOX_COLOR)
        patch.set_linewidth(BOX_LW)

    mean_offsets = {1: (7, -1), 2: (-7, -1), 3: (-7, -1), 4: (7, -1), 5: (7, -1), 6: (-7, -1)}
    ci_offsets = {1: (0, 8), 2: (0, 8), 3: (0, 8), 4: (0, 8), 5: (0, 8), 6: (0, 10)}
    label_box = dict(boxstyle="square,pad=0.08", fc="white", ec="none", alpha=0.88)

    for idx, row in enumerate(summary_df.itertuples(), start=1):
        x = idx
        mean_val, ci_val = row.Mean, row.Ci

        ax.plot(x, ci_val, marker="*", markersize=STAR_MS,
                color=STAR_COLOR, markeredgecolor=STAR_COLOR, linestyle="None", zorder=7)
        dx, dy = ci_offsets[idx]
        ax.annotate(f"{ci_val:.1f}", xy=(x, ci_val), xytext=(dx, dy), textcoords="offset points",
                    ha="center" if dx == 0 else ("left" if dx > 0 else "right"), va="bottom",
                    fontsize=ANNOT_SIZE, fontweight="bold", color=TEXT_COLOR, bbox=label_box, zorder=10)

        dx, dy = mean_offsets[idx]
        ax.annotate(f"{mean_val:.1f}", xy=(x, mean_val), xytext=(dx, dy), textcoords="offset points",
                    ha="left" if dx >= 0 else "right", va="center",
                    fontsize=ANNOT_SIZE, fontweight="bold", color=TEXT_COLOR, bbox=label_box, zorder=10)

    ax.set_xlim(0.45, 6.60)
    ax.set_ylim(0, 90)
    ax.set_ylabel("Execution time per sample (ms)", labelpad=2, fontweight="bold")
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(axis="y", which="major", color=GRID_COLOR, linewidth=GRID_LW, linestyle="-", zorder=0)
    ax.grid(axis="y", which="minor", color=GRID_COLOR, linewidth=0.22, linestyle="-", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xticks(positions)
    ax.set_xticklabels([""] * len(positions))
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", labelsize=TICK_SIZE, pad=6)
    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")

    trans = blended_transform_factory(ax.transData, ax.transAxes)
    ki_y_axes = -0.045
    model_y_axes = -0.110
    for pos, (ki, model_name) in zip(positions, MODELS_AND_LABELS):
        i = int(ki[1])
        ax.text(pos, ki_y_axes, rf"$K_{{{i}}}$", transform=trans, ha="center", va="top",
                fontsize=KI_SIZE, fontweight="bold", color=TEXT_COLOR, clip_on=False)
        ax.text(pos, model_y_axes, model_name, transform=trans, ha="right", va="top",
                rotation=20, rotation_mode="anchor",
                fontsize=MODEL_SIZE_ - 1, fontweight="bold", color=TEXT_COLOR, clip_on=False)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#7A7A7A")
        spine.set_linewidth(AXIS_LW)

    legend_handles = [
        Line2D([0], [0], color=MEAN_COLOR, lw=MEAN_LW, label="mean"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor=STAR_COLOR,
               markeredgecolor=STAR_COLOR, markersize=STAR_MS, label="observed maximum"),
    ]
    legend = ax.legend(handles=legend_handles, loc="upper left", frameon=True,
                        borderpad=0.18, handlelength=1.1, handletextpad=0.35, labelspacing=0.18)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#B8B8B8")
    legend.get_frame().set_linewidth(0.45)
    legend.get_frame().set_alpha(0.96)
    for txt in legend.get_texts():
        txt.set_fontweight("bold")

    os.makedirs(outdir, exist_ok=True)
    pdf_path = os.path.join(outdir, "fig2_execution_time.pdf")
    png_path = os.path.join(outdir, "fig2_execution_time.png")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path, summary_df


def save_classifier_summary(summary_df: pd.DataFrame, tables_outdir: str):
    """Write the paper's per-classifier summary table (Section V-A, the
    unlabeled table immediately following Fig. 2), which reports Ci
    (worst-case), C_bar (average) and the individual classifier accuracy
    for each of K1..K6. Laid out with classifiers as columns, exactly as
    the paper presents it."""
    os.makedirs(tables_outdir, exist_ok=True)
    table = pd.DataFrame(
        {
            row.Ki: [round(row.Ci, 1), round(row.Mean, 1), round(row.Accuracy, 4)]
            for row in summary_df.itertuples()
        },
        index=["Ci", "C_bar", "Accuracy"],
    )
    table.index.name = "Classifier"
    out_path = os.path.join(tables_outdir, "classifier_summary_Ci_Cbar_accuracy.csv")
    table.to_csv(out_path)
    print()
    print_title("Per-classifier summary (paper Section V-A, table following Fig. 2)")
    print(table.to_string())
    return out_path, table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../../data/all_6_models_inference_results_RTSS.csv")
    ap.add_argument("--outdir", default="../../outputs/figures")
    ap.add_argument("--tables_outdir", default=None,
                     help="Where to write the Section V-A per-classifier summary CSV. "
                          "Defaults to a 'tables' directory alongside --outdir.")
    args = ap.parse_args()

    print_title("Fig. 2: Per-classifier execution-time distribution (IEEE style)")
    df = pd.read_csv(args.data)
    pdf_path, png_path, summary_df = make_figure(df, args.outdir)
    print(f"\nSaved: {pdf_path}")
    print(f"Saved: {png_path}")

    tables_outdir = args.tables_outdir or os.path.join(os.path.dirname(os.path.abspath(args.outdir)), "tables")
    summary_path, _ = save_classifier_summary(summary_df, tables_outdir)
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
