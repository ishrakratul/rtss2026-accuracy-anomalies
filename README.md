# RTSS 2026 Artifact: Accuracy Anomalies in Classifier Cascades

Code and data for **"Accuracy Anomalies in Classifier Cascades"** by
Sanjoy Baruah, Ishrak Jahan Ratul, and Kecheng Yang (alphabetical author
order), 47th IEEE Real-Time Systems Symposium (RTSS 2026).
This artifact passed RTSS 2026 artifact evaluation.

- **Trained checkpoints (~1.3 GB):** on Zenodo, see
  [Getting the trained models](#getting-the-trained-models). Not needed
  for Option A.
- **Dataset:** Tiny-ImageNet-200 downloads automatically from its
  official source when Option B or C needs it.

This artifact reproduces the paper's **Tables IV–VII**, the **Section V-A
per-classifier table**, and **Figures 2 and 3**, using measurements from
six trained image classifiers.

**Fastest path:** install, then run one command (Option A below). No GPU
needed, takes a few minutes.

**Tested on:** Python 3.9–3.11, Linux. Training was validated on an
NVIDIA RTX A4000 (16 GB VRAM).

---

## Step 1 — Install

For Option A (reproducing tables and figures — no GPU):

```bash
git clone https://github.com/ishrakratul/rtss2026-accuracy-anomalies.git
cd rtss2026-accuracy-anomalies
pip install -r requirements.txt
```

For Option B or C (inference or training — needs PyTorch and a GPU):

```bash
pip install -r requirements-train.txt
```

`requirements-train.txt` already includes everything in
`requirements.txt`, so you only ever run one of these.

---

## Step 2 — Run one of three options

### Option A — reproduce the tables and figures (recommended)

Uses the measurement data already shipped in `data/`. No GPU, no
training, no dataset download.

```bash
python3 run.py --mode algorithms --target all
```

Takes about 2–3 minutes on our machine. Your time will differ — Figure
3's timing sweep is the slow part and depends on your CPU.

### Option B — re-run inference using the trained models

First get the checkpoints (see
[Getting the trained models](#getting-the-trained-models)), then:

```bash
python3 scripts/download_models.py
python3 run.py --mode inference --tiny_root ./tiny-imagenet-200
python3 run.py --mode algorithms --target all
```

### Option C — train all six models from scratch

Needs a GPU and takes several hours.

```bash
python3 run.py --mode train --tiny_root ./tiny-imagenet-200 --pretrained --amp
python3 run.py --mode inference --tiny_root ./tiny-imagenet-200
python3 run.py --mode algorithms --target all
```

**If training runs out of GPU memory:** add `--batch_size 32` (or `16`)
to the train command. The default of `64` needs roughly 16 GB of VRAM,
so smaller cards will fail without this.

Options B and C download Tiny-ImageNet-200 (~237 MB) automatically if
it is not already present.

---

## Step 3 — Find your results

Each output file corresponds to one item in the paper:

| Output file | Paper item |
|---|---|
| `outputs/tables/classifier_summary_Ci_Cbar_accuracy.csv` | Section V-A table (Ci / C̄i / Accuracy), the one just after Fig. 2 |
| `outputs/tables/table4_prcond_example.csv` | Table IV |
| `outputs/tables/table5_expected_accuracy.csv` | Table V |
| `outputs/tables/table6_expected_time_ms.csv` | Table VI |
| `outputs/tables/table7_synthesized_cascades.csv` | Table VII |
| `outputs/tables/table7_stage_accuracy_case_study.csv` | Table VII's stage-accuracy case study |
| `outputs/figures/fig2_execution_time.png` / `.pdf` | Figure 2 |
| `outputs/figures/fig3_runtime_complexity.png` | Figure 3 |

To regenerate just one item, replace `all` with `table4`, `table5_6`,
`table7`, `fig2`, or `fig3`:

```bash
python3 run.py --mode algorithms --target table4
```

---

## Step 4 — Check the code is working

```bash
python3 -m pytest tests/ -v
```

This runs the paper's hand-worked toy example (Section II) through
`src/cascade/calibration.py`'s real `build_calibration` function — the
same function used to produce every table in this artifact, not a copy
of it.

---

## What matches the paper, and what does not

**These match the paper exactly:**

- Table IV
- Tables V and VI — all 9 algorithm settings (Alg1, Alg2 at α = 0.70 /
  0.75 / 0.80, accObl, and all 4 brute-force baselines)
- Table VII's synthesized cascades and its **Expected** columns
- The Section V-A per-classifier table
- Figure 2

**These are measurements, so they will differ slightly:**

- **Table VII's "Observed" columns.** The paper's original 2,000-image
  sample was not retained, so the artifact draws its own (seed 0,
  class-stratified). On our runs the results land within 1.6 percentage
  points of accuracy and 4.3 ms of the paper's.
- **Figure 3's millisecond values.** These depend on your CPU. The
  *shape* of the result is what matters and it matches the paper: brute
  force diverges sharply from the three DP algorithms as n grows.

**Two notes on Figure 3:**

- It measures *worst-case algorithmic* complexity. The timed versions
  deliberately skip the deadline and accuracy shortcuts that the real
  algorithms use, because those shortcuts make runtime depend on the
  specific input rather than on n. This is what lets the O(n·2ⁿ) versus
  O(n!) trend show cleanly.
- Add `--quick` to any `algorithms` command for a faster, smaller sweep.

**On dependency versions:** the floors in `requirements.txt`
(`pandas>=1.3`, `matplotlib>=3.5`) are simply the oldest versions we
have confirmed, not an upper limit. Newer versions work too — we have
also tested pandas 2.3.3 and matplotlib 3.10.7.

---

## Getting the trained models

The six trained checkpoints (about 1.3 GB, CC-BY-4.0) are archived on
Zenodo: **https://doi.org/10.5281/zenodo.24243817**

```bash
python3 scripts/download_models.py
```

This downloads them into `models/pretrained/` with the exact filenames
the code expects (already-present files are skipped) and verifies each
file's MD5 checksum. You can also download them from Zenodo by hand and
copy them there. Option A does not need the checkpoints.

---

## Citation

If you use this artifact, please cite the paper (see also
`CITATION.cff`):

```bibtex
@inproceedings{baruah2026accuracy,
  title     = {Accuracy Anomalies in Classifier Cascades},
  author    = {Baruah, Sanjoy and Ratul, Ishrak Jahan and Yang, Kecheng},
  booktitle = {Proceedings of the 47th IEEE Real-Time Systems Symposium (RTSS)},
  year      = {2026}
}
```

---

## What's in this folder

```
rtss2026-accuracy-anomalies/
├── run.py                   # the only script you need to run
├── requirements.txt         # Option A dependencies
├── requirements-train.txt   # adds PyTorch etc. for Options B and C
├── data/                    # the paper's per-classifier measurements
├── models/pretrained/       # checkpoints go here (downloaded from Zenodo)
├── scripts/download_models.py  # downloads the checkpoints from Zenodo
├── src/
│   ├── cascade/             # the core engine: calibration.py + algorithms.py
│   ├── figures/             # one generator per table and figure
│   ├── inference/           # Option B: run models over the dataset
│   ├── training/            # Option C: train the six classifiers
│   ├── data_prep/           # builds the test-label CSV from Tiny-ImageNet
│   └── utils/               # dataset loading and console helpers
├── tests/                   # toy-example correctness check
└── outputs/                 # your results appear here
```
