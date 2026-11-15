"""
algorithms.py

Cascade-synthesis algorithms from *Accuracy Anomalies in Classifier Cascades*:

  - algorithm1: maximize overall accuracy subject to a hard deadline (paper
    Algorithm 1, Section IV-A).
  - algorithm2: minimize expected classification time subject to a hard
    deadline and a minimum stage accuracy alpha (paper Algorithm 2,
    Section IV-B).
  - algorithm7 (accObl): the accuracy-oblivious baseline from [12], reproduced
    here with K_det as a regular IDK classifier.
  - brute_force_search: exhaustive O(n!) search used as a correctness check
    on the three DP algorithms above.

"""
from itertools import combinations, permutations

from .calibration import Calibration, members, remove, subsets_of_size
from ..utils import print_subtitle


# ---------------------------------------------------------------------------
# Cascade evaluation (paper Equations 8 and 9)
# ---------------------------------------------------------------------------
def expected_metrics(cas: list, calib: Calibration):
    """Return (expected_accuracy, expected_classification_time) for a cascade,
    following paper Equations 9 (accuracy) and 8 (time).

    Eq. 8 uses AVERAGE-CASE execution time C_bar_i (not worst-case Ci).
    Every classifier is treated as an IDK classifier; no special case.
    """
    exp_acc, exp_t = 0.0, 0.0
    prefix_mask = 0
    for ki in cas:
        exp_t += calib.C_bar[ki] * calib.PrIDK[prefix_mask]
        exp_acc += calib.PrCond[(ki, "correct", prefix_mask)] * calib.PrIDK[prefix_mask]
        prefix_mask |= 1 << calib.labels.index(ki)
    return exp_acc, exp_t


def wcet(cas: list, calib: Calibration) -> float:
    """Worst-case execution time = sum of WORST-CASE per-classifier times Ci."""
    return sum(calib.Ci[k] for k in cas)


def print_cascade(name: str, result: dict, calib: Calibration):
    """Print a cascade with its per-classifier confidence threshold H."""
    if result["Cas"] is None:
        print(f"{name:18s}  infeasible")
        return
    cas = result["Cas"]
    parts = [f"{ki}(H={calib.H[ki]:.4f})" for ki in cas]
    cas_str = " -> ".join(parts)
    print_subtitle(name)
    print(f"  Cascade : {cas_str}")
    print(f"  Acc     : {result['expected_accuracy']:.4f}")
    print(f"  E[t]    : {result['expected_time']*1000:.2f} ms")
    print(f"  WCET    : {result['wcet']*1000:.2f} ms")


# ---------------------------------------------------------------------------
# Algorithm 1 -- maximize overall accuracy subject to a hard deadline
# ---------------------------------------------------------------------------
def algorithm1(calib: Calibration, D: float) -> dict:
    """Algorithm 1 (paper Section IV-A): maximize overall accuracy
    subject to a hard deadline D. Implementation follows the pseudocode
    literally; see paper Algorithm 1, lines 1-30.

    Cost model: line 17 uses WORST-CASE execution time Ci for the
    deadline-feasibility check casWCET <= D.
    """
    K_labels = calib.labels
    nK = len(K_labels)
    Ci, PrCond, PrIDK = calib.Ci, calib.PrCond, calib.PrIDK

    acc = {0: 0.0}      # line 1
    last = {0: None}    # line 2

    for nS in range(1, nK + 1):                                       # line 3
        for S in subsets_of_size(nK, nS):                             # line 4
            acc[S] = 0.0                                              # line 5
            last[S] = None                                            # line 6
            for Ki in members(S, K_labels):                           # line 7
                S_prime = remove(S, Ki, K_labels)                     # line 8
                thisAcc = acc[S_prime] + PrCond[(Ki, "correct", S_prime)] * PrIDK[S_prime]  # line 9
                if acc[S] < thisAcc:                                  # line 10
                    acc[S] = thisAcc                                  # line 11
                    last[S] = Ki                                      # line 12

    casAcc = 0.0      # line 13
    casLast = None    # line 14
    casSet = 0         # line 15
    for S in range(1 << nK):                                          # line 16
        casWCET = sum(Ci[Ki] for Ki in members(S, K_labels))          # line 17
        if casWCET <= D:                                              # line 18
            if casAcc < acc[S]:                                       # line 19
                casAcc = acc[S]                                       # line 20
                casLast = last[S]                                     # line 21
                casSet = S                                            # line 22

    if casAcc == 0.0:                                                 # line 23
        return {"Cas": None, "expected_accuracy": 0.0,                # line 24
                "expected_time": None, "wcet": None}

    Cas = [casLast]                                                   # line 25
    S = remove(casSet, casLast, K_labels)                             # line 26 (corrected: uses casLast)
    while S != 0:                                                     # line 27
        Cas = [last[S]] + Cas                                         # line 28
        S = remove(S, last[S], K_labels)                              # line 29
    # line 30: return Cas

    exp_acc, exp_t = expected_metrics(Cas, calib)
    return {"Cas": Cas, "expected_accuracy": exp_acc,
            "expected_time": exp_t, "wcet": wcet(Cas, calib)}


# ---------------------------------------------------------------------------
# Algorithm 2 -- minimize expected time s.t. stage accuracy alpha
# ---------------------------------------------------------------------------
def algorithm2(calib: Calibration, D: float, alpha: float) -> dict:
    """Algorithm 2 (paper Section IV-B): minimize expected classification
    duration subject to a hard deadline D and minimal stage-accuracy
    alpha at every stage. Implementation follows the pseudocode;
    see paper Algorithm 2, lines 1-37.

    Cost model:
      - Lines 19/21 (preCasWCET, casWCET, deadline check): WORST-CASE Ci.
      - Lines 11/25 (thisCost, expected-cost objective): AVERAGE-CASE C_bar.
    """
    K_labels = calib.labels
    nK = len(K_labels)
    Ci, C_bar, PrCond, PrIDK = calib.Ci, calib.C_bar, calib.PrCond, calib.PrIDK
    INF = float("inf")

    preCost = {0: 0.0}      # line 1
    preLast = {0: None}     # line 2

    for nS in range(1, nK):                                           # line 3
        for S in subsets_of_size(nK, nS):                             # line 4
            preCost[S] = INF                                          # line 5
            preLast[S] = None                                         # line 6
            for Ki in members(S, K_labels):                           # line 7
                S_prime = remove(S, Ki, K_labels)                     # line 8
                p_correct = PrCond[(Ki, "correct", S_prime)]
                p_wrong = PrCond[(Ki, "incorrect", S_prime)]
                # line 9 (paper's exact piecewise definition): stageAcc = 0
                # when PrCond(Ki,S',correct) == 0 (guards the 0/0 case too,
                # since p_wrong would then also be 0), else the normal ratio.
                if p_correct == 0.0:
                    stageAcc = 0.0
                else:
                    stageAcc = p_correct / (p_correct + p_wrong)
                if stageAcc >= alpha:                                 # line 10
                    thisCost = preCost[S_prime] + C_bar[Ki] * PrIDK[S_prime]  # line 11
                    if preCost[S] > thisCost:                         # line 12
                        preCost[S] = thisCost                         # line 13
                        preLast[S] = Ki                                # line 14

    casCost = INF        # line 15
    casLast = None        # line 16
    casPreSet = 0          # line 17

    for S in range(1 << nK):                                          # line 18
        preCasWCET = sum(Ci[Kj] for Kj in members(S, K_labels))       # line 19 (corrected:  casWCET)
        for Ki in K_labels:                                           # line 20
            if (S >> K_labels.index(Ki)) & 1:
                continue                                              # Ki must not be in S
            casWCET = preCasWCET + Ci[Ki]                             # line 21 (corrected: preCasWCET + Ci)
            if casWCET <= D:                                          # line 22
                stageAcc = PrCond[(Ki, "correct", S)]                 # line 23
                if stageAcc >= alpha:                                 # line 24
                    thisCost = preCost[S] + C_bar[Ki] * PrIDK[S]      # line 25
                    if casCost > thisCost:                            # line 26
                        casCost = thisCost                            # line 27
                        casLast = Ki                                  # line 28
                        casPreSet = S                                 # line 29

    if casCost == INF:                                                # line 30
        return {"Cas": None, "expected_accuracy": 0.0,                # line 31
                "expected_time": None, "wcet": None}

    Cas = [casLast]                                                   # line 32
    S = casPreSet                                                     # line 33
    while S != 0:                                                     # line 34
        Cas = [preLast[S]] + Cas                                      # line 35
        S = remove(S, preLast[S], K_labels)                           # line 36
    # line 37: return Cas

    exp_acc, exp_t = expected_metrics(Cas, calib)
    return {"Cas": Cas, "expected_accuracy": exp_acc,
            "expected_time": exp_t, "wcet": wcet(Cas, calib)}


# ---------------------------------------------------------------------------
# Algorithm 7 -- accObl (K_det is a regular IDK classifier)
# ---------------------------------------------------------------------------
def algorithm7(calib: Calibration, D: float, k_det: str) -> dict:
    """Cost model:
      - Line 16 (casWCET, deadline check): WORST-CASE Ci.
      - Lines 9/11/18 (expected-cost objective): AVERAGE-CASE C_bar.
    """
    K_labels = calib.labels
    Ci, C_bar, PrCond, PrIDK = calib.Ci, calib.C_bar, calib.PrCond, calib.PrIDK
    INF = float("inf")
    C_det_wcet = Ci[k_det]      # worst-case, for deadline check (line 16)
    C_det_mean = C_bar[k_det]   # average-case, for expected-cost objective (line 18)

    prefix_labels = [k for k in K_labels if k != k_det]
    n_prefix = len(prefix_labels)

    def prefix_mask_to_full(mask_pref: int) -> int:
        full = 0
        for i, k in enumerate(prefix_labels):
            if mask_pref & (1 << i):
                full |= 1 << K_labels.index(k)
        return full

    cost = {0: 0.0}      # line 1
    last = {0: None}     # line 2

    for nS in range(1, n_prefix + 1):                                 # line 3
        for S_pref in subsets_of_size(n_prefix, nS):                  # line 4
            cost[S_pref] = INF                                        # line 5
            last[S_pref] = None                                       # line 6
            for Ki in members(S_pref, prefix_labels):                 # line 7
                S_prime_pref = remove(S_pref, Ki, prefix_labels)
                S_prime_full = prefix_mask_to_full(S_prime_pref)
                thisCost = cost[S_prime_pref] + C_bar[Ki] * PrIDK[S_prime_full]  # line 9/11
                if cost[S_pref] > thisCost:                           # line 10
                    cost[S_pref] = thisCost                           # line 11
                    last[S_pref] = Ki                                  # line 12

    casCost = INF        # line 13
    casPreSet = 0          # line 14
    for S_pref in range(1 << n_prefix):                               # line 15
        S_full = prefix_mask_to_full(S_pref)
        casWCET = sum(Ci[Ki] for Ki in members(S_pref, prefix_labels)) + C_det_wcet  # line 16
        if casWCET <= D:                                              # line 17
            thisCost = cost[S_pref] + C_det_mean * PrIDK[S_full]      # line 18
            if casCost > thisCost:                                    # line 19
                casCost = thisCost                                    # line 20
                casPreSet = S_pref                                    # line 21

    if casCost == INF:                                                # line 22
        return {"Cas": None, "expected_accuracy": 0.0,                # line 23
                "expected_time": None, "wcet": None}

    Cas = [k_det]                                                     # line 24
    S_pref = casPreSet                                                # line 25
    while S_pref != 0:                                                # line 26
        Cas = [last[S_pref]] + Cas                                    # line 27
        S_pref = remove(S_pref, last[S_pref], prefix_labels)          # line 28
    # line 29: return Cas

    exp_acc, exp_t = expected_metrics(Cas, calib)
    return {"Cas": Cas, "expected_accuracy": exp_acc,
            "expected_time": exp_t, "wcet": wcet(Cas, calib)}


# ---------------------------------------------------------------------------
# Brute-force verification (exhaustive O(n!) search, used as a
# correctness check on Algorithms 1/2/7 -- NOT one of the paper's
# proposed algorithms)
# ---------------------------------------------------------------------------
def brute_force_search(calib: Calibration, D: float, thresholds=(0.70, 0.75, 0.80)):
    """Exhaustive search over ALL cascades (every ordering of every non-empty
    subset of calib.labels), subject to the hard deadline D (WCET <= D).

    Returns
    -------
    best_max_acc : dict        -- cascade with the highest Exp Accuracy
    best_by_threshold : dict   -- {t: cascade with ExpAcc >= t and minimal ExpTime}
    """
    K_labels = calib.labels
    nK = len(K_labels)

    best_max_acc = {"Cas": None, "expected_accuracy": -1.0,
                     "expected_time": None, "wcet": None}
    best_by_threshold = {
        t: {"Cas": None, "expected_accuracy": None,
            "expected_time": float("inf"), "wcet": None}
        for t in thresholds
    }

    for size in range(1, nK + 1):
        for subset in combinations(K_labels, size):
            for cas in permutations(subset):
                cas = list(cas)
                w = wcet(cas, calib)
                if w > D:
                    continue

                exp_acc, exp_t = expected_metrics(cas, calib)

                if exp_acc > best_max_acc["expected_accuracy"]:
                    best_max_acc = {"Cas": cas, "expected_accuracy": exp_acc,
                                     "expected_time": exp_t, "wcet": w}

                for t in thresholds:
                    if exp_acc >= t and exp_t < best_by_threshold[t]["expected_time"]:
                        best_by_threshold[t] = {"Cas": cas, "expected_accuracy": exp_acc,
                                                  "expected_time": exp_t, "wcet": w}

    return best_max_acc, best_by_threshold


def stage_accuracy_table(cascade, calib: Calibration, alpha_values=(0.70, 0.75, 0.80)):
    """Return a stage-accuracy table for a cascade (paper Definition 4).

    For each stage Ki with prefix S = {classifiers before Ki, all IDK'd}:
        stageAcc = PrCond(Ki, S, correct) / (PrCond(Ki, S, correct) + PrCond(Ki, S, incorrect))
    """
    import pandas as pd

    labels = calib.labels
    PrCond, PrIDK = calib.PrCond, calib.PrIDK

    def mask_of(ks):
        m = 0
        for k in ks:
            m |= 1 << labels.index(k)
        return m

    rows = []
    prefix_ks = []
    for idx, Ki in enumerate(cascade):
        is_terminal = (idx == len(cascade) - 1)
        S_mask = mask_of(prefix_ks)

        pc = PrCond[(Ki, "correct", S_mask)]
        pw = PrCond[(Ki, "incorrect", S_mask)]
        pi = PrCond[(Ki, "IDK", S_mask)]
        pridk_S = PrIDK[S_mask]
        stageAcc = 0.0 if pc == 0.0 else pc / (pc + pw)

        row = {
            "Stage": idx + 1,
            "Classifier": Ki,
            "Type": "terminal" if is_terminal else "prefix",
            "PrIDK(S)": round(pridk_S, 4),
            "PrCond(correct)": round(pc, 4),
            "PrCond(incorrect)": round(pw, 4),
            "PrCond(IDK)": round(pi, 4),
            "stageAcc": round(stageAcc, 4),
        }
        for a in alpha_values:
            row[f"a={a:.2f}"] = "PASS" if stageAcc >= a else "FAIL"
        rows.append(row)
        prefix_ks.append(Ki)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Observed metrics -- paper Table VII's "Observed Accuracy" / "Observed
# Classification Time" columns. Paper Section V-B: "These observed
# measurements were obtained by executing the cascade for 2,000 randomly
# selected images out of the 10,000 images, from which the probability
# tables were built."
# ---------------------------------------------------------------------------
def observed_metrics(cascade: list, calib: Calibration, df, n_sample: int = 2000, seed: int = 0,
                      stratify_by_class: bool = True, class_col: str = "true_label"):
    """Re-execute a synthesized cascade on a random subset of n_sample rows
    Returns (observed_accuracy, observed_time_seconds).
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    n_total = calib.n_samples
    n_sample = min(n_sample, n_total)

    if stratify_by_class and class_col in df.columns:
        classes = df[class_col].to_numpy()
        unique_classes = np.unique(classes)
        n_per_class = n_sample // len(unique_classes)
        remainder = n_sample - n_per_class * len(unique_classes)
        idx_parts = []
        for i, c in enumerate(unique_classes):
            class_idx = np.where(classes == c)[0]
            take = n_per_class + (1 if i < remainder else 0)
            take = min(take, len(class_idx))
            idx_parts.append(rng.choice(class_idx, size=take, replace=False))
        idx = np.concatenate(idx_parts)
        rng.shuffle(idx)
    else:
        idx = rng.choice(n_total, size=n_sample, replace=False)

    n_sample = len(idx)
    done = np.zeros(n_sample, dtype=bool)
    correct = np.zeros(n_sample, dtype=bool)
    total_time = np.zeros(n_sample, dtype=float)

    for ki in cascade:
        model_col = calib.models[calib.labels.index(ki)]
        status_ki = calib.status[ki][idx]
        time_ki = df[f"{model_col}_inference_time"].to_numpy()[idx]

        # Every not-yet-resolved sample invokes this classifier and pays its cost.
        total_time[~done] += time_ki[~done]

        newly_resolved = (~done) & (status_ki != "IDK")
        correct[newly_resolved] = (status_ki[newly_resolved] == "correct")
        done = done | newly_resolved

    observed_accuracy = float(correct.mean())
    observed_time = float(total_time.mean())
    return observed_accuracy, observed_time
