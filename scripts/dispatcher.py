"""Experiment infrastructure shared by the experiment scripts.

LOADER       - the real datasets by name; each is a function
               input_dir -> (response_mat, gt_labels)
ModelResult  - result of one method run, ready for evaluation
evaluate()   - computes the metrics from a ModelResult and the ground truth
run_*        - one function per method: (response_mat, **options) -> ModelResult
METHODS      - the methods of the paper by name; each is a function
               response_mat -> ModelResult (RepAlg also needs `adv_frac`, the
               true fraction of adversaries)

Usage in an experiment script
-----------------------------
    result = METHODS[method](response_mat)
    row = evaluate(result, gt_adversaries, gt_targeted, gt_labels)
"""

from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from crowdkit.aggregation import GLAD
from sklearn import metrics as sk_metrics

import subcad
from subcad.thirdparty import ebcc_labels, la_one_pass_labels, la_two_pass_labels

MSCRIPTS_DIR = Path(subcad.PROJECT_DIR, "research", "mscripts")

# The real datasets by name; each is a function input_dir -> (response_mat, gt_labels)
LOADER = {
    "rte": lambda x: subcad.data.fetch_rte(x, return_X_y=True),
    "sp": lambda x: subcad.data.fetch_sp(x, return_X_y=True),
    "dog": lambda x: subcad.data.fetch_dog(x, return_X_y=True),
    "web": lambda x: subcad.data.fetch_web(x, return_X_y=True),
    "adult2": lambda x: subcad.data.fetch_adult2(x, return_X_y=True),
    "temp": lambda x: subcad.data.fetch_temp(x, return_X_y=True),
}


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class ModelResult:
    """Output of a single method run, ready for evaluation.

    Attributes
    ----------
    label : str
        Display name of the method (e.g. "Greedy++ / Weighted Dawid-Skene").
    params : dict
        Hyperparameter values used for this run (e.g. {"iterations": 5}).
    worker_scores : ndarray (M,) or None
        Per-worker adversary scores, e.g. a detector's `worker_scores_`.
    task_scores : ndarray (N,) or None
        Per-task adversary scores, e.g. a detector's `task_scores_`.
    n_adversaries_hat : int or None
        Estimated number of adversarial workers, e.g. a selector's output.
    n_targets_hat : int or None
        Estimated number of targeted tasks, e.g. a selector's output.
    labels_hat : ndarray (N,) or None
        Fused/aggregated task labels, e.g. an aggregator's output.
    """

    label: str
    params: dict
    worker_scores: npt.NDArray[np.float64] | None = None
    task_scores: npt.NDArray[np.float64] | None = None
    n_adversaries_hat: int | None = None
    n_targets_hat: int | None = None
    labels_hat: npt.NDArray | None = None


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def eval_worker_detection(
    gt_adversaries: npt.NDArray,
    worker_scores: npt.NDArray,
    row: dict,
) -> None:
    row["WorkerROCAUC"] = sk_metrics.roc_auc_score(gt_adversaries, worker_scores)
    row["WorkerAP"] = sk_metrics.average_precision_score(gt_adversaries, worker_scores)


def eval_task_detection(
    gt_targeted: npt.NDArray,
    task_scores: npt.NDArray,
    row: dict,
) -> None:
    row["TaskROCAUC"] = sk_metrics.roc_auc_score(gt_targeted, task_scores)
    row["TaskAP"] = sk_metrics.average_precision_score(gt_targeted, task_scores)


def eval_size_estimation(
    gt_adversaries: npt.NDArray,
    gt_targeted: npt.NDArray,
    n_adversaries_hat: int,
    n_targets_hat: int,
    row: dict,
) -> None:
    row["NAdversariesTrue"] = int(gt_adversaries.sum())
    row["NAdversariesHat"] = n_adversaries_hat
    row["NTargetsTrue"] = int(gt_targeted.sum())
    row["NTargetsHat"] = n_targets_hat


def eval_label_fusion(
    gt_labels: npt.NDArray,
    gt_targeted: npt.NDArray,
    labels_hat: npt.NDArray,
    row: dict,
) -> None:
    row["Accuracy"] = sk_metrics.accuracy_score(gt_labels, labels_hat)
    row["TargetAccuracy"] = sk_metrics.accuracy_score(
        gt_labels[gt_targeted == 1], labels_hat[gt_targeted == 1]
    )


def evaluate(
    result: ModelResult,
    gt_adversaries: npt.NDArray,
    gt_targeted: npt.NDArray,
    gt_labels: npt.NDArray,
) -> dict:
    """Compute all applicable metrics for one ModelResult.

    Parameters
    ----------
    result : ModelResult
    gt_adversaries : ndarray (M,) — ground-truth binary adversary labels.
    gt_targeted : ndarray (N,) — ground-truth binary targeted-task labels.
    gt_labels : ndarray (N,) — ground-truth task class labels.
    """
    row: dict = {"Method": result.label, **result.params}

    if result.worker_scores is not None:
        eval_worker_detection(gt_adversaries, result.worker_scores, row)
    if result.task_scores is not None:
        eval_task_detection(gt_targeted, result.task_scores, row)
    if result.n_adversaries_hat is not None and result.n_targets_hat is not None:
        eval_size_estimation(
            gt_adversaries,
            gt_targeted,
            result.n_adversaries_hat,
            result.n_targets_hat,
            row,
        )
    if result.labels_hat is not None:
        eval_label_fusion(gt_labels, gt_targeted, result.labels_hat, row)

    return row


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------


def run_subcad(
    response_mat: npt.NDArray,
    kind: str,
    detector: str,
    selector: str,
    aggregator: str,
    iterations: int | None = None,
    adv_frac: float | None = None,
    target_frac: float | None = None,
    scale: float = 5.0,
) -> ModelResult:
    """SubCAD: detect (rank), estimate the sizes, aggregate with weights.

    `kind` is the graph ("weighted" or "binary"), `detector` is "greedy",
    "greedypp" (with `iterations`) or "spectral", `selector` is "density" or
    "spectral" and `aggregator` is "mv" or "ds". `adv_frac` / `target_frac`
    (the kappas of the weights) are the fractions estimated by the selector
    unless given.
    """
    n_workers, n_tasks = response_mat.shape

    if detector == "greedy":
        det = subcad.GreedyDetector(kind=kind)
        detector_label = "Greedy"
        detector_params = {}
    elif detector == "greedypp":
        det = subcad.GreedyPPDetector(kind=kind, iterations=int(iterations))
        detector_label = "GreedyPP"
        detector_params = {"iterations": int(iterations)}
    elif detector == "spectral":
        det = subcad.SpectralDetector(kind=kind)
        detector_label = "Spectral"
        detector_params = {}
    else:
        raise ValueError(f"Unknown detector '{detector}'.")

    worker_scores, task_scores = det.fit_predict(response_mat)

    if selector == "density":
        sel = subcad.DensitySelector()
        selector_label = "Density"
    elif selector == "spectral":
        sel = subcad.SpectralSeededSelector()
        selector_label = "SpectralSeeded"
    else:
        raise ValueError(f"Unknown selector '{selector}'.")

    n_adversaries, n_targeted = sel.select(det.biadj_mat_, worker_scores, task_scores)

    if adv_frac is None:
        adv_frac = n_adversaries / n_workers
    if target_frac is None:
        target_frac = n_targeted / n_tasks
    worker_penalties, task_penalties = det.aggregation_penalties(
        float(adv_frac), float(target_frac), float(scale)
    )

    if aggregator == "mv":
        agg = subcad.WeightedMajorityVoting()
        aggregator_label = "MV"
    elif aggregator == "ds":
        agg = subcad.WeightedDawidSkene()
        aggregator_label = "DS"
    else:
        raise ValueError(f"Unknown aggregator '{aggregator}'.")

    labels = agg.fit_predict(response_mat, worker_penalties, task_penalties)

    return ModelResult(
        label=f"SubCAD{detector_label}{selector_label}{aggregator_label}",
        params={
            **detector_params,
            "adv_frac": float(adv_frac),
            "target_frac": float(target_frac),
            "scale": float(scale),
        },
        worker_scores=worker_scores,
        task_scores=task_scores,
        n_adversaries_hat=n_adversaries,
        n_targets_hat=n_targeted,
        labels_hat=labels,
    )


def _start_matlab_engine(method: str):
    """Import the MATLAB Engine for Python and start an engine session.

    Not a poetry dependency: `matlabengine` isn't a portable PyPI package
    (its version must match the caller's local MATLAB release), so it's
    imported lazily here rather than declared in `pyproject.toml`.
    """
    try:
        import matlab
        import matlab.engine
    except ImportError as e:
        raise ImportError(
            f"{method} requires the MATLAB Engine for Python, which ships "
            "with a MATLAB install but isn't a poetry dependency (it isn't "
            "a portable PyPI package -- its version must match your local "
            "MATLAB release). Install it into this project's virtualenv "
            "with `pip install <MATLABROOT>/extern/engines/python`, where "
            "MATLABROOT is your MATLAB install directory."
        ) from e

    eng = matlab.engine.start_matlab()
    eng.addpath(eng.genpath(str(MSCRIPTS_DIR)), nargout=0)
    eng.warning("off", "all", nargout=0)
    return matlab, eng


def _detect_dacs_groups(response_mat: npt.NDArray, rhos: npt.NDArray) -> npt.NDArray:
    """Run the MATLAB DACS implementation and return a hard 0/1 worker partition.

    Calls `research/mscripts/dacs/dacs.m`, then flips the partition (if
    needed) so `1` always denotes the smaller (adversary) cluster.
    """
    n_workers, _ = response_mat.shape
    n_classes = len(np.unique(response_mat[response_mat > 0]))

    matlab, eng = _start_matlab_engine("DACS")
    try:
        f = matlab.double(response_mat.tolist())
        rhos_mat = matlab.double(rhos.tolist())
        groups = eng.dacs(f, float(n_classes), rhos_mat, nargout=1)
        groups = np.array(groups, dtype=float).flatten()
    finally:
        eng.quit()

    # Take the smaller cluster as the adversary group
    if groups.sum() > n_workers - groups.sum():
        groups = 1 - groups
    return groups


def run_dacs(
    response_mat: npt.NDArray,
    rhos=(1.1, 2, 5, 10, 20, 50, 80, 100, 500, 800, 1000),
) -> ModelResult:
    """Detect adversaries with the MATLAB DACS implementation, then aggregate.

    Starts a MATLAB engine and calls `research/mscripts/dacs/dacs.m` to get a
    hard honest/adversary worker partition. Labels are fused with vanilla
    `subcad.DawidSkene` run on the workers DACS flags as honest.
    """
    rhos = np.asarray(rhos)
    worker_penalties = _detect_dacs_groups(response_mat, rhos)

    labels = subcad.DawidSkene().fit_predict(response_mat[worker_penalties == 0, :])

    return ModelResult(
        label="DACS",
        params={"rhos": rhos.tolist()},
        worker_scores=worker_penalties,
        labels_hat=labels,
    )


def run_mmsr(response_mat: npt.NDArray) -> ModelResult:
    """Detect adversaries with the MATLAB MMSR implementation, then aggregate.

    Starts a MATLAB engine and calls `research/mscripts/mmsr/mmsr.m` to get a
    continuous per-worker reliability score `probWorker` (higher means more
    honest) and `predlabel`.
    """
    n_classes = len(np.unique(response_mat[response_mat > 0]))

    matlab, eng = _start_matlab_engine("MMSR")
    try:
        f = matlab.double(response_mat.tolist())
        prob_worker, predlabel = eng.mmsr(f, float(n_classes), nargout=2)
        prob_worker = np.array(prob_worker, dtype=float).flatten()
        labels = np.array(predlabel, dtype=int).flatten()
    finally:
        eng.quit()

    worker_scores = np.clip(1 - prob_worker, 0, 1)

    return ModelResult(label="MMSR", params={}, worker_scores=worker_scores, labels_hat=labels)


def run_repalg(response_mat: npt.NDArray, detector: str, adv_frac: float) -> ModelResult:
    """Detect adversaries with RepAlg ("soft" or "hard" penalty), then aggregate.

    Neither RepAlg algorithm estimates how many workers are adversarial, so
    the `round(adv_frac * n_workers)` highest-scoring workers are eliminated
    (`adv_frac` is the true fraction in the experiments, i.e. an oracle) and
    labels are fused with vanilla `subcad.DawidSkene` on the remaining workers.
    """
    class_ids = np.unique(response_mat[response_mat > 0])
    if len(class_ids) != 2:
        raise ValueError(
            f"RepAlg only supports binary classification, got {len(class_ids)} classes."
        )

    n_workers, _ = response_mat.shape

    if detector == "soft":
        det = subcad.SoftPenaltyDetector()
        detector_label = "RepAlgSoft"
    elif detector == "hard":
        det = subcad.HardPenaltyDetector()
        detector_label = "RepAlgHard"
    else:
        raise ValueError(f"Unknown detector '{detector}'.")

    scores = det.fit_predict(response_mat)

    n_adv = int(round(float(adv_frac) * n_workers))
    is_adversary = np.zeros(n_workers, dtype=bool)
    if n_adv > 0:
        adv_idx = np.argpartition(scores, -n_adv)[-n_adv:]
        is_adversary[adv_idx] = True

    labels = subcad.DawidSkene().fit_predict(response_mat[~is_adversary, :])

    return ModelResult(
        label=detector_label,
        params={"detector": detector, "adv_frac": float(adv_frac)},
        worker_scores=scores,
        labels_hat=labels,
    )


def _response_mat_to_crowdkit_df(response_mat: npt.NDArray) -> pd.DataFrame:
    worker_idx, task_idx = np.nonzero(response_mat)
    return pd.DataFrame(
        {
            "task": task_idx,
            "worker": worker_idx,
            "label": response_mat[worker_idx, task_idx],
        }
    )


def run_glad(response_mat: npt.NDArray) -> ModelResult:
    """Aggregate task labels with GLAD (`crowdkit.aggregation.GLAD`)."""
    n_tasks = response_mat.shape[1]
    df = _response_mat_to_crowdkit_df(response_mat)
    labels = GLAD().fit_predict(df).reindex(range(n_tasks)).to_numpy()

    return ModelResult(label="GLAD", params={}, labels_hat=labels)


def run_ebcc(response_mat: npt.NDArray) -> ModelResult:
    """Aggregate task labels with EBCC (`subcad.thirdparty.ebcc_labels`)."""
    return ModelResult(label="EBCC", params={}, labels_hat=ebcc_labels(response_mat))


def run_la_one_pass(response_mat: npt.NDArray) -> ModelResult:
    """Aggregate task labels with the LA one-pass baseline."""
    return ModelResult(label="LAOnePass", params={}, labels_hat=la_one_pass_labels(response_mat))


def run_la_two_pass(response_mat: npt.NDArray) -> ModelResult:
    """Aggregate task labels with the LA two-pass baseline."""
    return ModelResult(label="LATwoPass", params={}, labels_hat=la_two_pass_labels(response_mat))


def run_mv(response_mat: npt.NDArray) -> ModelResult:
    """Aggregate task labels with vanilla Majority Voting (`subcad.MajorityVoting`)."""
    return ModelResult(label="MV", params={}, labels_hat=subcad.MajorityVoting().fit_predict(response_mat))


def run_ds(response_mat: npt.NDArray) -> ModelResult:
    """Aggregate task labels with vanilla Dawid-Skene (`subcad.DawidSkene`)."""
    return ModelResult(label="DS", params={}, labels_hat=subcad.DawidSkene().fit_predict(response_mat))


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------

# Each method is a function `response_mat -> ModelResult`. RepAlg cannot
# estimate the number of adversaries and also needs `adv_frac=` (the true
# fraction of adversaries, an oracle); SubCAD estimates it with the selector.
# Method names: subcad-<w|b>(eighted/binary graph)<g>(reedy peeling)
# <d|s>(ensity/spectral size selector)<mv|ds>(aggregator).
METHODS = {
    # adversary-agnostic baselines
    "mv": run_mv,
    "ds": run_ds,
    "glad": run_glad,
    "ebcc": run_ebcc,
    "la-onepass": run_la_one_pass,
    "la-twopass": run_la_two_pass,
    # adversarial crowdsourcing methods
    "dacs": partial(run_dacs, rhos=[100]),
    "mmsr": run_mmsr,
    "repalg-soft": partial(run_repalg, detector="soft"),
    "repalg-hard": partial(run_repalg, detector="hard"),
    # SubCAD
    "subcad-wgdmv": partial(run_subcad, kind="weighted", detector="greedy", selector="density", aggregator="mv"),
    "subcad-wgdds": partial(run_subcad, kind="weighted", detector="greedy", selector="density", aggregator="ds"),
    "subcad-wgsmv": partial(run_subcad, kind="weighted", detector="greedy", selector="spectral", aggregator="mv"),
    "subcad-wgsds": partial(run_subcad, kind="weighted", detector="greedy", selector="spectral", aggregator="ds"),
    "subcad-bgsmv": partial(run_subcad, kind="binary", detector="greedy", selector="spectral", aggregator="mv"),
    "subcad-bgsds": partial(run_subcad, kind="binary", detector="greedy", selector="spectral", aggregator="ds"),
}
