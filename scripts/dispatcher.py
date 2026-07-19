"""Three-layer experiment infrastructure for subcad experiments.

Layers
------
ModelResult   — result container produced by runners
evaluate()    — computes metrics from a ModelResult + ground truth
REGISTRY      — maps method names to runner callables

Runner signature: (response_mat, cfg) -> Iterator[ModelResult]

cfg is the [method.*] dict from the TOML. Runners resolve their own search
arrays from it via _resolve_array; if a key is absent the module default is used.

  iteration / iterations / iteration_logspace / iteration_linspace  → Greedy++ sweep
  scale / scales / scale_logspace / scale_linspace                      → aggregator scale sweep
  adv_frac / adv_fracs / adv_frac_logspace / adv_frac_linspace          → aggregator worker-cutoff sweep
  target_frac / target_fracs / target_frac_logspace / target_frac_linspace → aggregator task-cutoff sweep

Usage in an experiment script
------------------------------
    rows = [evaluate(r, gt_adversaries, gt_targeted, gt_labels)
            for r in REGISTRY[method](response_mat, method_cfg)]
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import numpy.typing as npt
import pandas as pd
from crowdkit.aggregation import GLAD
from sklearn import metrics as sk_metrics

import subcad
from subcad.thirdparty import ebcc_labels, la_one_pass_labels, la_two_pass_labels

MSCRIPTS_DIR = Path(subcad.PROJECT_DIR, "research", "mscripts")


def _resolve_array(cfg: dict, key: str) -> npt.NDArray:
    if f"{key}_logspace" in cfg:
        s, e, n = cfg[f"{key}_logspace"]
        return np.logspace(s, e, int(n))
    if f"{key}_linspace" in cfg:
        s, e, n = cfg[f"{key}_linspace"]
        return np.linspace(s, e, int(n))
    if f"{key}s" in cfg:
        return np.array(cfg[f"{key}s"])
    if key in cfg:
        return np.array([cfg[key]])
    raise ValueError(
        f"No search config found for '{key}' in method cfg. "
        f"Add '{key}', '{key}s', '{key}_logspace', or '{key}_linspace'."
    )


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


def run_subcad(response_mat: npt.NDArray, cfg: dict = {}) -> Iterator[ModelResult]:
    n_workers, n_tasks = response_mat.shape
    kind = cfg["kind"]

    if cfg["detector"] == "greedypp":
        iterations_values = _resolve_array(cfg, "iteration")
    else:
        iterations_values = [None]

    for iterations in iterations_values:
        if cfg["detector"] == "greedy":
            detector = subcad.GreedyDetector(kind=kind)
            detector_label = "Greedy"
            detector_params = {}
        elif cfg["detector"] == "greedypp":
            detector = subcad.GreedyPPDetector(kind=kind, iterations=int(iterations))
            detector_label = "GreedyPP"
            detector_params = {"iterations": int(iterations)}
        elif cfg["detector"] == "spectral":
            detector = subcad.SpectralDetector(kind=kind)
            detector_label = "Spectral"
            detector_params = {}
        else:
            raise ValueError(f"Unknown detector '{cfg['detector']}' in method cfg.")

        worker_scores, task_scores = detector.fit_predict(response_mat)

        if cfg["selector"] == "density":
            selector = subcad.DensitySelector()
            selector_label = "Density"
        elif cfg["selector"] == "spectral":
            selector = subcad.SpectralSeededSelector()
            selector_label = "SpectralSeeded"
        else:
            raise ValueError(f"Unknown selector '{cfg['selector']}' in method cfg.")

        n_adversaries, n_targeted = selector.select(
            detector.biadj_mat_, worker_scores, task_scores
        )

        try:
            adv_fracs = _resolve_array(cfg, "adv_frac")
        except ValueError:
            adv_fracs = np.array([n_adversaries / n_workers])
        try:
            target_fracs = _resolve_array(cfg, "target_frac")
        except ValueError:
            target_fracs = np.array([n_targeted / n_tasks])
        try:
            scales = _resolve_array(cfg, "scale")
        except ValueError:
            scales = np.array([5.0])

        for adv_frac in adv_fracs:
            for target_frac in target_fracs:
                for scale in scales:
                    worker_penalties, task_penalties = detector.aggregation_penalties(
                        float(adv_frac), float(target_frac), float(scale)
                    )

                    if cfg["aggregator"] == "mv":
                        aggregator = subcad.WeightedMajorityVoting()
                        aggregator_label = "MV"
                    elif cfg["aggregator"] == "ds":
                        aggregator = subcad.WeightedDawidSkene()
                        aggregator_label = "DS"
                    else:
                        raise ValueError(
                            f"Unknown aggregator '{cfg['aggregator']}' in method cfg."
                        )

                    labels = aggregator.fit_predict(
                        response_mat, worker_penalties, task_penalties
                    )

                    yield ModelResult(
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


def run_dacs(response_mat: npt.NDArray, cfg: dict = {}) -> Iterator[ModelResult]:
    """Detect adversaries with the MATLAB DACS implementation, then aggregate.

    Starts a MATLAB engine and calls `research/mscripts/dacs/dacs.m` to get a
    hard honest/adversary worker partition. That partition is used directly as
    `worker_penalties` for `subcad.WeightedDawidSkene`, which zeroes out the
    contribution of workers DACS flags as adversarial.
    """

    n_workers, _ = response_mat.shape
    n_classes = len(np.unique(response_mat[response_mat > 0]))
    try:
        rhos = _resolve_array(cfg, "rho")
    except ValueError:
        rhos = np.array([1.1, 2, 5, 10, 20, 50, 80, 100, 500, 800, 1000])

    # Detection using MATLAB
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
    worker_penalties = groups

    # Aggregation
    aggregator = subcad.WeightedDawidSkene()
    labels = aggregator.fit_predict(response_mat, worker_penalties, None)

    yield ModelResult(
        label=f"DACS",
        params={"rhos": rhos.tolist()},
        worker_scores=worker_penalties,
        labels_hat=labels,
    )


def run_mmsr(response_mat: npt.NDArray, cfg: dict = {}) -> Iterator[ModelResult]:
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

    yield ModelResult(
        label="MMSR",
        params={},
        worker_scores=worker_scores,
        labels_hat=labels,
    )


def run_repalg(response_mat: npt.NDArray, cfg: dict = {}) -> Iterator[ModelResult]:
    """Detect adversaries with the MATLAB RepAlg implementation, then aggregate.

    Starts a MATLAB engine and calls `research/mscripts/repalg.m`, which returns 
    a continuous per-worker score, representing workers' penalty. The penalties 
    are used as `worker_penalties` in `subcad.WeightedDawidSkene` to find labels.
    """
    class_ids = np.unique(response_mat[response_mat > 0])
    if len(class_ids) != 2:
        raise ValueError(
            f"RepAlg only supports binary classification, got {len(class_ids)} classes."
        )

    matlab, eng = _start_matlab_engine("RepAlg")
    try:
        f = matlab.double(response_mat.tolist())
        scores = eng.repalg(f, nargout=1)
        scores = np.array(scores, dtype=float).flatten()
    finally:
        eng.quit()

    worker_penalties = np.clip(scores, 0, 1)

    aggregator = subcad.WeightedDawidSkene()
    labels = aggregator.fit_predict(response_mat, worker_penalties, None)

    yield ModelResult(
        label="RepAlg",
        params={},
        worker_scores=worker_penalties,
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


def run_glad(response_mat: npt.NDArray, cfg: dict = {}) -> Iterator[ModelResult]:
    """Aggregate task labels with GLAD (`crowdkit.aggregation.GLAD`).

    No adversary detection step; only produces `labels_hat`.
    """
    n_tasks = response_mat.shape[1]
    df = _response_mat_to_crowdkit_df(response_mat)
    labels = GLAD().fit_predict(df).reindex(range(n_tasks)).to_numpy()

    yield ModelResult(label="GLAD", params={}, labels_hat=labels)


def run_ebcc(response_mat: npt.NDArray, cfg: dict = {}) -> Iterator[ModelResult]:
    """Aggregate task labels with EBCC (`subcad.thirdparty.ebcc_labels`).

    No adversary detection step; only produces `labels_hat`.
    """
    labels = ebcc_labels(response_mat)

    yield ModelResult(
        label="EBCC", params={}, labels_hat=labels
    )


def run_la_one_pass(response_mat: npt.NDArray, cfg: dict = {}) -> Iterator[ModelResult]:
    """Aggregate task labels with the LA one-pass baseline.

    No adversary detection step; only produces `labels_hat`.
    """
    labels = la_one_pass_labels(response_mat)

    yield ModelResult(label="LAOnePass", params={}, labels_hat=labels)


def run_la_two_pass(response_mat: npt.NDArray, cfg: dict = {}) -> Iterator[ModelResult]:
    """Aggregate task labels with the LA two-pass baseline.

    No adversary detection step; only produces `labels_hat`.
    """
    labels = la_two_pass_labels(response_mat)

    yield ModelResult(label="LATwoPass", params={}, labels_hat=labels)


def run_mv(response_mat: npt.NDArray, cfg: dict = {}) -> Iterator[ModelResult]:
    """Aggregate task labels with vanilla Majority Voting (`subcad.MajorityVoting`).

    No adversary detection step; only produces `labels_hat`.
    """
    labels = subcad.MajorityVoting().fit_predict(response_mat)

    yield ModelResult(label="MV", params={}, labels_hat=labels)


def run_ds(response_mat: npt.NDArray, cfg: dict = {}) -> Iterator[ModelResult]:
    """Aggregate task labels with vanilla Dawid-Skene (`subcad.DawidSkene`).

    No adversary detection step; only produces `labels_hat`.
    """
    labels = subcad.DawidSkene().fit_predict(response_mat)

    yield ModelResult(label="DS", params={}, labels_hat=labels)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

REGISTRY: dict[str, callable] = {
    "subcad": run_subcad,
    "dacs": run_dacs,
    "mmsr": run_mmsr,
    "repalg": run_repalg,
    "glad": run_glad,
    "ebcc": run_ebcc,
    "la-onepass": run_la_one_pass,
    "la-twopass": run_la_two_pass,
    "mv": run_mv,
    "ds": run_ds,
}

LOADER = {
    "rte": lambda x: subcad.data.fetch_rte(x, return_X_y=True),
    "sp": lambda x: subcad.data.fetch_sp(x, return_X_y=True),
    "dog": lambda x: subcad.data.fetch_dog(x, return_X_y=True),
    "web": lambda x: subcad.data.fetch_web(x, return_X_y=True),
    "adult2": lambda x: subcad.data.fetch_adult2(x, return_X_y=True),
    "temp": lambda x: subcad.data.fetch_temp(x, return_X_y=True),
}
