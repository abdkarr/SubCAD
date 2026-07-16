"""LA one-pass / two-pass label-aggregation baselines.

Port of `research/scripts/la.py`, a baseline used for comparison against
`subcad.aggregators` rather than a `subcad` design of our own. Each worker's
ability is a Beta(`alpha`, `beta`)-distributed correctness rate, updated
online as tasks are processed (one-pass), then optionally replayed with the
final ability estimates held fixed (two-pass).
"""

import numpy as np
import numpy.typing as npt

from ..typing import RNGType
from ..utils import check_rng


def _one_pass(
    e2wl: dict,
    w2el: dict,
    label_set: list,
    rng: np.random.Generator,
    alpha: float = 2,
    beta: float = 2,
) -> tuple[dict, dict]:
    items = list(e2wl.keys())
    c: dict = {}
    t: dict = {}
    a: dict = {}
    truths: dict = {}
    for worker in w2el.keys():
        c[worker] = alpha - 1
        t[worker] = alpha + beta - 2
        a[worker] = c[worker] / t[worker]

    rng.shuffle(items)

    for item in items:
        votes: dict = {}
        for worker, worker_label in e2wl[item]:
            votes[worker_label] = votes.get(worker_label, 0) + a[worker]
        candidate = []
        max_ = -1
        for class_ in label_set:
            if votes.get(class_) is None:
                continue
            if votes.get(class_) > max_:
                candidate = [class_]
                max_ = votes.get(class_)
            elif votes.get(class_) == max_:
                candidate.append(class_)
        truths[item] = rng.choice(candidate)

        for worker, worker_label in e2wl[item]:
            t[worker] = t[worker] + 1
            if worker_label == truths[item]:
                c[worker] = c[worker] + 1
            a[worker] = c[worker] / t[worker]

    return truths, a


def _two_pass(e2wl: dict, a: dict, label_set: list, rng: np.random.Generator) -> dict:
    K = len(label_set)
    truths: dict = {}
    for item, votes_wl in e2wl.items():
        votes: dict = {}
        for worker, worker_label in votes_wl:
            votes[worker_label] = votes.get(worker_label, 0) + (a[worker] * K - 1)
        candidate = []
        max_ = -1
        for class_ in label_set:
            if votes.get(class_) is None:
                continue
            if votes.get(class_) > max_:
                candidate = [class_]
                max_ = votes.get(class_)
            elif votes.get(class_) == max_:
                candidate.append(class_)
        truths[item] = rng.choice(candidate)

    return truths


def _response_mat_to_el(response_mat: npt.NDArray) -> tuple[dict, dict, list]:
    n_workers, n_tasks = response_mat.shape
    worker_idx, task_idx = np.nonzero(response_mat)

    e2wl: dict = {}
    w2el: dict = {w: [] for w in range(n_workers)}
    for worker, task in zip(worker_idx.tolist(), task_idx.tolist()):
        label = int(response_mat[worker, task])
        e2wl.setdefault(task, []).append([worker, label])
        w2el[worker].append([task, label])
    label_set = np.unique(response_mat[response_mat > 0]).astype(int).tolist()

    return e2wl, w2el, label_set


def la_one_pass_labels(
    response_mat: npt.NDArray,
    alpha: float = 2,
    beta: float = 2,
    random_state: RNGType = None,
) -> npt.NDArray:
    """Aggregate task labels with the LA one-pass baseline.

    Parameters
    ----------
    response_mat
        `(n_workers, n_tasks)` response matrix; see `subcad`'s response
        matrix convention (0 = unobserved, positive ints = class labels).
    alpha, beta
        Shape parameters of each worker's Beta ability prior.
    random_state
        See `subcad.typing.RNGType`. Controls task-processing order and
        tie-breaking among equally-voted classes.

    Returns
    -------
    labels : ndarray (n_tasks,)
        Aggregated task labels for tasks with at least one response; tasks
        with no responses are left unset and will raise `KeyError`.
    """
    response_mat = np.asarray(response_mat)
    rng = check_rng(random_state)

    e2wl, w2el, label_set = _response_mat_to_el(response_mat)
    truths, _ = _one_pass(e2wl, w2el, label_set, rng, alpha=alpha, beta=beta)

    return np.array([truths[t] for t in range(response_mat.shape[1])])


def la_two_pass_labels(
    response_mat: npt.NDArray,
    alpha: float = 2,
    beta: float = 2,
    random_state: RNGType = None,
) -> npt.NDArray:
    """Aggregate task labels with the LA two-pass baseline.

    Runs `la_one_pass_labels`'s one-pass fit to estimate worker abilities,
    then replays all tasks a second time with those abilities held fixed.

    Parameters
    ----------
    response_mat, alpha, beta, random_state
        See `la_one_pass_labels`.

    Returns
    -------
    labels : ndarray (n_tasks,)
        Aggregated task labels for tasks with at least one response; tasks
        with no responses are left unset and will raise `KeyError`.
    """
    response_mat = np.asarray(response_mat)
    rng = check_rng(random_state)

    e2wl, w2el, label_set = _response_mat_to_el(response_mat)
    _, a = _one_pass(e2wl, w2el, label_set, rng, alpha=alpha, beta=beta)
    truths = _two_pass(e2wl, a, label_set, rng)

    return np.array([truths[t] for t in range(response_mat.shape[1])])
