import numpy.typing as npt

import numpy as np

from numba import njit
from scipy.special import expit

from ..agreement_mat import calc_agreement_mat


@njit
def _calc_edge_weights(
    response_mat: npt.NDArray,
    workers: npt.NDArray,
    tasks: npt.NDArray,
    responses: npt.NDArray,
) -> npt.NDArray:
    """
    Calculate edge weights of the bipartite graph representation of a crowdsourced
    dataset based on agreement rates and co-labeling.
    """

    agreement_mat, _ = calc_agreement_mat(response_mat)

    n_responses = len(responses)
    weights = np.zeros(n_responses)
    for i in range(n_responses):
        worker = workers[i]
        task = tasks[i]
        label = responses[i]

        # Find workers who labeled the current task the same as current worker
        task_labels = response_mat[:, task]
        matching_workers = np.where(task_labels == label)[0]
        matching_workers = np.setdiff1d(matching_workers, worker, assume_unique=True)

        # Calculate edge weight
        if len(matching_workers) > 0:
            weights[i] = np.mean(agreement_mat[worker, matching_workers])

    return weights


def _construct_biadj_mat(
    response_mat: npt.NDArray, kind: str = "binary"
) -> npt.NDArray:
    """
    Construct the bi-adjacency matrix of the bipartite graph representation
    of a crowdsourced dataset.
    """

    n_workers, n_tasks = response_mat.shape

    workers, tasks = np.nonzero(response_mat)
    responses = response_mat[workers, tasks]

    if kind == "binary":
        weights = 1
    elif kind == "weighted":
        weights = _calc_edge_weights(response_mat, workers, tasks, responses)

    biadj_mat = np.zeros((n_workers, n_tasks))
    biadj_mat[workers, tasks] = weights

    return biadj_mat


def _calc_adversary_scores(workers_order, tasks_order):
    """
    Calculate adversary scores of workers and tasks based on their peeling order
    """
    n_workers = len(workers_order)
    n_tasks = len(tasks_order)

    worker_scores = np.zeros(n_workers)
    task_scores = np.zeros(n_tasks)

    for i, w in enumerate(workers_order):
        worker_scores[w] = i + 1
    for i, t in enumerate(tasks_order):
        task_scores[t] = i + 1

    worker_scores = worker_scores / n_workers
    task_scores = task_scores / n_tasks

    return worker_scores, task_scores


def _sigmoid(x, frac=0.2, scale=5):
    return expit(-(x - frac) * scale * 10)


def _calc_adversary_penalties(
    worker_scores: npt.NDArray,
    task_scores: npt.NDArray,
    adv_frac: float = 0.2,
    target_frac: float = 0.2,
    scale: float = 5,
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Convert adversary scores into adversary-aware aggregation penalties via
    a sigmoid cutoff, so that workers/tasks ranked above the expected
    adversarial fraction get a penalty close to 1 (i.e. not trustworthy --
    see each detector's `aggregation_penalties` method, whose shared
    implementation this is, for the combination convention).
    """
    worker_penalties = _sigmoid(1 - worker_scores, frac=adv_frac, scale=scale)
    task_penalties = _sigmoid(1 - task_scores, frac=target_frac, scale=scale)

    return worker_penalties, task_penalties
