import numpy.typing as npt

import numpy as np

from numba import njit
from sklearn.base import BaseEstimator

from ..calc_agreement_mat import calc_agreement_mat


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


class BaseDetector(BaseEstimator):
    """Base class for dense-subgraph adversary detectors.

    Not meant to be instantiated directly. Use `GreedyDetector` or
    `GreedyPPDetector`, which differ only in the peeling algorithm used
    (`_peel`).

    The detection is performed by first constructing a bipartite graph
    $G=(W, T, E)$ from the response matrix of a crowdsourced dataset. In $G$,
    $W$ and $T$ are the nodes representing workers and tasks and edges
    connect a worker and a task if that worker provided a label for that
    task. An edge weighting mechanism that employs worker agreement rates
    and co-labeling is also provided (`kind="weighted"`). The constructed
    bipartite graph is then processed using a peeling algorithm whose
    order is used to calculate adversary scores such that higher scores
    indicate higher likelihood for a worker to be adversarial or a task to
    be targeted. Further details can be found in [1].

    Detectors only rank workers/tasks -- they do not estimate how many are
    actually adversarial/targeted. For that, pair a fitted detector with a
    selector from `subcad.selection` (e.g. `DensitySelector` or
    `SpectralSeededSelector`), passing `biadj_mat_`/`worker_scores_`/
    `task_scores_`.

    Parameters
    ----------
    kind
        Kind of bipartite graph to construct. Must be either "binary" or
        "weighted". In the latter case, the edges of the bipartite graph
        are weighted as described in [1].

    Attributes
    ----------
    biadj_mat_ : npt.NDArray
        $(M, N)$ dimensional bi-adjacency matrix of the worker-task
        bipartite graph constructed from the response matrix. Set after
        calling `fit`.
    worker_scores_ : npt.NDArray
        $(M, )$ dimensional array where `worker_scores_[i]` is the
        adversary score of $i$th worker indicating the likelihood of $i$
        being an adversary. Set after calling `fit`.
    task_scores_ : npt.NDArray
        $(N, )$ dimensional array where `task_scores_[i]` is the adversary
        score of $i$th task indicating the likelihood of $i$ being a
        targeted task. Set after calling `fit`.

    References
    ----------
    [1] Karaaslanli, Abdullah, Panagiotis A. Traganitis, and Aritra Konar.
    "Identifying Adversarial Attacks in Crowdsourcing via Dense Subgraph
    Detection." ICASSP 2025-2025 IEEE International Conference on
    Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2025.
    """

    def __init__(self, kind: str = "binary"):
        self.kind = kind

    def _peel(self, biadj_mat: npt.NDArray):
        raise NotImplementedError

    def fit(self, response_mat: npt.NDArray, y=None) -> "BaseDetector":
        """Detect adversarial workers and their targeted tasks.

        Parameters
        ----------
        response_mat
            $(M, N)$ dimensional matrix where `response_mat[i, j]` is the
            label provided by $i$th worker for $j$th task.
            `response_mat[i, j] = 0` is assumed to indicate no label is
            given by $i$th worker for $j$th task.
        y
            Ignored. Present for scikit-learn API consistency.

        Returns
        -------
        self
        """
        biadj_mat = _construct_biadj_mat(response_mat, self.kind)
        workers_order, tasks_order = self._peel(biadj_mat)
        worker_scores, task_scores = _calc_adversary_scores(workers_order, tasks_order)

        self.biadj_mat_ = biadj_mat
        self.worker_scores_ = worker_scores
        self.task_scores_ = task_scores

        return self

    def fit_predict(
        self, response_mat: npt.NDArray, y=None
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Fit the detector and return worker/task adversary scores.

        Equivalent to calling `fit` followed by reading `worker_scores_`
        and `task_scores_`.

        Parameters
        ----------
        response_mat
            See `fit`.
        y
            Ignored. Present for scikit-learn API consistency.

        Returns
        -------
        worker_scores : npt.NDArray
            See `worker_scores_`.
        task_scores : npt.NDArray
            See `task_scores_`.
        """
        self.fit(response_mat)
        return self.worker_scores_, self.task_scores_
