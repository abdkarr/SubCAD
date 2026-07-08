import numpy.typing as npt

import numpy as np

from .base import _calc_adversary_scores, _construct_biadj_mat


class SpectralDetector:
    """Adversary detector using the leading singular vectors of the
    bipartite graph's biadjacency matrix.

    The detection is performed by first constructing a bipartite graph
    $G=(W, T, E)$ from the response matrix of a crowdsourced dataset. In $G$,
    $W$ and $T$ are the nodes representing workers and tasks and edges
    connect a worker and a task if that worker provided a label for that
    task. An edge weighting mechanism that employs worker agreement rates
    and co-labeling is also provided (`kind="weighted"`). Unlike
    `GreedyDetector`/`GreedyPPDetector`, which rank nodes by Fraudar-style
    peeling order, this detector ranks workers/tasks by the magnitude of
    their loading in the top left/right singular vector of the (optionally
    agreement-weighted) biadjacency matrix: workers/tasks that load
    heavily on the dominant singular direction are more likely to be part
    of a coordinated adversarial/targeted block.

    Ranks nodes about as well as peeling order (comparable AUROC), via an
    entirely different mechanism (spectral rather than combinatorial
    peeling), which makes it a natural candidate for ensembling alongside
    `GreedyDetector`/`GreedyPPDetector` rather than a replacement for them.
    Pair with a selector from `subcad.selection` (e.g.
    `SpectralSeededSelector`) for a size estimate.

    !!! Example
        ```python
        from subcad.detection import SpectralDetector

        detector = SpectralDetector(kind="weighted")
        worker_scores, task_scores = detector.fit_predict(response_mat)
        ```

    Parameters
    ----------
    kind
        Kind of bipartite graph to construct. Must be either "binary" or
        "weighted". In the latter case, the edges of the bipartite graph
        are weighted using worker agreement rates and co-labeling.

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
    """

    def __init__(self, kind: str = "binary"):
        self.kind = kind

    def fit(self, response_mat: npt.NDArray, y=None) -> "SpectralDetector":
        """Detect adversarial workers and their targeted tasks.

        Parameters
        ----------
        response_mat
            $(M, N)$ dimensional matrix where `response_mat[i, j]` is the
            label provided by $i$th worker for $j$th task.
            `response_mat[i, j] = 0` is assumed to indicate no label is
            given by $i$th worker for $j$th task.
        y
            Ignored. Present for API consistency.

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
            Ignored. Present for API consistency.

        Returns
        -------
        worker_scores : npt.NDArray
            See `worker_scores_`.
        task_scores : npt.NDArray
            See `task_scores_`.
        """
        self.fit(response_mat)
        return self.worker_scores_, self.task_scores_

    def _peel(self, biadj_mat: npt.NDArray):
        """
        Rank workers/tasks by their loading magnitude in the top left/right
        singular vector of `biadj_mat`.
        """
        u, _, vt = np.linalg.svd(biadj_mat, full_matrices=False)
        worker_loadings = np.abs(u[:, 0])
        task_loadings = np.abs(vt[0, :])

        # Ascending loading -> matches the peeling detectors' convention
        # that workers_order/tasks_order run from least to most suspicious.
        workers_order = np.argsort(worker_loadings)
        tasks_order = np.argsort(task_loadings)

        return workers_order, tasks_order
