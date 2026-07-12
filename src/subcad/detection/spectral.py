import numpy.typing as npt

import numpy as np

from .base import _calc_adversary_penalties, _calc_adversary_scores, _construct_biadj_mat


class SpectralDetector:
    r"""Adversary detector using the leading singular vectors of the
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

    `normalize=True` degree-normalizes `biadj_mat`'s rows ($A_{ij}/\sqrt{d_i}$,
    $d_i$ = worker $i$'s degree) before the SVD, so a handful of highly
    prolific workers can't dominate the top singular vector by raw response
    volume alone and bury a smaller-magnitude but structurally coherent
    adversary block -- the same hub-domination issue diagnosed for
    `subcad.selection.SpectralSeededSelector` (see the "Adversary Size
    Estimation Investigation" note, Phase 5). Benchmarked against the
    `scripts/planted_attacks.py` setup across all six bundled real datasets:
    improved both `WorkerROCAUC` (0.92->0.95) and `TaskROCAUC` (0.91->0.95)
    on average, with no regression found in any dataset or swept
    `adv_frac`/`target_frac`/`target_obs`/`camo_reliability` config.
    Symmetric normalization ($A_{ij}/\sqrt{d_i d_j}$, also normalizing task
    degree) was tried too and performed worse (worker-side `AP` dropped
    sharply) -- not exposed as an option here. Off by default for backward
    compatibility. Peeling-based detectors (`GreedyDetector`/
    `GreedyPPDetector`) do not have an equivalent flag: the same
    normalization was tested there too and consistently *hurt* ranking
    quality (their raw-degree comparison is the mechanism that already
    works, not a bug to fix).

    !!! Example
        ```python
        from subcad.detection import SpectralDetector

        detector = SpectralDetector(kind="weighted", normalize=True)
        worker_scores, task_scores = detector.fit_predict(response_mat)
        ```

    Parameters
    ----------
    kind
        Kind of bipartite graph to construct. Must be either "binary" or
        "weighted". In the latter case, the edges of the bipartite graph
        are weighted using worker agreement rates and co-labeling.
    normalize
        If `True`, degree-normalize `biadj_mat`'s rows before the SVD (see
        above). Defaults to `False`.

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

    def __init__(self, kind: str = "binary", normalize: bool = False):
        self.kind = kind
        self.normalize = normalize

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

    def aggregation_penalties(
        self, adv_frac: float = 0.2, target_frac: float = 0.2, scale: float = 5
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Compute adversary-aware aggregation penalties from this detector's scores.

        Applies a sigmoid cutoff to `worker_scores_`/`task_scores_`,
        producing $[0, 1]$-valued `worker_penalties`/`task_penalties` that
        approach 1 for workers/tasks ranked above the expected adversarial
        fraction (higher = less trustworthy). An adversary-aware
        aggregator (e.g. `subcad.aggregators.WeightedMajorityVoting`,
        `subcad.aggregators.WeightedDawidSkene`) combines the returned
        `worker_penalties`/`task_penalties` as
        `1 - worker_penalties[i] * task_penalties[j]` to weight worker
        `i`'s response to task `j`. Must be called after `fit`.

        !!! Example
            ```python
            from subcad.detection import SpectralDetector
            from subcad.aggregators import WeightedMajorityVoting

            detector = SpectralDetector(kind="weighted").fit(response_mat)
            worker_penalties, task_penalties = detector.aggregation_penalties()
            labels_hat = WeightedMajorityVoting().fit_predict(
                response_mat, worker_penalties, task_penalties
            )
            ```

        Parameters
        ----------
        adv_frac
            Expected fraction of workers that are adversarial. Sets the
            cutoff of the sigmoid applied to `worker_scores_`: workers
            ranked above the top `adv_frac` fraction are penalized.
        target_frac
            Expected fraction of tasks that are targeted. Sets the cutoff
            of the sigmoid applied to `task_scores_`: tasks ranked above
            the top `target_frac` fraction are penalized.
        scale
            Steepness of the sigmoid transition at the
            `adv_frac`/`target_frac` cutoffs.

        Returns
        -------
        worker_penalties : npt.NDArray
            $(M, )$ dimensional array of per-worker, $[0, 1]$-valued
            aggregation penalties (higher meaning less trustworthy).
        task_penalties : npt.NDArray
            $(N, )$ dimensional array of per-task, $[0, 1]$-valued
            aggregation penalties (higher meaning less trustworthy).
        """
        return _calc_adversary_penalties(
            self.worker_scores_, self.task_scores_, adv_frac, target_frac, scale
        )

    def _peel(self, biadj_mat: npt.NDArray):
        """
        Rank workers/tasks by their loading magnitude in the top left/right
        singular vector of `biadj_mat` (optionally row-degree-normalized
        first, see `normalize`).
        """
        mat = biadj_mat
        if self.normalize:
            row_deg = mat.sum(axis=1, keepdims=True)
            row_deg = np.where(row_deg == 0, 1.0, row_deg)
            mat = mat / np.sqrt(row_deg)

        u, _, vt = np.linalg.svd(mat, full_matrices=False)
        worker_loadings = np.abs(u[:, 0])
        task_loadings = np.abs(vt[0, :])

        # Ascending loading -> matches the peeling detectors' convention
        # that workers_order/tasks_order run from least to most suspicious.
        workers_order = np.argsort(worker_loadings)
        tasks_order = np.argsort(task_loadings)

        return workers_order, tasks_order
