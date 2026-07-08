import numpy.typing as npt

from sklearn.base import BaseEstimator


class BaseSizeSelector(BaseEstimator):
    """Base class for adversary/targeted-task size selectors.

    Not meant to be instantiated directly. Use `DensitySelector` or
    `SpectralSeededSelector`.

    A size selector estimates the number of adversarial workers and
    targeted tasks from adversary scores, independently of how those
    scores were produced. It is deliberately decoupled from
    `subcad.detection`'s detectors: `select` takes a bipartite graph and a
    pair of pre-ranked score arrays (ascending suspicion, i.e. the same
    convention as `BaseDetector.worker_scores_`/`task_scores_`) and returns
    a size estimate for each side, so any detector's scores -- or an
    ensembled combination of several -- can be paired with any selector.

    !!! Example
        ```python
        from subcad import GreedyDetector
        from subcad.selection import DensitySelector

        detector = GreedyDetector(kind="weighted").fit(response_mat)
        selector = DensitySelector()
        n_adversaries, n_targets = selector.select(
            detector.biadj_mat_, detector.worker_scores_, detector.task_scores_
        )
        ```
    """

    def select(
        self,
        biadj_mat: npt.NDArray,
        worker_scores: npt.NDArray,
        task_scores: npt.NDArray,
    ) -> tuple[int, int]:
        """Estimate the number of adversarial workers and targeted tasks.

        Parameters
        ----------
        biadj_mat
            $(M, N)$ dimensional bi-adjacency matrix of the worker-task
            bipartite graph that `worker_scores`/`task_scores` were derived
            from, e.g. a fitted detector's `biadj_mat_` attribute.
        worker_scores
            $(M, )$ dimensional array of per-worker adversary scores,
            higher indicating higher likelihood of being adversarial, e.g.
            a fitted detector's `worker_scores_` attribute.
        task_scores
            $(N, )$ dimensional array of per-task adversary scores, higher
            indicating higher likelihood of being targeted, e.g. a fitted
            detector's `task_scores_` attribute.

        Returns
        -------
        n_adversaries : int
            Estimated number of adversarial workers.
        n_targets : int
            Estimated number of targeted tasks.
        """
        raise NotImplementedError
