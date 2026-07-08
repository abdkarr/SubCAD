import numpy.typing as npt

import numpy as np


def _otsu_cut(scores_desc: npt.NDArray, min_k: int = 1) -> int:
    """
    Cut a descending-sorted score array via Otsu's method: the threshold
    that maximizes between-group variance of a 2-way split. Uses the whole
    distribution's shape rather than a single local difference, so it is
    far less sensitive to one noisy neighbor-to-neighbor jump than a
    largest-gap cut.
    """
    n = len(scores_desc)
    if n <= min_k:
        return n
    v = scores_desc[::-1]  # ascending
    total_sum = v.sum()
    cumsum = np.cumsum(v)
    idx = np.arange(1, n)
    n_left, n_right = idx, n - idx
    sum_left = cumsum[:-1]
    sum_right = total_sum - sum_left
    mean_left = sum_left / n_left
    mean_right = sum_right / n_right
    between = n_left * n_right * (mean_left - mean_right) ** 2
    best_i = int(np.argmax(between))
    k = n - idx[best_i]
    return max(k, min_k)


class SpectralSeededSelector:
    """Two-stage seeded spectral size selector (Phase 4 of the "Adversary
    Size Estimation Investigation").

    A size selector estimates the number of adversarial workers and
    targeted tasks from adversary scores, independently of how those
    scores were produced. It is deliberately decoupled from
    `subcad.detection`'s detectors: `select` takes a bipartite graph and a
    pair of pre-ranked score arrays (ascending suspicion, i.e. the same
    convention as a detector's `worker_scores_`/`task_scores_`) and returns
    a size estimate for each side, so any detector's scores -- or an
    ensembled combination of several -- can be paired with this selector.

    Unlike `DensitySelector`, this method looks past
    `worker_scores`/`task_scores` at `biadj_mat` itself: it pools
    candidate workers/tasks by rank (top
    `worker_pool_frac`/`task_pool_frac` fraction each), takes the top left
    singular vector of the pooled biadjacency submatrix and cuts it via
    Otsu's method for a worker-side estimate, then uses that worker
    estimate as a *seed* -- rather than the full (still mostly honest)
    pool -- to compute the top right singular vector of
    (seed workers x task pool), again cut via Otsu, for the task-side
    estimate. Seeding the task-side stage with a tight worker estimate
    keeps the honest majority from dominating the SVD with its own
    consensus pattern and drowning out the adversary-specific signal.

    Validated across four synthetic configs (balanced/skewed task:worker
    ratio x easy/hard camouflage difficulty; see the investigation note):
    mean size estimate 0.96x-1.12x of the true count on 3 of 4 configs
    (worker and task side both), including the axis ("skewed/hard"
    task-side) that every density-based and label-content method tried
    earlier could not predict well (0.99x +/- 0.03 here).

    Parameters
    ----------
    worker_pool_frac
        Fraction of workers (by score, highest first) to pool for the
        worker-side singular vector.
    task_pool_frac
        Fraction of tasks (by score, highest first) to pool for both
        stages' task columns.

    !!! Example
        ```python
        from subcad import SpectralDetector
        from subcad.selection import SpectralSeededSelector

        detector = SpectralDetector(kind="weighted").fit(response_mat)
        n_adversaries, n_targets = SpectralSeededSelector().select(
            detector.biadj_mat_, detector.worker_scores_, detector.task_scores_
        )
        ```
    """

    def __init__(self, worker_pool_frac: float = 0.7, task_pool_frac: float = 0.7):
        self.worker_pool_frac = worker_pool_frac
        self.task_pool_frac = task_pool_frac

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
        n_workers, n_tasks = biadj_mat.shape
        worker_pool_size = max(4, int(self.worker_pool_frac * n_workers))
        task_pool_size = max(4, int(self.task_pool_frac * n_tasks))

        worker_pool = np.argsort(-worker_scores)[:worker_pool_size]
        task_pool = np.argsort(-task_scores)[:task_pool_size]

        # Stage 1: worker-side estimate from the pooled submatrix's top
        # left singular vector.
        sub1 = biadj_mat[np.ix_(worker_pool, task_pool)]
        u, _, _ = np.linalg.svd(sub1, full_matrices=False)
        u1 = np.abs(u[:, 0])
        worker_order_in_pool = np.argsort(-u1)
        n_adversaries = _otsu_cut(u1[worker_order_in_pool])
        worker_seed = worker_pool[worker_order_in_pool[:n_adversaries]]

        # Stage 2: task-side estimate, restricting rows to just the worker
        # seed so the honest majority doesn't dominate the SVD.
        sub2 = biadj_mat[np.ix_(worker_seed, task_pool)]
        _, _, vt2 = np.linalg.svd(sub2, full_matrices=False)
        v1 = np.abs(vt2[0, :])
        task_order_in_pool = np.argsort(-v1)
        n_targets = _otsu_cut(v1[task_order_in_pool])

        return int(n_adversaries), int(n_targets)
