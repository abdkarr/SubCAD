import numpy.typing as npt

import numpy as np


def _otsu_cut(scores_desc: npt.NDArray, min_k: int = 1) -> int:
    """
    Cut a descending-sorted score array via Otsu's method: the threshold
    that maximizes between-group variance of a 2-way split.
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
    """Two-stage seeded spectral size selector.

    This method pools candidate workers/tasks by rank, takes the top left singular 
    vector of the pooled biadjacency submatrix and cuts it via Otsu's method for 
    a worker-side estimate, then uses that worker estimate as a *seed* to compute 
    the top right singular vector of (seed workers x task pool). Right singular 
    vector is cut via Otsu  for the task-side estimate. Seeding the task-side 
    stage with a tight worker estimate keeps the honest majority from dominating 
    the SVD with its own consensus pattern and drowning out the adversary-specific 
    signal.

    The worker-side stage degree-normalizes the pooled submatrix's rows ($D^{-1/2}$) 
    before the SVD. Without it, a handful of highly prolific honest workers can 
    dominate the top singular vector by raw response volume alone and bury the 
    (smaller-magnitude but structurally coherent) adversary block. The task-side 
    stage is deliberately left unnormalized: once rows are restricted to the 
    worker seed, raw column magnitude (how much attack weight landed on a task) 
    is itself the useful signal, and degree-normalizing it away was found to hurt 
    task-count accuracy.

    Parameters
    ----------
    worker_pool_frac
        Fraction of workers (by score, highest first) to pool for the
        worker-side singular vector.
    task_pool_frac
        Fraction of tasks (by score, highest first) to pool for both
        stages' task columns.

    Examples
    --------
    ```python
    from subcad import SpectralDetector
    from subcad.selection import SpectralSeededSelector

    detector = SpectralDetector(kind="weighted").fit(response_mat)
    n_adversaries, n_targets = SpectralSeededSelector().select(
        detector.biadj_mat_, detector.worker_scores_, detector.task_scores_
    )
    ```
    """

    def __init__(self, worker_pool_frac: float = 0.5, task_pool_frac: float = 0.5):
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
        # left singular vector. Rows are D^-1/2 degree-normalized first so
        # a few high-volume honest workers can't dominate the singular
        # vector by raw magnitude and bury the adversary block.
        sub1 = biadj_mat[np.ix_(worker_pool, task_pool)].astype(float)
        row_deg = sub1.sum(axis=1, keepdims=True)
        row_deg[row_deg == 0] = 1.0
        u, _, _ = np.linalg.svd(sub1 / np.sqrt(row_deg), full_matrices=False)
        u1 = np.abs(u[:, 0])
        worker_order_in_pool = np.argsort(-u1)
        n_adversaries = _otsu_cut(u1[worker_order_in_pool])
        worker_seed = worker_pool[worker_order_in_pool[:n_adversaries]]

        # Stage 2: task-side estimate, restricting rows to just the worker
        # seed so the honest majority doesn't dominate the SVD. Left
        # unnormalized -- with only the worker seed as rows, raw column
        # magnitude is itself the useful attack-concentration signal.
        sub2 = biadj_mat[np.ix_(worker_seed, task_pool)]
        _, _, vt2 = np.linalg.svd(sub2, full_matrices=False)
        v1 = np.abs(vt2[0, :])
        task_order_in_pool = np.argsort(-v1)
        n_targets = _otsu_cut(v1[task_order_in_pool])

        return int(n_adversaries), int(n_targets)
