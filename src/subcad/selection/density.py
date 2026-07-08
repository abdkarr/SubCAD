import numpy.typing as npt

import numpy as np


class DensitySelector:
    """Size selector based on the densest point along a peeling trajectory.

    A size selector estimates the number of adversarial workers and
    targeted tasks from adversary scores, independently of how those
    scores were produced. It is deliberately decoupled from
    `subcad.detection`'s detectors: `select` takes a bipartite graph and a
    pair of pre-ranked score arrays (ascending suspicion, i.e. the same
    convention as a detector's `worker_scores_`/`task_scores_`) and returns
    a size estimate for each side, so any detector's scores -- or an
    ensembled combination of several -- can be paired with this selector.

    This is the "vanilla" approach originally used by
    `GreedyDetector`/`GreedyPPDetector` (Fraudar-style): treat
    `worker_scores`/`task_scores` as a peeling order (ascending suspicion,
    ties broken by index), replay removing nodes in that order one at a
    time -- always peeling whichever side's next candidate currently has
    the smaller remaining (weighted) degree, same tie-breaking as the
    original peeling loop -- and track the size of the remaining
    worker/task sets at the point of maximum density (total remaining
    edge weight / number of remaining nodes).

    Since it only needs a fixed order to replay (not to discover), it
    requires no search structure (unlike the `MinTree`-based discovery in
    `GreedyDetector._peel`) and works on **any** detector's scores, not
    just a peeling detector's own -- e.g. it can be paired with
    `SpectralDetector` too, or with an ensembled score array.

    !!! Example
        ```python
        from subcad import GreedyDetector
        from subcad.selection import DensitySelector

        detector = GreedyDetector(kind="weighted").fit(response_mat)
        n_adversaries, n_targets = DensitySelector().select(
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
        n_workers, n_tasks = biadj_mat.shape

        # Ascending suspicion -- lossless recovery of a peeling order from
        # scores, since detector scores are a strictly monotonic rank
        # encoding of their own peeling order (no ties).
        workers_order = np.argsort(worker_scores)
        tasks_order = np.argsort(task_scores)

        reordered = biadj_mat[workers_order][:, tasks_order].astype(float)
        worker_remaining_degree = reordered.sum(axis=1)
        task_remaining_degree = reordered.sum(axis=0)

        total_weight = float(reordered.sum())
        best_density = total_weight / (n_workers + n_tasks)
        best_n_workers = n_workers
        best_n_tasks = n_tasks

        # pa/pb: index of the next not-yet-peeled worker/task in the fixed
        # order above -- nothing left to discover, only to replay.
        pa, pb = 0, 0
        while pa < n_workers and pb < n_tasks:
            min_worker_degree = worker_remaining_degree[pa]
            min_task_degree = task_remaining_degree[pb]

            if min_worker_degree <= min_task_degree:
                task_remaining_degree[pb:] -= reordered[pa, pb:]
                total_weight -= min_worker_degree
                pa += 1
            else:
                worker_remaining_degree[pa:] -= reordered[pa:, pb]
                total_weight -= min_task_degree
                pb += 1

            n_workers_remaining = n_workers - pa
            n_tasks_remaining = n_tasks - pb
            n_nodes = n_workers_remaining + n_tasks_remaining
            if n_nodes > 0:
                curr_density = total_weight / n_nodes
                if curr_density > best_density:
                    best_density = curr_density
                    best_n_workers = n_workers_remaining
                    best_n_tasks = n_tasks_remaining

        return int(best_n_workers), int(best_n_tasks)
