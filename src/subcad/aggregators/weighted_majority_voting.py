import numpy as np

import numpy.typing as npt


class WeightedMajorityVoting:
    """Adversary-aware weighted majority-voting label aggregator.

    Each task's label is estimated as a weighted majority vote, where each
    worker's vote is weighted by `1 - worker_penalties[i] * task_penalties[j]`
    (`worker_penalties`/`task_penalties` are $[0, 1]$-valued, higher
    meaning less trustworthy), so that an untrustworthy worker voting on
    an untrustworthy task contributes little. `worker_penalties`/
    `task_penalties` are computed ahead of time, e.g. via a fitted
    detector's `aggregation_penalties` method (see
    `subcad.detection.GreedyDetector.aggregation_penalties`).

    !!! Example
        ```python
        from subcad import GreedyDetector
        from subcad.aggregators import WeightedMajorityVoting

        detector = GreedyDetector(kind="weighted").fit(response_mat)
        worker_penalties, task_penalties = detector.aggregation_penalties()
        labels_hat = WeightedMajorityVoting().fit_predict(
            response_mat, worker_penalties, task_penalties
        )
        ```

    Attributes
    ----------
    labels_ : npt.NDArray
        $(N, )$ dimensional array where `labels_[i]` is the label of $i$th
        task estimated by weighted majority voting. Set after calling
        `fit`.
    """

    def fit(
        self,
        response_mat: npt.NDArray,
        worker_penalties: npt.NDArray,
        task_penalties: npt.NDArray | None = None,
        y=None,
    ) -> "WeightedMajorityVoting":
        """Estimate task labels via adversary-aware weighted majority voting.

        Parameters
        ----------
        response_mat
            $(M, N)$ dimensional matrix where `response_mat[i, j]` is the
            label provided by $i$th worker for $j$th task.
            `response_mat[i, j] = 0` is assumed to indicate no label is
            given by $i$th worker for $j$th task.
        worker_penalties
            $(M, )$ dimensional, $[0, 1]$-valued array of per-worker
            aggregation penalties (higher meaning less trustworthy), e.g.
            from a fitted detector's `aggregation_penalties` method.
        task_penalties
            $(N, )$ dimensional, $[0, 1]$-valued array of per-task
            aggregation penalties (higher meaning less trustworthy), e.g.
            from a fitted detector's `aggregation_penalties` method. If
            `None`, tasks do not affect the weighting -- votes are
            weighted only by `worker_penalties`.
        y
            Ignored. Present for API consistency.

        Returns
        -------
        self
        """
        n_tasks = response_mat.shape[1]

        class_ids = np.unique(response_mat)
        if 0 in class_ids:
            class_ids = class_ids[class_ids != 0]

        labels_hat = np.zeros(n_tasks)
        for t in range(n_tasks):
            t_workers = np.where(response_mat[:, t])[0]

            if task_penalties is None:
                weights = 1 - worker_penalties[t_workers]
            else:
                weights = 1 - worker_penalties[t_workers] * task_penalties[t]
            t_labels = response_mat[t_workers, t]

            max_weight = -np.inf
            for c in class_ids:
                c_weight = np.sum(weights[t_labels == c])
                if c_weight > max_weight:
                    max_weight = c_weight
                    best_c = c

            labels_hat[t] = best_c

        self.labels_ = labels_hat

        return self

    def fit_predict(
        self,
        response_mat: npt.NDArray,
        worker_penalties: npt.NDArray,
        task_penalties: npt.NDArray | None = None,
        y=None,
    ) -> npt.NDArray:
        """Fit the aggregator and return the estimated task labels.

        Equivalent to calling `fit` followed by reading `labels_`.

        Parameters
        ----------
        response_mat, worker_penalties, task_penalties
            See `fit`.
        y
            Ignored. Present for API consistency.

        Returns
        -------
        labels : npt.NDArray
            See `labels_`.
        """
        self.fit(response_mat, worker_penalties, task_penalties)
        return self.labels_
