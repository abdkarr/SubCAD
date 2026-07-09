import numpy as np

import numpy.typing as npt

from scipy.special import expit


def _sigmoid(x, frac=0.2, scale=5):
    return expit(-(x - frac) * scale * 10)


class WeightedMajorityVoting:
    """Adversary-aware weighted majority-voting label aggregator.

    Each task's label is estimated as a weighted majority vote, where each
    worker's vote is downweighted by a sigmoid of its adversary score
    (`worker_scores`) and the targeted-task score of the task being voted
    on (`task_scores`), so that suspicious workers voting on suspicious
    tasks contribute little.

    !!! Example
        ```python
        from subcad import GreedyDetector
        from subcad.aggregators import WeightedMajorityVoting

        detector = GreedyDetector(kind="weighted").fit(response_mat)
        labels_hat = WeightedMajorityVoting().fit_predict(
            response_mat, detector.worker_scores_, detector.task_scores_
        )
        ```

    Parameters
    ----------
    adv_frac
        Expected fraction of workers that are adversarial. Sets the
        cutoff of the sigmoid applied to worker scores: workers ranked
        above the top `adv_frac` fraction are downweighted.
    target_frac
        Expected fraction of tasks that are targeted. Sets the cutoff of
        the sigmoid applied to task scores: tasks ranked above the top
        `target_frac` fraction are downweighted.
    scale
        Steepness of the sigmoid transition at the `adv_frac`/`target_frac`
        cutoffs.

    Attributes
    ----------
    labels_ : npt.NDArray
        $(N, )$ dimensional array where `labels_[i]` is the label of $i$th
        task estimated by weighted majority voting. Set after calling
        `fit`.
    """

    def __init__(
        self,
        adv_frac: float = 0.2,
        target_frac: float = 0.2,
        scale: float = 5,
    ):
        self.adv_frac = adv_frac
        self.target_frac = target_frac
        self.scale = scale

    def fit(
        self,
        response_mat: npt.NDArray,
        worker_scores: npt.NDArray,
        task_scores: npt.NDArray,
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
        worker_scores
            $(M, )$ dimensional array of per-worker adversary scores,
            e.g. a fitted detector's `worker_scores_` attribute.
        task_scores
            $(N, )$ dimensional array of per-task adversary scores, e.g.
            a fitted detector's `task_scores_` attribute.
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

            weights = np.array(
                [
                    1
                    - _sigmoid(1 - worker_scores[w], frac=self.adv_frac, scale=self.scale)
                    * _sigmoid(1 - task_scores[t], frac=self.target_frac, scale=self.scale)
                    for w in t_workers
                ]
            )

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
        worker_scores: npt.NDArray,
        task_scores: npt.NDArray,
        y=None,
    ) -> npt.NDArray:
        """Fit the aggregator and return the estimated task labels.

        Equivalent to calling `fit` followed by reading `labels_`.

        Parameters
        ----------
        response_mat, worker_scores, task_scores
            See `fit`.
        y
            Ignored. Present for API consistency.

        Returns
        -------
        labels : npt.NDArray
            See `labels_`.
        """
        self.fit(response_mat, worker_scores, task_scores)
        return self.labels_
