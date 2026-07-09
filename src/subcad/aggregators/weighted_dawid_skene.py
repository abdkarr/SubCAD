import numpy as np
import numpy.typing as npt

from scipy import sparse

from .weighted_majority_voting import WeightedMajorityVoting, _sigmoid


class WeightedDawidSkene:
    r"""Adversary-aware weighted Dawid-Skene label aggregator.

    Same Dawid-Skene EM algorithm as `DawidSkene`, except each worker's
    contribution to a task's onehot label (used to seed the confusion
    matrices) is downweighted by a sigmoid of its adversary score
    (`worker_scores`) and that task's targeted-task score (`task_scores`),
    same mechanism as `WeightedMajorityVoting`, which is also used to
    initialize EM here instead of plain majority voting.

    !!! Example
        ```python
        from subcad import GreedyDetector
        from subcad.aggregators import WeightedDawidSkene

        detector = GreedyDetector(kind="weighted").fit(response_mat)
        labels_hat = WeightedDawidSkene().fit_predict(
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
    max_iter
        Maximum number of EM iterations.
    tol
        Tolerance used for convergence, see `DawidSkene`.

    Attributes
    ----------
    labels_ : npt.NDArray
        $(N, )$ dimensional array where `labels_[i]` is the label
        estimated for $i$th task. Set after calling `fit`.
    probs_ : npt.NDArray
        $(N, K)$ dimensional array where `probs_[i, j]` is the probability
        of $i$th task being $j$th class. Set after calling `fit`.
    confusion_mats_ : npt.NDArray
        $(M, K, K)$ dimensional array where `confusion_mats_[i, :, :]` is
        the estimated confusion matrix of $i$th worker. Set after calling
        `fit`.
    class_priors_ : npt.NDArray
        $(K, )$ dimensional array of estimated class priors. Set after
        calling `fit`.
    """

    def __init__(
        self,
        adv_frac: float = 0.2,
        target_frac: float = 0.2,
        scale: float = 5,
        max_iter: int = 100,
        tol: float = 1e-6,
    ):
        self.adv_frac = adv_frac
        self.target_frac = target_frac
        self.scale = scale
        self.max_iter = max_iter
        self.tol = tol

    def fit(
        self,
        response_mat: npt.NDArray,
        worker_scores: npt.NDArray,
        task_scores: npt.NDArray,
        y=None,
    ) -> "WeightedDawidSkene":
        """Estimate task labels via adversary-aware weighted Dawid-Skene.

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
        n_workers, n_tasks = response_mat.shape

        responses = response_mat[response_mat > 0]
        class_ids = np.unique(responses)
        n_classes = len(class_ids)
        class_to_idx = {c: i for i, c in enumerate(class_ids)}

        # Initialize structure for EM algorithm
        onehot_labels = []
        confusion_mats = []
        mv_labels = WeightedMajorityVoting(
            adv_frac=self.adv_frac, target_frac=self.target_frac, scale=self.scale
        ).fit_predict(response_mat, worker_scores, task_scores)
        for m in range(n_workers):
            m_tasks = np.where(response_mat[m, :])[0]
            m_responses = np.array([class_to_idx[l] for l in response_mat[m, m_tasks]])

            weights = [
                1
                - _sigmoid(1 - worker_scores[m], frac=self.adv_frac, scale=self.scale)
                * _sigmoid(1 - task_scores[t], frac=self.target_frac, scale=self.scale)
                for t in m_tasks
            ]

            onehot_labels.append(
                sparse.csr_array(
                    (weights, (m_tasks, m_responses)), shape=(n_tasks, n_classes)
                )
            )

            # Calculate m's confusion matrix from MV labels
            m_confusion = np.zeros((n_classes, n_classes))
            m_tasks_gt = mv_labels[m_tasks]
            for k1 in range(n_classes):
                m_tasks_in_k1 = m_tasks_gt == class_ids[k1]
                for k2 in range(n_classes):
                    # Number of tasks in class k1 that is labeled as k2 by mth worker
                    m_confusion[k2, k1] = np.sum(m_responses[m_tasks_in_k1] == k2)

                if np.sum(m_confusion[:, k1]) == 0:
                    m_confusion[:, k1] = np.ones(n_classes)
                m_confusion[:, k1] /= np.sum(m_confusion[:, k1])

            confusion_mats.append(m_confusion)
        confusion_mats = np.array(confusion_mats)

        # Calculate class priors from MV labels
        class_priors = np.zeros(n_classes)
        for k1 in range(n_classes):
            class_priors[k1] = np.sum(mv_labels == class_ids[k1]) / n_tasks

        probs = np.ones((n_tasks, n_classes)) / n_classes

        # EM iterations
        for _ in range(self.max_iter):
            probs_prev = probs
            confusion_mats_prev = confusion_mats
            class_priors_prev = class_priors

            # E-step
            probs = np.array(
                [
                    onehot_labels[m] @ np.log(confusion_mats[m] + 1e-6)
                    for m in range(n_workers)
                ]
            ).sum(axis=0)
            probs += np.log(class_priors)
            probs = np.exp(probs)
            probs /= np.sum(probs, axis=1, keepdims=True)

            # M-step
            class_priors = np.sum(probs, axis=0)
            class_priors /= np.sum(class_priors)

            confusion_mats = np.array(
                [onehot_labels[m].T @ probs for m in range(n_workers)]
            )
            normalizer = np.sum(confusion_mats, axis=1, keepdims=True)
            normalizer[normalizer == 0] = 1
            confusion_mats /= normalizer

            # Check convergence
            probs_change = np.linalg.norm((probs_prev - probs).flatten()) / n_tasks
            confusion_mats_chage = (
                np.linalg.norm((confusion_mats_prev - confusion_mats).flatten()) / n_workers
            )
            class_priors_change = np.linalg.norm(
                (class_priors_prev - class_priors).flatten()
            )

            if (
                (probs_change < self.tol)
                & (confusion_mats_chage < self.tol)
                & (class_priors_change < self.tol)
            ):
                break

        self.labels_ = class_ids[np.argmax(probs, axis=1)]
        self.probs_ = probs
        self.confusion_mats_ = confusion_mats
        self.class_priors_ = class_priors

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
