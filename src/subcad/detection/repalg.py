import numpy.typing as npt

import numpy as np

from ..utils.semimatching import OptimalSemiMatching


def _conflict_side_masks(
    response_mat: npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Split a binary response matrix into its two per-label boolean masks
    and flag the conflict tasks (tasks with at least one label of each
    kind) -- the only tasks informative for reputation scoring, since
    consensus tasks can't distinguish honest from adversarial workers.
    """
    class_ids = np.unique(response_mat[response_mat > 0])
    if len(class_ids) != 2:
        raise ValueError(
            "SoftPenaltyDetector/HardPenaltyDetector only support binary "
            f"classification, got {len(class_ids)} classes."
        )

    pos_mask = response_mat == class_ids[0]
    neg_mask = response_mat == class_ids[1]
    conflict = np.any(pos_mask, axis=0) & np.any(neg_mask, axis=0)

    return pos_mask, neg_mask, conflict


class SoftPenaltyDetector:
    r"""Reputation-based adversary detector using RepAlg's soft-penalty
    algorithm [1].
    
    For a conflict task $j$  -- labeled both ways by at least one worker each -- 
    with $d_j^+$ workers labeling one class and $d_j^-$ the other, a worker on 
    either side is penalized $1/d_j^+$ or $1/d_j^-$ (whichever side it's on) 
    -- so the penalty budget of $1$ per task is spread evenly across everyone 
    on that side, rewarding the less-agreed-with side less. `worker_scores_` is 
    each worker's penalty averaged over its own conflict tasks (normalizing for 
    how many conflict tasks it was even exposed to); higher means more likely
    adversarial, matching `subcad.detection`'s scoring convention.

    !!! Example
        ```python
        from subcad.detection import SoftPenaltyDetector

        worker_scores = SoftPenaltyDetector().fit_predict(response_mat)
        ```

    Attributes
    ----------
    worker_scores_ : npt.NDArray
        $(M,)$ dimensional array where `worker_scores_[i]` is the
        reputation penalty of $i$th worker; higher means more likely to
        be adversarial. Set after calling `fit`.

    References
    ----------
    [1] Jagabathula, Srikanth, Lakshminarayanan Subramanian, and Ashwin
    Venkataraman. "Identifying unreliable and adversarial workers in
    crowdsourced labeling tasks." Journal of Machine Learning Research 18,
    no. 93 (2017): 1-67.
    """

    def fit(self, response_mat: npt.NDArray, y=None) -> "SoftPenaltyDetector":
        """Detect adversarial workers.

        Parameters
        ----------
        response_mat
            $(M, N)$ dimensional matrix where `response_mat[i, j]` is the
            label provided by $i$th worker for $j$th task.
            `response_mat[i, j] = 0` is assumed to indicate no label is
            given by $i$th worker for $j$th task. Must be binary (exactly
            two distinct positive labels present).
        y
            Ignored. Present for API consistency.

        Returns
        -------
        self
        """
        n_workers = response_mat.shape[0]
        pos_mask, neg_mask, conflict = _conflict_side_masks(response_mat)

        n_pos = pos_mask.sum(axis=0)
        n_neg = neg_mask.sum(axis=0)
        inv_pos = np.divide(
            1.0, n_pos, out=np.zeros(n_pos.shape), where=n_pos > 0
        )
        inv_neg = np.divide(
            1.0, n_neg, out=np.zeros(n_neg.shape), where=n_neg > 0
        )

        pos_contrib = pos_mask & conflict
        neg_contrib = neg_mask & conflict

        penalties = np.zeros(response_mat.shape)
        penalties[pos_contrib] = np.broadcast_to(inv_pos, response_mat.shape)[
            pos_contrib
        ]
        penalties[neg_contrib] = np.broadcast_to(inv_neg, response_mat.shape)[
            neg_contrib
        ]

        n_conflict_labeled = (pos_contrib | neg_contrib).sum(axis=1)
        worker_scores = np.divide(
            penalties.sum(axis=1),
            n_conflict_labeled,
            out=np.zeros(n_workers),
            where=n_conflict_labeled > 0,
        )

        self.worker_scores_ = worker_scores

        return self

    def fit_predict(self, response_mat: npt.NDArray, y=None) -> npt.NDArray:
        """Fit the detector and return worker adversary scores.

        Equivalent to calling `fit` followed by reading `worker_scores_`.

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
        """
        self.fit(response_mat)
        return self.worker_scores_


class HardPenaltyDetector:
    r"""Reputation-based adversary detector using RepAlg's hard-penalty
    algorithm [1].

    Like `SoftPenaltyDetector`, only "conflict" tasks (labeled both ways
    by at least one worker each) are used. But instead of spreading each
    conflict task's penalty budget of $1$ evenly across every worker on a
    side, hard-penalty gives the *entire* penalty to a single
    representative worker per side per task. Representatives are chosen
    via an optimal semi-matching (`subcad.utils.OptimalSemiMatching`)
    between workers and task-copies $t_j^+/t_j^-$ (one copy per side of
    each conflict task $j$, worker $i$ connected to $t_j^+$ iff it gave
    the "+"-side label, symmetrically for $t_j^-$). `worker_scores_` is each 
    worker's degree in the resulting semi-matching, *without* the 
    degree-normalization `SoftPenaltyDetector` applies -- by design, per the 
    paper's discussion of why the lack of normalization is what makes 
    hard-penalty robust to high-degree, colluding adversaries (whereas 
    `SoftPenaltyDetector` is the better fit for low-degree adversaries).

    !!! Example
        ```python
        from subcad.detection import HardPenaltyDetector

        worker_scores = HardPenaltyDetector().fit_predict(response_mat)
        ```

    Attributes
    ----------
    worker_scores_ : npt.NDArray
        $(M,)$ dimensional array where `worker_scores_[i]` is $i$th
        worker's degree in the optimal semi-matching; higher means more
        likely to be adversarial. Set after calling `fit`.

    References
    ----------
    [1] Jagabathula, Srikanth, Lakshminarayanan Subramanian, and Ashwin
    Venkataraman. "Identifying unreliable and adversarial workers in
    crowdsourced labeling tasks." Journal of Machine Learning Research 18,
    no. 93 (2017): 1-67.
    """

    def fit(self, response_mat: npt.NDArray, y=None) -> "HardPenaltyDetector":
        """Detect adversarial workers.

        Parameters
        ----------
        response_mat
            $(M, N)$ dimensional matrix where `response_mat[i, j]` is the
            label provided by $i$th worker for $j$th task.
            `response_mat[i, j] = 0` is assumed to indicate no label is
            given by $i$th worker for $j$th task. Must be binary (exactly
            two distinct positive labels present).
        y
            Ignored. Present for API consistency.

        Returns
        -------
        self
        """
        n_workers = response_mat.shape[0]
        pos_mask, neg_mask, conflict = _conflict_side_masks(response_mat)

        conflict_task_ids = np.flatnonzero(conflict)
        n_conflict_tasks = len(conflict_task_ids)

        # Right nodes = one task-copy per side per conflict task, ordered
        # [t_0^+, t_0^-, t_1^+, t_1^-, ...].
        task_copy_biadj = np.zeros((n_workers, 2 * n_conflict_tasks), dtype=bool)
        task_copy_biadj[:, 0::2] = pos_mask[:, conflict_task_ids]
        task_copy_biadj[:, 1::2] = neg_mask[:, conflict_task_ids]

        semimatching = OptimalSemiMatching().fit(task_copy_biadj)

        self.worker_scores_ = semimatching.degrees_.astype(float)

        return self

    def fit_predict(self, response_mat: npt.NDArray, y=None) -> npt.NDArray:
        """Fit the detector and return worker adversary scores.

        Equivalent to calling `fit` followed by reading `worker_scores_`.

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
        """
        self.fit(response_mat)
        return self.worker_scores_
