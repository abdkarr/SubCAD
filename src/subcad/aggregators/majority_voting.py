import numpy as np

import numpy.typing as npt

from scipy import stats


class MajorityVoting:
    """Majority-voting label aggregator for a crowdsourced dataset.

    Each task's label is estimated as the mode of the labels provided by
    the workers who labeled it.

    !!! Example
        ```python
        from subcad.aggregators import MajorityVoting

        labels_hat = MajorityVoting().fit_predict(response_mat)
        ```

    Attributes
    ----------
    labels_ : npt.NDArray
        $(N, )$ dimensional array where `labels_[i]` is the label of $i$th
        task estimated by majority voting, or `0` if no worker labeled
        task $i$ (e.g. after upstream adversary elimination stripped it of
        every labeler) -- the same sentinel `response_mat` itself uses for
        "no label given", rather than leaving it to `scipy.stats.mode`'s
        `nan` result for an empty slice to be silently (and, depending on
        `response_mat`'s dtype, not always consistently) cast into
        `labels_`'s dtype.
    """

    def fit(self, response_mat: npt.NDArray, y=None) -> "MajorityVoting":
        """Estimate task labels via majority voting.

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
        has_labels = np.any(response_mat != 0, axis=0)

        labels = np.zeros(response_mat.shape[1], dtype=response_mat.dtype)
        if np.any(has_labels):
            labeled_mat = response_mat[:, has_labels]
            masked_responses = np.ma.masked_array(labeled_mat, labeled_mat == 0)
            labels[has_labels] = stats.mode(masked_responses, axis=0).mode

        self.labels_ = labels

        return self

    def fit_predict(self, response_mat: npt.NDArray, y=None) -> npt.NDArray:
        """Fit the aggregator and return the estimated task labels.

        Equivalent to calling `fit` followed by reading `labels_`.

        Parameters
        ----------
        response_mat
            See `fit`.
        y
            Ignored. Present for API consistency.

        Returns
        -------
        labels : npt.NDArray
            See `labels_`.
        """
        self.fit(response_mat)
        return self.labels_
