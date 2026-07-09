import os

from pathlib import Path

import numpy.typing as npt

from scipy.io import loadmat
from sklearn.utils import Bunch

from .base import _preprocess


def load_dataset(
    data_home: os.PathLike | str,
    name: str,
    return_X_y: bool = False,
) -> Bunch | tuple[npt.NDArray, npt.NDArray]:
    """Load a `.mat`-format crowdsourcing benchmark dataset.

    Parameters
    ----------
    data_home :
        The directory under which to look for `{name}.mat`.
    name :
        Name of the dataset to load (e.g. `sp`, `dog`, `web`, `adult2`, `temp`).
    return_X_y :
        If True, returns `(response_mat, gt_labels)` instead of a
        [`sklearn.utils.Bunch`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.Bunch.html)
        object, mirroring the `return_X_y` convention used by
        `sklearn.datasets` loaders.

    Returns
    -------
    data : Bunch | tuple[npt.NDArray, npt.NDArray]
        If `return_X_y` is False, a `Bunch` with the following fields:

        - `data` : $(M, N)$ dimensional matrix where `data[i, j]` is the label
          provided by ith worker for jth task. `data[i, j] = 0` if no label is
          provided by the ith worker for jth task.
        - `target` : $(N, )$ dimensional vector where `target[i]` is the
          ground truth label of ith task.

        If `return_X_y` is True, the `(data, target)` tuple is returned directly.
    """

    fname = Path(data_home, f"{name}.mat")
    data_dict = loadmat(fname, squeeze_me=True)
    response_mat = data_dict["f"]
    gt_labels = data_dict["y"]

    # Remove tasks with no ground truth information
    valid_tasks = data_dict["valid_index"] - 1
    response_mat = response_mat[:, valid_tasks]
    gt_labels = gt_labels[valid_tasks]

    response_mat = _preprocess(response_mat)

    if return_X_y:
        return response_mat, gt_labels
    return Bunch(data=response_mat, target=gt_labels)
