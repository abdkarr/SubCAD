import numpy as np
import numpy.typing as npt


def _preprocess(response_mat: npt.NDArray) -> npt.NDArray:
    # remove workers with fewer than 10 labels
    n_tasks_per_worker = np.count_nonzero(response_mat, axis=1)
    response_mat = response_mat[n_tasks_per_worker >= 10, :]

    return response_mat
