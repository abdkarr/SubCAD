import numpy as np
import numpy.typing as npt

from numba import njit


@njit
def calc_agreement_mat(
    response_mat: npt.NDArray, fill_diagonal: bool = False
) -> tuple[npt.NDArray, npt.NDArray]:
    n_workers = response_mat.shape[0]

    agreement_mat = np.zeros((n_workers, n_workers))
    n_co_observed = np.zeros((n_workers, n_workers))

    observed_tasks = []
    for i in range(n_workers):
        observed_tasks.append(np.where(response_mat[i, :] > 0)[0])

    for i in range(n_workers):
        if fill_diagonal:
            n_observed_i = len(observed_tasks[i])
            n_co_observed[i, i] = n_observed_i
            agreement_mat[i, i] = 1.0 if n_observed_i > 0 else 0.0

        for j in range(i + 1, n_workers):
            co_observed = np.intersect1d(
                observed_tasks[i], observed_tasks[j], assume_unique=True
            )

            n_co_observed[i, j] = len(co_observed)
            n_co_observed[j, i] = n_co_observed[i, j]

            if len(co_observed) > 0:
                responses_i = response_mat[i, co_observed]
                responses_j = response_mat[j, co_observed]
                agreement_mat[i, j] = np.mean(responses_i == responses_j)
            else:
                agreement_mat[i, j] = 0

            agreement_mat[j, i] = agreement_mat[i, j]

    return agreement_mat, n_co_observed
