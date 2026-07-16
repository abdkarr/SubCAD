"""Enhanced Bayesian Classifier Combination (EBCC), a label-aggregation baseline.

Port of `research/scripts/ebcc.py`, a baseline used for comparison against
`subcad.aggregators` rather than a `subcad` design of our own.
"""

import numpy as np
import numpy.typing as npt
import scipy.sparse as ssp
from scipy.special import digamma, gammaln
from scipy.stats import dirichlet, entropy

from ..typing import RNGType
from ..utils import check_rng


def _ebcc_vb(
    tuples: npt.NDArray,
    num_groups: int = 10,
    a_pi: float = 0.1,
    alpha: float = 1,
    a_v: float = 4,
    b_v: float = 1,
    seed: int = 1234,
    max_iter: int = 500,
    empirical_prior: bool = False,
) -> tuple[npt.NDArray, float]:
    """Single restart of the EBCC variational-Bayes fit.

    Parameters
    ----------
    tuples
        `(n_annotations, 3)` array of `(item, worker, label)` triples, all
        zero-indexed.
    num_groups, a_pi, alpha, a_v, b_v, max_iter, empirical_prior
        EBCC hyperparameters, unchanged from `research/scripts/ebcc.py`.
    seed
        Seeds this restart's random group initialization.

    Returns
    -------
    z_ik : ndarray (n_items, n_classes)
        Per-item soft class assignment.
    elbo : float
        Evidence lower bound reached by this restart, used by `ebcc_labels`
        to pick the best of several random restarts.
    """
    num_items = len(np.unique(tuples[:, 0]))
    num_workers = len(np.unique(tuples[:, 1]))
    num_classes = len(np.unique(tuples[:, 2]))

    y_is_one_lij = []
    y_is_one_lji = []
    for k in range(num_classes):
        selected = tuples[:, 2] == k
        coo_ij = ssp.coo_matrix(
            (np.ones(selected.sum()), tuples[selected, :2].T),
            shape=(num_items, num_workers),
            dtype=np.bool_,
        )
        y_is_one_lij.append(coo_ij.tocsr())
        y_is_one_lji.append(coo_ij.T.tocsr())

    beta_kl = np.eye(num_classes) * (a_v - b_v) + b_v

    z_ik = np.zeros((num_items, num_classes))
    for l in range(num_classes):
        z_ik[:, [l]] += y_is_one_lij[l].sum(axis=-1)
    z_ik /= z_ik.sum(axis=-1, keepdims=True)

    if empirical_prior:
        alpha = z_ik.sum(axis=0)

    rng = np.random.default_rng(seed)
    zg_ikm = rng.dirichlet(np.ones(num_groups), z_ik.shape) * z_ik[:, :, None]
    for _ in range(max_iter):
        eta_km = a_pi / num_groups + zg_ikm.sum(axis=0)
        nu_k = alpha + z_ik.sum(axis=0)

        mu_jkml = (
            np.zeros((num_workers, num_classes, num_groups, num_classes))
            + beta_kl[None, :, None, :]
        )
        for l in range(num_classes):
            for k in range(num_classes):
                mu_jkml[:, k, :, l] += y_is_one_lji[l].dot(zg_ikm[:, k, :])

        Eq_log_pi_km = digamma(eta_km) - digamma(eta_km.sum(axis=-1, keepdims=True))
        Eq_log_tau_k = digamma(nu_k) - digamma(nu_k.sum())
        Eq_log_v_jkml = digamma(mu_jkml) - digamma(mu_jkml.sum(axis=-1, keepdims=True))

        zg_ikm[:] = Eq_log_pi_km[None, :, :] + Eq_log_tau_k[None, :, None]
        for l in range(num_classes):
            for k in range(num_classes):
                zg_ikm[:, k, :] += y_is_one_lij[l].dot(Eq_log_v_jkml[:, k, :, l])

        zg_ikm = np.exp(zg_ikm)
        zg_ikm /= zg_ikm.reshape(num_items, -1).sum(axis=-1)[:, None, None]

        last_z_ik = z_ik
        z_ik = zg_ikm.sum(axis=-1)

        if np.allclose(last_z_ik, z_ik, atol=1e-3):
            break

    elbo = (
        ((eta_km - 1) * Eq_log_pi_km).sum()
        + ((nu_k - 1) * Eq_log_tau_k).sum()
        + ((mu_jkml - 1) * Eq_log_v_jkml).sum()
    )
    elbo += dirichlet.entropy(nu_k)
    for k in range(num_classes):
        elbo += dirichlet.entropy(eta_km[k])
    elbo += (gammaln(mu_jkml) - (mu_jkml - 1) * digamma(mu_jkml)).sum()
    alpha0_jkm = mu_jkml.sum(axis=-1)
    elbo += (
        (alpha0_jkm - num_classes) * digamma(alpha0_jkm) - gammaln(alpha0_jkm)
    ).sum()
    elbo += entropy(zg_ikm.reshape(num_items, -1).T).sum()

    return z_ik, float(elbo)


def ebcc_labels(
    response_mat: npt.NDArray,
    n_restarts: int = 40,
    random_state: RNGType = None,
    **ebcc_vb_kwargs,
) -> npt.NDArray:
    """Aggregate task labels with EBCC, keeping the best of several restarts.

    Since `_ebcc_vb`'s group assignment is randomly initialized, it is run
    `n_restarts` times with different seeds and the restart reaching the
    highest ELBO is kept -- same restart-and-select strategy as
    `research/scripts/commons.py`'s `apply_ebcc`.

    Parameters
    ----------
    response_mat
        `(n_workers, n_tasks)` response matrix; see `subcad`'s response
        matrix convention (0 = unobserved, positive ints = class labels).
    n_restarts
        Number of random restarts.
    random_state
        See `subcad.typing.RNGType`. Seeds each restart's initialization.
    **ebcc_vb_kwargs
        Forwarded to `_ebcc_vb` (e.g. `num_groups`, `max_iter`).

    Returns
    -------
    labels : ndarray (n_tasks,)
        Aggregated task labels, using the same 1-indexed class convention
        as `response_mat`.
    """
    response_mat = np.asarray(response_mat)
    rng = check_rng(random_state)

    worker_idx, task_idx = np.nonzero(response_mat)
    tuples = np.column_stack(
        [task_idx, worker_idx, response_mat[worker_idx, task_idx] - 1]
    )

    best_elbo = -np.inf
    best_z_ik = None
    for _ in range(n_restarts):
        seed = int(rng.integers(0, int(1e8)))
        z_ik, elbo = _ebcc_vb(tuples, seed=seed, empirical_prior=True, **ebcc_vb_kwargs)
        if elbo > best_elbo:
            best_elbo = elbo
            best_z_ik = z_ik

    return np.argmax(best_z_ik, axis=1) + 1
