"""
Planting synthetic adversaries into a real crowdsourcing dataset.

`plant_attacks` is the one place where a corrupted dataset is generated; the
experiment scripts (`planted_attacks.py`, `sweeps.py`) call it with their own
settings. `LOADER` loads the honest responses.
"""

import numpy as np
import numpy.typing as npt

import subcad

# ---------------------------------------------------------------------------
# Honest responses
# ---------------------------------------------------------------------------


LOADER = {
    "rte": lambda x: subcad.data.fetch_rte(x, return_X_y=True),
    "sp": lambda x: subcad.data.fetch_sp(x, return_X_y=True),
    "dog": lambda x: subcad.data.fetch_dog(x, return_X_y=True),
    "web": lambda x: subcad.data.fetch_web(x, return_X_y=True),
    "adult2": lambda x: subcad.data.fetch_adult2(x, return_X_y=True),
    "temp": lambda x: subcad.data.fetch_temp(x, return_X_y=True),
}


# ---------------------------------------------------------------------------
# Corrupting a dataset
# ---------------------------------------------------------------------------


def plant_attacks(
    honest_responses: npt.NDArray,
    gt_labels: npt.NDArray,
    seed: int,
    *,
    adv_frac: float,
    target_frac: float,
    target_obs: float,
    camo_reliability: float,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Add adversaries to the honest responses.

    `adv_frac` is the fraction of adversaries among *all* workers and
    `target_frac` the fraction of targeted tasks. Adversaries label a targeted
    task with probability `target_obs` (adversaries agree on the wrong label)
    and behave like workers of reliability `camo_reliability` elsewhere,
    labeling as many tasks as the average honest worker does.

    Returns the response matrix (workers are shuffled), the ground-truth
    adversary indicator of the workers and the targeted indicator of the tasks.
    """
    n_honests, n_tasks = honest_responses.shape
    honest_obs = np.mean(np.count_nonzero(honest_responses, axis=1) / n_tasks)
    n_adversaries = int(adv_frac * n_honests / (1 - adv_frac))

    rng = np.random.default_rng(seed)
    adv_responses, gt_targeted = subcad.data.make_adversaries(
        gt_labels,
        n_adversaries,
        target_frac=target_frac,
        camo_obs=honest_obs,
        target_obs=target_obs,
        camo_reliability=camo_reliability,
        random_state=rng,
    )

    response_mat = np.vstack([honest_responses, adv_responses])
    gt_adversaries = np.hstack([np.zeros(n_honests), np.ones(n_adversaries)])
    idx = np.arange(response_mat.shape[0])
    rng.shuffle(idx)

    return response_mat[idx, :], gt_adversaries[idx], gt_targeted
