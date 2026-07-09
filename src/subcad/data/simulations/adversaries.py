import numpy as np
import numpy.typing as npt

from ...utils import check_rng
from .confusion_matrix import make_confusion_matrix
from .worker_labels import make_worker_labels


def _draw_targeted_responses(
    gt_labels: npt.NDArray,
    n_workers: int,
    p_obs: float,
    rng: np.random.Generator,
):
    n_tasks = len(gt_labels)
    class_ids = np.unique(gt_labels)

    response_mat = np.zeros((n_workers, n_tasks))
    for k in class_ids:
        # Targeted tasks in class k
        tasks_in_k = gt_labels == k

        # Possible labels adversaries can provide for tasks in k
        possible_labels = np.setdiff1d(class_ids, k)

        # All targeted tasks in k labeled with the same label by all adversaries
        response_mat[:, tasks_in_k] = rng.choice(possible_labels)

    if p_obs < 1:
        response_mat *= rng.binomial(1, p_obs, size=(n_workers, n_tasks))

    return response_mat


def make_adversaries(
    gt_labels: npt.NDArray,
    n_adversaries: int,
    target_frac: float,
    camo_obs: float,
    target_obs: float = 1,
    camo_reliability: float = 1,
    targeted_tasks=None,
    random_state=None,
) -> tuple[npt.NDArray, npt.NDArray]:
    r"""Generate a simulated response matrix for adversarial workers.

    The function selects a set of tasks that are considered to be targeted by
    adversaries. Adversaries provide a wrong label to targeted points. Each
    adversary also provide labels for `p_camo` fraction of the remaining points
    as a way to camouflage themselves. Labeling behaviour of adversaries are
    follows:

    For targeted tasks:

    - For a targeted task $n$ that is in class $k$, an adversary provide the
    the label $\widehat{y}_{mn} = k' \neq k$ with probability `target_obs` and
    $\widehat{y}_{mn} = 0$, otherwise. $k'$ is the same for all targeted $n$'s
    in class k. This can be considered as adversaries trying to flip label of $n$
    from $k$ to $k'$.

    For camouflage:

    - For a non-targeted task $n$, each adversary independently provide a label
    with probability `camo_obs`, i.e., $\widehat{y}_{mn} = k \in {\cal C}$
    with probability `camo_obs` and $\widehat{y}_{mn} = 0$, otherwise. $\cal C$
    is the set of class ids. $k$ is sampled using Dawid-Skene model whose confusion
    matrix is drawn using
    [`make_confusion_matrix`][subcad.data.simulations.confusion_matrix.make_confusion_matrix]
    function with reliability set to `camo_reliability`.

    ??? Example
        The following code first generates random ground truth labels for 1000
        tasks where each task belongs to one of 3 classes. Responses for 20
        adversaries are then generated.

        ```python
        import numpy as np
        import subcad

        # Generate random class labels
        rng = np.random.default_rng()
        gt_labels = rng.integers(1, 4, 1000)

        subcad.data.make_adversaries(gt_labels, 20, target_frac=0.1, camo_obs=0.1)
        ```

    Parameters
    ----------
    gt_labels
        Ground truth labels of tasks
    n_adversaries
        Number of adversaries to produce responses for
    target_frac
        Fraction of tasks that is selected as targeted tasks
    camo_obs
        Probability of an adversary providing a label for a non-targeted task for
        camouflage.
    target_obs
        Probability of an adversary providing a label for a targeted point
    camo_reliability
        Reliability of adversaries when they are labeling non-targeted points,
        see [`make_confusion_matrix`][subcad.data.simulations.confusion_matrix.make_confusion_matrix]
        for the definition of reliability.
    random_state
        Controls the randomness used to draw responses. Pass an `int` for
        reproducible results across calls, or a `np.random.Generator`, see
        `subcad.typing.RNGType`.

    Returns
    -------
    response_mat : npt.NDArray
        $(M, N)$ dimensional generated response matrix where $M$ is the number
        of adversaries and $N$ is the number of tasks.
    targeted_tasks : npt.NDArray
        $(N, )$ dimensional binary array where `targeted_tasks[i] = 1` if $i$th
        task is targeted, 0 otherwise.
    """

    # Input Checks
    rng = check_rng(random_state)

    n_tasks = len(gt_labels)
    n_targeted = int(np.floor(n_tasks * target_frac))

    class_ids = np.unique(gt_labels)
    n_classes = len(class_ids)

    # Draw the set of tasks targeted by adversaries
    if targeted_tasks is None:
        targeted_tasks = np.zeros(n_tasks, dtype=np.bool)
        targeted_tasks[rng.choice(n_tasks, n_targeted, replace=False)] = 1

    # Determine labels for targeted points
    targeted_tasks_gt = gt_labels[targeted_tasks]
    targeted_response = _draw_targeted_responses(
        targeted_tasks_gt, n_adversaries, target_obs, rng
    )

    # Determine labels for camouflage
    confusion_mats = [
        make_confusion_matrix(n_classes, camo_reliability, rng)
        for _ in range(n_adversaries)
    ]
    camouflage_response = make_worker_labels(
        gt_labels[~targeted_tasks], confusion_mats, camo_obs, rng
    )

    response_mat = np.zeros((n_adversaries, n_tasks), dtype=np.int64)
    response_mat[:, targeted_tasks] = targeted_response
    response_mat[:, ~targeted_tasks] = camouflage_response

    return response_mat, targeted_tasks.astype(np.int64)
