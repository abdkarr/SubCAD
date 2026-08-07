import numpy as np
import pytest

from subcad.data.simulations import make_adversaries, make_confusion_matrix, make_worker_labels
from subcad.detection import HardPenaltyDetector, SoftPenaltyDetector

# response_mat[i, j] = label given by worker i for task j (0 = no label).
# Conflict tasks (both labels 1 and 2 present) are task0 and task2; task1
# and task3 are consensus/underlabeled and don't contribute.
RESPONSE_MAT = np.array(
    [
        [1, 1, 2, 1],
        [1, 1, 1, 0],
        [2, 1, 0, 0],
    ]
)


def test_soft_penalty_matches_hand_worked_example():
    scores = SoftPenaltyDetector().fit_predict(RESPONSE_MAT)
    np.testing.assert_allclose(scores, [0.75, 0.75, 1.0])


def test_hard_penalty_matches_hand_worked_example():
    # t0+ (workers 0,1) forces one of them to degree 2; t0- (worker 2),
    # t2+ (worker 1), t2- (worker 0) are each forced to their sole
    # neighbor. Sorted degree sequence is uniquely [1, 1, 2].
    scores = HardPenaltyDetector().fit_predict(RESPONSE_MAT)
    assert sorted(scores) == [1.0, 1.0, 2.0]


@pytest.mark.parametrize("detector_cls", [SoftPenaltyDetector, HardPenaltyDetector])
def test_detectors_have_no_task_scores_attribute(detector_cls):
    detector = detector_cls().fit(RESPONSE_MAT)
    assert not hasattr(detector, "task_scores_")


@pytest.mark.parametrize("detector_cls", [SoftPenaltyDetector, HardPenaltyDetector])
def test_raises_on_non_binary_response_mat(detector_cls):
    response_mat = np.array([[1, 2, 3], [3, 1, 2]])
    with pytest.raises(ValueError):
        detector_cls().fit(response_mat)


@pytest.mark.parametrize("detector_cls", [SoftPenaltyDetector, HardPenaltyDetector])
def test_scores_separate_honest_from_adversarial(detector_cls):
    rng = np.random.default_rng(0)
    n_classes = 2
    n_tasks = 400
    n_honest = 40
    n_adversaries = 10

    gt_labels = rng.integers(1, n_classes + 1, n_tasks)

    confusion_mats = [
        make_confusion_matrix(n_classes, reliability=8, random_state=rng)
        for _ in range(n_honest)
    ]
    honest_responses = make_worker_labels(gt_labels, confusion_mats, p_obs=0.3, random_state=rng)

    adversary_responses, _ = make_adversaries(
        gt_labels,
        n_adversaries,
        target_frac=0.3,
        camo_obs=0.2,
        camo_reliability=1,
        random_state=rng,
    )

    response_mat = np.vstack([honest_responses, adversary_responses]).astype(int)
    is_adversary = np.concatenate([np.zeros(n_honest), np.ones(n_adversaries)]).astype(bool)

    scores = detector_cls().fit_predict(response_mat)

    assert scores[is_adversary].mean() > scores[~is_adversary].mean()
