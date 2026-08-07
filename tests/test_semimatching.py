import numpy as np
import pytest

from subcad.utils import OptimalSemiMatching


def test_star_graph_forces_full_degree():
    # 1 left node connected to every right node -- no choice but degree 3.
    biadj = np.array([[1, 1, 1]])
    osm = OptimalSemiMatching().fit(biadj)
    assert list(osm.degrees_) == [3]
    assert list(osm.matching_) == [0, 0, 0]


def test_complete_bipartite_balances_evenly():
    # K_{2,2}: optimal semi-matching must split 1-1, not 2-0.
    biadj = np.array([[1, 1], [1, 1]])
    osm = OptimalSemiMatching().fit(biadj)
    assert sorted(osm.degrees_) == [1, 1]


def test_load_balance_tradeoff():
    # right2 can go to either left node; assigning it to left1 (which
    # otherwise has degree 0) is strictly cheaper than piling a 3rd unit
    # onto left0.
    biadj = np.array([[1, 1, 1], [0, 0, 1]])
    osm = OptimalSemiMatching().fit(biadj)
    assert sorted(osm.degrees_) == [1, 2]
    assert osm.matching_[2] == 1


def test_degree_sequence_is_order_invariant():
    # Lemma 28 (paper): the optimal semi-matching's sorted degree sequence
    # is unique regardless of left-node tie-break ordering.
    biadj = np.array(
        [
            [1, 1, 0, 1],
            [1, 0, 1, 1],
            [0, 1, 1, 0],
        ]
    )
    baseline = sorted(OptimalSemiMatching().fit(biadj).degrees_)

    rng = np.random.default_rng(0)
    for _ in range(5):
        perm = rng.permutation(biadj.shape[0])
        permuted = sorted(OptimalSemiMatching().fit(biadj[perm, :]).degrees_)
        assert permuted == baseline


def test_matching_respects_graph_edges():
    rng = np.random.default_rng(1)
    biadj = rng.random((6, 9)) < 0.4
    # Every right node needs at least one neighbor for feasibility.
    for j in range(biadj.shape[1]):
        if not biadj[:, j].any():
            biadj[rng.integers(biadj.shape[0]), j] = True

    osm = OptimalSemiMatching().fit(biadj)
    for j, i in enumerate(osm.matching_):
        assert biadj[i, j]
    assert np.array_equal(np.bincount(osm.matching_, minlength=biadj.shape[0]), osm.degrees_)


def test_raises_on_unmatchable_right_node():
    biadj = np.array([[1, 0], [0, 0]])
    with pytest.raises(ValueError):
        OptimalSemiMatching().fit(biadj)
