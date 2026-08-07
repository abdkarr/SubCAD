import networkx as nx
import numpy as np
import numpy.typing as npt


class OptimalSemiMatching:
    r"""Load-balancing-optimal semi-matching of a bipartite graph.

    A *semi-matching* of a bipartite graph $G = (U, V, E)$ matches every
    node in $V$ to exactly one incident node in $U$ (nodes in $U$ are
    unconstrained -- their degree in the matching $M$, $\deg_M(u)$, can be
    anything from $0$ up to $\deg_G(u)$). Harvey, Ladner, Lovász & Tamir [1]
    define the *optimal* semi-matching as the one minimizing
    $\sum_{u \in U} \mathrm{cost}_M(u)$ with $\mathrm{cost}_M(u) =
    \binom{\deg_M(u) + 1}{2}$ -- the same convex cost used to analyze
    scheduling/load-balancing problems, so an optimal semi-matching spreads
    $V$ across $U$ as evenly as the graph's structure allows.

    [1] give an $O(|V||E|)$ combinatorial algorithm based on repeated
    "cost-reducing path" augmentation. This implementation instead solves
    the *equivalent* convex-cost min-cost-flow problem: since
    $\mathrm{cost}_M(u) = \sum_{k=1}^{\deg_M(u)} k$ is a separable convex
    function of $\deg_M(u)$, it is exactly representable by giving $u$ one
    unit-capacity "slot" arc of cost $k$ for each $k = 1, \ldots,
    \deg_G(u)$ -- a standard reduction for convex-cost flow (e.g. Ahuja,
    Magnanti & Orlin, *Network Flows*, Sec. 14.4) -- and solved exactly via
    `networkx`'s network simplex (`nx.min_cost_flow`), which is guaranteed
    to return an integral optimum on this integer-capacity network. Same
    optimality criterion as [1], a different (also exact) solution method.

    !!! Example
        ```python
        from subcad.utils import OptimalSemiMatching

        # 2 left nodes, 3 right nodes
        biadj = [[1, 1, 0], [0, 1, 1]]
        osm = OptimalSemiMatching().fit(biadj)
        osm.degrees_    # per-left-node degree in the optimal semi-matching
        osm.matching_   # per-right-node index of its assigned left node
        ```

    Attributes
    ----------
    degrees_ : npt.NDArray
        $(M,)$ dimensional array where `degrees_[i]` is left node $i$'s
        degree in the optimal semi-matching, $\deg_M(u_i)$. Set after
        calling `fit`.
    matching_ : npt.NDArray
        $(N,)$ dimensional array where `matching_[j]` is the index of the
        left node matched to right node $j$. Set after calling `fit`.

    References
    ----------
    [1] Harvey, Nicholas JA, Richard E. Ladner, László Lovász, and Tami
    Tamir. "Semi-matchings for bipartite graphs and load balancing."
    Workshop on Algorithms and Data Structures. Springer, 2003.
    """

    def fit(self, biadj_mat: npt.ArrayLike) -> "OptimalSemiMatching":
        """Compute the optimal semi-matching of a bipartite graph.

        Parameters
        ----------
        biadj_mat
            $(M, N)$ dimensional bi-adjacency matrix of a bipartite graph
            with $M$ left nodes and $N$ right nodes; `biadj_mat[i, j]`
            truthy iff left node $i$ and right node $j$ are connected.
            Every right node must have at least one incident left node.

        Returns
        -------
        self
        """
        biadj_mat = np.asarray(biadj_mat) != 0
        n_left, n_right = biadj_mat.shape

        left_degrees_in_graph = biadj_mat.sum(axis=1)
        right_degrees_in_graph = biadj_mat.sum(axis=0)
        if n_right > 0 and np.any(right_degrees_in_graph == 0):
            raise ValueError(
                "Every right node must have at least one incident left "
                "node to admit a semi-matching."
            )

        graph = nx.DiGraph()
        graph.add_node("sink", demand=int(n_right))
        for j in range(n_right):
            graph.add_node(("right", j), demand=-1)
        for i in range(n_left):
            graph.add_node(("left", i), demand=0)
            for k in range(1, int(left_degrees_in_graph[i]) + 1):
                slot = ("slot", i, k)
                graph.add_edge(("left", i), slot, capacity=1, weight=0)
                graph.add_edge(slot, "sink", capacity=1, weight=k)
        for i in range(n_left):
            for j in np.flatnonzero(biadj_mat[i]):
                graph.add_edge(("right", int(j)), ("left", i), capacity=1, weight=0)

        flow = nx.min_cost_flow(graph) if n_right > 0 else {}

        degrees = np.zeros(n_left, dtype=int)
        matching = np.full(n_right, -1, dtype=int)
        for j in range(n_right):
            for node, sent in flow.get(("right", j), {}).items():
                if sent > 0:
                    left_idx = node[1]
                    matching[j] = left_idx
                    degrees[left_idx] += 1
                    break

        self.degrees_ = degrees
        self.matching_ = matching
        return self

    def fit_predict(self, biadj_mat: npt.ArrayLike) -> npt.NDArray:
        """Fit the semi-matching and return the per-left-node degree sequence.

        Equivalent to calling `fit` followed by reading `degrees_`.

        Parameters
        ----------
        biadj_mat
            See `fit`.

        Returns
        -------
        degrees : npt.NDArray
            See `degrees_`.
        """
        self.fit(biadj_mat)
        return self.degrees_
