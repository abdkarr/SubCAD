import numpy.typing as npt

import numpy as np
import networkx as nx

from .base import BaseDetector


class GreedyPPDetector(BaseDetector):
    """Adversary detector using the Greedy++ peeling algorithm [1].

    See `BaseDetector` for the shared detection procedure, parameters, and
    fitted attributes.

    !!! Example
        ```python
        from subcad import GreedyPPDetector

        detector = GreedyPPDetector(kind="weighted", iterations=10)
        worker_scores, task_scores = detector.fit_predict(response_mat)
        ```

    Parameters
    ----------
    kind
        See `BaseDetector`.
    iterations
        Number of Greedy++ peeling passes to run. Each pass carries
        accumulated node loads from the previous ones. The order returned
        is from whichever pass achieved the densest subgraph, defaulting
        to pass 1 if no later pass improves on it.

    References
    ----------
    [1] Boob, Digvijay, et al. "Flowless: Extracting densest subgraphs
    without flow computations." Proceedings of The Web Conference 2020.
    """

    def __init__(self, kind: str = "binary", iterations: int = 10):
        super().__init__(kind=kind)
        self.iterations = iterations

    def _peel(self, biadj_mat: npt.NDArray):
        """
        Run the Greedy++ peeling algorithm [1] on a bipartite graph.
        Greedy++ reruns Charikar/Fraudar-style peeling for
        `self.iterations` iterations, carrying accumulated node "loads"
        across iterations, which tends to converge to a denser subgraph
        than a single greedy pass.

        Each pass's own best-density point (remaining edge weight /
        remaining node count, tracked along that pass's removal
        trajectory) is compared across passes, and the order from the
        densest pass is returned -- defaulting to pass 1 (plain Charikar
        peeling) if no later pass beats it, so the result is never worse
        than the paper's proven 1/2-approximation baseline. This mirrors
        the paper's own prescribed "return densest subgraph found"
        behavior, but purely to pick which of the `self.iterations`
        self-generated candidate orders to trust as `worker_scores_`/
        `task_scores_` -- not to produce a standalone size estimate (see
        `subcad.selection` for that); density is not guaranteed to
        improve monotonically pass-over-pass (the paper's Conjecture 5.1
        is an asymptotic guarantee, not a per-pass one), so "always the
        last pass" has no such floor.

        References
        ----------
        [1] Boob, Digvijay, et al. "Flowless: Extracting densest subgraphs
        without flow computations." Proceedings of The Web Conference 2020.
        """
        # Construct adjacency matrix from bi-adjacency of bipartite graph
        n_workers, n_tasks = biadj_mat.shape
        n_nodes_total = n_workers + n_tasks
        adj = np.block(
            [
                [np.zeros((n_workers, n_workers)), biadj_mat],
                [biadj_mat.T, np.zeros((n_tasks, n_tasks))],
            ]
        )
        G = nx.from_numpy_array(adj)
        total_graph_weight = G.size(weight="weight")

        loads = dict.fromkeys(G.nodes, 0)  # Load vector for Greedy++.

        best_node_order = None
        best_density = -np.inf

        for _ in range(self.iterations):
            # Initialize heap for fast access to minimum weighted degree.
            heap = nx.utils.BinaryHeap()

            # Compute initial weighted degrees and add nodes to the heap.
            for node, degree in G.degree(weight="weight"):
                heap.insert(node, loads[node] + degree)

            # Set up tracking for current graph state.
            remaining_nodes = set(G.nodes)
            current_degrees = dict(G.degree(weight="weight"))

            iter_node_order = []
            remaining_weight = total_graph_weight
            iter_best_density = remaining_weight / n_nodes_total
            while remaining_nodes:
                # Pop the node with the smallest weighted degree.
                node, _ = heap.pop()
                if node not in remaining_nodes:
                    continue  # Skip nodes already removed.

                iter_node_order.append(node)

                # Update the load of the popped node.
                loads[node] += current_degrees[node]
                remaining_weight -= current_degrees[node]

                # Update neighbors' degrees and the heap.
                for neighbor in G.neighbors(node):
                    if neighbor in remaining_nodes:
                        current_degrees[neighbor] -= G[node][neighbor]["weight"]
                        heap.insert(neighbor, loads[neighbor] + current_degrees[neighbor])

                # Remove the node from the remaining nodes.
                remaining_nodes.remove(node)

                n_remaining = len(remaining_nodes)
                if n_remaining > 0:
                    curr_density = remaining_weight / n_remaining
                    if curr_density > iter_best_density:
                        iter_best_density = curr_density

            if iter_best_density > best_density:
                best_density = iter_best_density
                best_node_order = np.array(iter_node_order)

        worker_order = best_node_order[best_node_order < n_workers]
        task_order = best_node_order[best_node_order >= n_workers] - n_workers

        return worker_order, task_order
