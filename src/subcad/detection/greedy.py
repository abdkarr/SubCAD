import numpy.typing as npt

import numpy as np

from scipy import sparse

from .base import _calc_adversary_scores, _construct_biadj_mat


class MinTree:
    """
    A tree structure that can be used to efficiently find the minimum value in
    an array while allowing updates to values of array entries. . Implementation
    is based on Fraudar [1].

    References
    ----------
    [1] Hooi, Bryan, et al. "Fraudar: Bounding graph fraud in the face of
    camouflage." Proceedings of the 22nd ACM SIGKDD international conference on
    knowledge discovery and data mining. 2016.
    """

    def __init__(self, values: npt.ArrayLike):
        """Initializer.

        Parameters
        ----------
        values
            Array values from which the tree is constructed
        """
        self.tree_height = int(np.ceil(np.log2(len(values))))
        self.n_leaves = 2**self.tree_height
        self.n_branches = self.n_leaves - 1
        self.n_nodes = self.n_branches + self.n_leaves

        self.nodes = [np.inf] * self.n_nodes
        # Leaf nodes
        for i in range(len(values)):
            self.nodes[self.n_branches + i] = values[i]

        # Parents carry the smallest value of their children
        for i in reversed(range(self.n_branches)):
            self.nodes[i] = min(self.nodes[2 * i + 1], self.nodes[2 * i + 2])

    def get_min_value(self):
        """
        Finds the leaf with the smallest value
        """
        curr = 0
        for i in range(self.tree_height):
            if self.nodes[2 * curr + 1] <= self.nodes[2 * curr + 2]:
                curr = 2 * curr + 1
            else:
                curr = 2 * curr + 2

        # return leaf index and value
        return (curr - self.n_branches, self.nodes[curr])

    def update_value(self, leaf, delta):
        """
        Update a value of leaf and its parent nodes in the tree
        """
        curr = self.n_branches + leaf
        self.nodes[curr] += delta
        for i in range(self.tree_height):
            parent = (curr - 1) // 2
            new_parent_val = min(self.nodes[2 * parent + 1], self.nodes[2 * parent + 2])

            if self.nodes[parent] == new_parent_val:
                break

            self.nodes[parent] = new_parent_val
            curr = parent


class GreedyDetector:
    """Adversary detector using the Fraudar greedy peeling algorithm [1].

    The detection is performed by first constructing a bipartite graph
    $G=(W, T, E)$ from the response matrix of a crowdsourced dataset. In $G$,
    $W$ and $T$ are the nodes representing workers and tasks and edges
    connect a worker and a task if that worker provided a label for that
    task. An edge weighting mechanism that employs worker agreement rates
    and co-labeling is also provided (`kind="weighted"`). The constructed
    bipartite graph is then peeled using the Fraudar greedy algorithm [1],
    whose removal order is used to calculate adversary scores such that
    higher scores indicate higher likelihood for a worker to be
    adversarial or a task to be targeted.

    This detector only ranks workers/tasks -- it does not estimate how
    many are actually adversarial/targeted. For that, pair a fitted
    detector with a selector from `subcad.selection` (e.g.
    `DensitySelector` or `SpectralSeededSelector`), passing
    `biadj_mat_`/`worker_scores_`/`task_scores_`.

    !!! Example
        ```python
        from subcad import GreedyDetector

        detector = GreedyDetector(kind="weighted")
        worker_scores, task_scores = detector.fit_predict(response_mat)
        ```

    Parameters
    ----------
    kind
        Kind of bipartite graph to construct. Must be either "binary" or
        "weighted". In the latter case, the edges of the bipartite graph
        are weighted as described in [1].

    Attributes
    ----------
    biadj_mat_ : npt.NDArray
        $(M, N)$ dimensional bi-adjacency matrix of the worker-task
        bipartite graph constructed from the response matrix. Set after
        calling `fit`.
    worker_scores_ : npt.NDArray
        $(M, )$ dimensional array where `worker_scores_[i]` is the
        adversary score of $i$th worker indicating the likelihood of $i$
        being an adversary. Set after calling `fit`.
    task_scores_ : npt.NDArray
        $(N, )$ dimensional array where `task_scores_[i]` is the adversary
        score of $i$th task indicating the likelihood of $i$ being a
        targeted task. Set after calling `fit`.

    References
    ----------
    [1] Hooi, Bryan, et al. "Fraudar: Bounding graph fraud in the face of
    camouflage." Proceedings of the 22nd ACM SIGKDD international conference on
    knowledge discovery and data mining. 2016.
    """

    def __init__(self, kind: str = "binary"):
        self.kind = kind

    def fit(self, response_mat: npt.NDArray, y=None) -> "GreedyDetector":
        """Detect adversarial workers and their targeted tasks.

        Parameters
        ----------
        response_mat
            $(M, N)$ dimensional matrix where `response_mat[i, j]` is the
            label provided by $i$th worker for $j$th task.
            `response_mat[i, j] = 0` is assumed to indicate no label is
            given by $i$th worker for $j$th task.
        y
            Ignored. Present for API consistency.

        Returns
        -------
        self
        """
        biadj_mat = _construct_biadj_mat(response_mat, self.kind)
        workers_order, tasks_order = self._peel(biadj_mat)
        worker_scores, task_scores = _calc_adversary_scores(workers_order, tasks_order)

        self.biadj_mat_ = biadj_mat
        self.worker_scores_ = worker_scores
        self.task_scores_ = task_scores

        return self

    def fit_predict(
        self, response_mat: npt.NDArray, y=None
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Fit the detector and return worker/task adversary scores.

        Equivalent to calling `fit` followed by reading `worker_scores_`
        and `task_scores_`.

        Parameters
        ----------
        response_mat
            See `fit`.
        y
            Ignored. Present for API consistency.

        Returns
        -------
        worker_scores : npt.NDArray
            See `worker_scores_`.
        task_scores : npt.NDArray
            See `task_scores_`.
        """
        self.fit(response_mat)
        return self.worker_scores_, self.task_scores_

    def _peel(self, biadj_mat: npt.NDArray):
        """
        Run peeling algorithm on a bipartite graph. Implementation is based
        on Fraudar [1].

        References
        ----------
        [1] Hooi, Bryan, et al. "Fraudar: Bounding graph fraud in the face of
        camouflage." Proceedings of the 22nd ACM SIGKDD international
        conference on knowledge discovery and data mining. 2016.
        """
        n_workers, n_tasks = biadj_mat.shape

        worker_degrees = np.sum(biadj_mat, axis=1)
        task_degrees = np.sum(biadj_mat, axis=0)

        # Construct Minimum search trees for peeling algorithm
        workers_tree = MinTree(worker_degrees)
        tasks_tree = MinTree(task_degrees)

        # Peeling algorithm - inputs
        biadj_lil = sparse.lil_array(biadj_mat)
        biadj_lil_t = sparse.lil_array(biadj_mat.T)
        workers = set(range(0, n_workers))
        tasks = set(range(0, n_tasks))

        # Peeling algorithm - outputs
        workers_order = []
        tasks_order = []

        while workers and tasks:
            # Find the node with the minimum degree in the bipartite graph
            min_worker, min_worker_degree = workers_tree.get_min_value()
            min_task, min_task_degree = tasks_tree.get_min_value()

            if min_worker_degree <= min_task_degree:  # Peel the worker
                for task in biadj_lil.rows[min_worker]:
                    change = biadj_mat[min_worker, task]
                    tasks_tree.update_value(task, -change)

                workers.remove(min_worker)
                workers_tree.update_value(min_worker, np.inf)
                workers_order.append(min_worker)

            else:  # Peel the task
                for worker in biadj_lil_t.rows[min_task]:
                    change = biadj_mat[worker, min_task]
                    workers_tree.update_value(worker, -change)

                tasks.remove(min_task)
                tasks_tree.update_value(min_task, np.inf)
                tasks_order.append(min_task)

        # Add the last remaining node to order arrays
        if len(workers) == 0:
            tasks_order.append(list(tasks)[0])
        if len(tasks) == 0:
            workers_order.append(list(workers)[0])

        return workers_order, tasks_order
