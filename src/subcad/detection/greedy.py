import numpy.typing as npt

import numpy as np

from scipy import sparse

from .base import BaseDetector


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


class GreedyDetector(BaseDetector):
    """Adversary detector using the Fraudar greedy peeling algorithm [1].

    See `BaseDetector` for the shared detection procedure, parameters, and
    fitted attributes.

    !!! Example
        ```python
        from subcad import GreedyDetector

        detector = GreedyDetector(kind="weighted")
        worker_scores, task_scores = detector.fit_predict(response_mat)
        ```

    References
    ----------
    [1] Hooi, Bryan, et al. "Fraudar: Bounding graph fraud in the face of
    camouflage." Proceedings of the 22nd ACM SIGKDD international conference on
    knowledge discovery and data mining. 2016.
    """

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
