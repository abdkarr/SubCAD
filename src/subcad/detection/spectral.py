import numpy.typing as npt

import numpy as np

from .base import BaseDetector


class SpectralDetector(BaseDetector):
    """Adversary detector using the leading singular vectors of the
    bipartite graph's biadjacency matrix.

    See `BaseDetector` for the shared detection procedure, parameters, and
    fitted attributes. Unlike `GreedyDetector`/`GreedyPPDetector`, which
    rank nodes by Fraudar-style peeling order, this detector ranks
    workers/tasks by the magnitude of their loading in the top left/right
    singular vector of the (optionally agreement-weighted) biadjacency
    matrix: workers/tasks that load heavily on the dominant singular
    direction are more likely to be part of a coordinated
    adversarial/targeted block.

    Ranks nodes about as well as peeling order (comparable AUROC), via an
    entirely different mechanism (spectral rather than combinatorial
    peeling), which makes it a natural candidate for ensembling alongside
    `GreedyDetector`/`GreedyPPDetector` rather than a replacement for them.
    Pair with a selector from `subcad.selection` (e.g.
    `SpectralSeededSelector`) for a size estimate.

    !!! Example
        ```python
        from subcad.detection import SpectralDetector

        detector = SpectralDetector(kind="weighted")
        worker_scores, task_scores = detector.fit_predict(response_mat)
        ```
    """

    def _peel(self, biadj_mat: npt.NDArray):
        """
        Rank workers/tasks by their loading magnitude in the top left/right
        singular vector of `biadj_mat`.
        """
        u, _, vt = np.linalg.svd(biadj_mat, full_matrices=False)
        worker_loadings = np.abs(u[:, 0])
        task_loadings = np.abs(vt[0, :])

        # Ascending loading -> matches the peeling detectors' convention
        # that workers_order/tasks_order run from least to most suspicious.
        workers_order = np.argsort(worker_loadings)
        tasks_order = np.argsort(task_loadings)

        return workers_order, tasks_order
