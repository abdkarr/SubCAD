import numpy as np
import numpy.typing as npt

from ...utils import check_rng


def make_confusion_matrix(
    n_classes: int, reliability: float, random_state=None
) -> npt.NDArray:
    r"""Generate a confusion matrix for a simulated crowdsourcing worker.

    The function generates a $K \times K$ dimensional matric ${\bf \Gamma}$
    where $K$ is the number of classes and $\Gamma_{ij}$ is the probability
    of worker giving label $i$ for a task in class $j$. $k$th column of
    $\bf \Gammsa is drawn from a Dirichlet distribution with parameter
    ${\bf \alpha} = {\bf 1} + (r - 1){\bf e}_k$ where ${\bf e}_k$ is the $k$th
    standard basis of $K$-dimensional space and $r$ is reliability. As $r$ grows,
    the worker becomes more reliable in limit sense. That is
    $\mathbb{E}[\Gamma_{ii}]/\mathbb{E}[\Gamma_{ij}] = r,\ \forall j \neq i$.

    Parameters
    ----------
    n_classes
        Number of classes to simulate.
    reliability
        Realibility of worker.
    random_state
        Controls the randomness used to draw the columns of the confusion
        matrix. Pass an `int` for reproducible results across calls, or a
        `np.random.Generator`, see `subcad.typing.RNGType`.

    Returns
    -------
    confusion_mat : npt.NDArray
        Generated confusion matrix.

    Raises
    ------
    ValueError
        If `relaibility` is not positive.

    Examples
    --------
    ??? Example
        The following code generates a confusion matrix for a simulated crowdsourcing
        problem with 5 classes:

        ```python
        import subcad

        n_classes = 5
        reliability = 2
        confusion_mat = subcad.data.make_confusion_matrix(n_classes, reliability)
        ```
    """

    rng = check_rng(random_state)

    # Check if reliability is valid
    if reliability <= 0:
        raise ValueError("Parameter `reliability` must be larger than 0.")

    confusion_mat = np.zeros((n_classes, n_classes))
    for k in range(n_classes):
        alphas = np.ones(n_classes)
        alphas[k] *= reliability
        confusion_mat[:, k] = rng.dirichlet(alphas)

    return confusion_mat
