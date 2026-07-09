import numpy as np

from .typing import RNGType


def check_rng(rng: RNGType) -> np.random.Generator:
    """Checks if a given input for random number generator is valid.

    A valid `rng` input can be either an `int` indicating a seed number, a
    `np.random.Generator` object, or `None`.

    Parameters
    ----------
    rng
        See `subcad.typing.RNGType`.

    Returns
    -------
    np.random.Generator
        `rng` itself if it is already a `np.random.Generator`, otherwise a
        new `np.random.Generator` seeded with `rng` (or unseeded if `rng`
        is `None`).
    """

    if rng is None:
        rng = np.random.default_rng()
    elif isinstance(rng, int):
        rng = np.random.default_rng(rng)

    return rng
