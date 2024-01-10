from __future__ import annotations

import contextlib

with contextlib.suppress(ImportError):  # Module not available when building docs
    import polars.polars as plr


def set_random_seed(seed: int) -> None:
    r"""
    Set the global random seed for Polars.

    This random seed is used to determine things such as shuffle ordering.


    Parameters
    ----------
    seed
        A non-negative integer < 2\ :sup:`64` used to seed the internal global
        random number generator.

    Examples
    --------
    >>> pl.set_random_seed(0)
    >>> x = pl.Series([1, 2, 3]).shuffle()
    >>> pl.set_random_seed(0)
    >>> y = pl.Series([1, 2, 3]).shuffle()
    >>> assert x.equals(y)
    """
    plr.set_random_seed(seed)
