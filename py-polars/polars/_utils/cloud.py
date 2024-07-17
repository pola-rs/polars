from __future__ import annotations

from typing import TYPE_CHECKING

import polars.polars as plr

if TYPE_CHECKING:
    from polars import LazyFrame


def assert_cloud_eligible(lf: LazyFrame) -> None:
    """
    Assert that the given LazyFrame is eligible to be executed on Polars Cloud.

    The following conditions will disqualify a LazyFrame from being eligible:

    - Contains a user-defined function
    - Scans a local filesystem

    Parameters
    ----------
    lf
        The LazyFrame to check.

    Raises
    ------
    AssertionError
        If the given LazyFrame is not eligible to be run on Polars Cloud.
    """
    plr.assert_cloud_eligible(lf._ldf)
