from __future__ import annotations

from typing import TYPE_CHECKING

import polars.polars as plr
from polars._utils.various import normalize_filepath

if TYPE_CHECKING:
    from pathlib import Path

    from polars import LazyFrame


def prepare_cloud_plan(
    lf: LazyFrame,
    uri: Path | str,
    **optimizations: bool,
) -> bytes:
    """
    Prepare the given LazyFrame for execution on Polars Cloud.

    Parameters
    ----------
    lf
        The LazyFrame to prepare.
    uri
        Path to which the file should be written.
        Must be a URI to an accessible object store location.
    **optimizations
        Optimizations to enable or disable in the query optimizer, e.g.
        `projection_pushdown=False`.

    Raises
    ------
    InvalidOperationError
        If the given LazyFrame is not eligible to be run on Polars Cloud.
        The following conditions will disqualify a LazyFrame from being eligible:

        - Contains a user-defined function
        - Scans or sinks to a local filesystem
    ComputeError
        If the given LazyFrame cannot be serialized.
    """
    uri = normalize_filepath(uri)
    pylf = lf._set_sink_optimizations(**optimizations)
    return plr.prepare_cloud_plan(pylf, uri)
