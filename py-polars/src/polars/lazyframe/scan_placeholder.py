from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polars import LazyFrame
    from polars._typing import SchemaDict

with contextlib.suppress(ImportError):
    import polars._plr as plr


def scan_placeholder(
    name: str,
    schema: SchemaDict,
) -> LazyFrame:
    """
    Create a placeholder LazyFrame with a given name and schema.

    This creates a reusable computation graph template. The placeholder must be
    bound to a concrete data source via :meth:`LazyFrame.bind` before collecting.

    Parameters
    ----------
    name
        Name of the placeholder. Used as a key when binding.
    schema
        Schema of the placeholder as a dictionary of column names to dtypes.

    Returns
    -------
    LazyFrame
        A LazyFrame containing a PlaceholderScan node.

    See Also
    --------
    LazyFrame.bind : Bind placeholders to concrete LazyFrames.
    LazyFrame.optimize_template : Optimize the template for repeated use.

    Examples
    --------
    >>> template = pl.scan_placeholder(
    ...     "input",
    ...     {"a": pl.Int64, "b": pl.String},
    ... ).filter(pl.col("a") > 0)
    >>> df = pl.DataFrame({"a": [1, -2, 3], "b": ["x", "y", "z"]})
    >>> result = template.bind({"input": df.lazy()}).collect()
    """
    from polars.lazyframe.frame import LazyFrame

    lf = LazyFrame.__new__(LazyFrame)
    lf._ldf = plr.PyLazyFrame.placeholder(name, schema)
    return lf
