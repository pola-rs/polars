from __future__ import annotations

from typing import TYPE_CHECKING

from polars._utils.wrap import wrap_df

if TYPE_CHECKING:
    from polars import DataFrame, LazyFrame
    from polars._plr import PyOptimizedTemplate


class OptimizedTemplate:
    """A pre-optimized query plan template for repeated bind+collect.

    Created by calling :meth:`LazyFrame.optimize_template` on a LazyFrame
    containing ``PlaceholderScan`` nodes (created via :func:`scan_placeholder`).
    The plan is optimized once at creation time and can be bound to different
    data repeatedly without re-optimization.

    .. warning::
        Designed for **in-memory data sources**. When binding file-based scans
        (``scan_parquet``, ``scan_csv``, etc.), scan-level optimizations
        (predicate pushdown into the reader, Hive partition pruning, slice
        pushdown) are skipped. For file-based sources, use
        :meth:`LazyFrame.bind` + :meth:`LazyFrame.collect` instead.
    """

    _opt_template: PyOptimizedTemplate

    def __init__(self) -> None:
        msg = (
            "OptimizedTemplate cannot be instantiated directly. "
            "Use LazyFrame.optimize_template() instead."
        )
        raise TypeError(msg)

    @classmethod
    def _from_pyot(cls, pyot: PyOptimizedTemplate) -> OptimizedTemplate:
        self = cls.__new__(cls)
        self._opt_template = pyot
        return self

    def bind_and_collect(self, bindings: dict[str, LazyFrame]) -> DataFrame:
        """Bind concrete LazyFrames and collect immediately.

        This is the fast path: replaces placeholders in the already-optimized
        IR and goes straight to execution, skipping re-optimization.

        .. warning::
            Scan-level optimizations (predicate pushdown into readers, Hive
            partition pruning, slice pushdown) are not applied to file-based
            bindings. For file-based sources, use :meth:`LazyFrame.bind` +
            :meth:`LazyFrame.collect` instead.

        Parameters
        ----------
        bindings
            A dictionary mapping placeholder names to concrete LazyFrames.

        Returns
        -------
        DataFrame
            The collected result.
        """
        rust_bindings = {k: v._ldf for k, v in bindings.items()}
        return wrap_df(self._opt_template.bind_and_collect(rust_bindings))

    def bind(self, bindings: dict[str, LazyFrame]) -> LazyFrame:
        """Bind concrete LazyFrames to placeholders, returning a LazyFrame.

        Parameters
        ----------
        bindings
            A dictionary mapping placeholder names to concrete LazyFrames.

        Returns
        -------
        LazyFrame
            A new LazyFrame with all placeholders replaced.
        """
        from polars.lazyframe.frame import LazyFrame

        rust_bindings = {k: v._ldf for k, v in bindings.items()}
        result_ldf = self._opt_template.bind(rust_bindings)
        return LazyFrame._from_pyldf(result_ldf)

    def placeholder_names(self) -> list[str]:
        """Get the names of all placeholders in this template.

        Returns
        -------
        list[str]
            List of placeholder names.
        """
        return self._opt_template.placeholder_names()
