from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import altair as alt

    from polars import DataFrame


class Plot:
    """DataFrame.plot namespace."""

    chart: alt.Chart

    def __init__(self, df: DataFrame) -> None:
        import altair as alt

        self.chart = alt.Chart(df)

    def line(
        self,
        x: str | Any|None=None,
        y: str | Any|None=None,
        color: str | Any|None=None,
        order: str | Any|None=None,
        tooltip: str | Any|None=None,
        *args: Any,
        **kwargs: Any,
    ) -> alt.Chart:
        """
        Draw line plot.

        Polars does not implement plottinng logic itself but instead defers to Altair.
        `df.plot.line(*args, **kwargs)` is shorthand for
        `alt.Chart(df).mark_line().encode(*args, **kwargs).interactive()`,
        as is intended for convenience - for full customisatibility, use a plotting
        library directly.

        .. versionchanged:: 1.4.0
            In prior versions of Polars, HvPlot was the plotting backend. If you would
            like to restore the previous plotting functionality, all you need to do
            add `import hvplot.polars` at the top of your script and replace
            `df.plot` with `df.hvplot`.

        Parameters
        ----------
        x
            Column with x-coordinates of lines.
        y
            Column with y-coordinates of lines.
        color
            Column to color lines by.
        order
            Column to use for order of data points in lines.
        tooltip
            Columns to show values of when hovering over points with pointer.
        *args, **kwargs
            Additional arguments and keyword arguments passed to Altair.

        Examples
        --------
        >>> from datetime import date
        >>> df = pl.DataFrame(
        ...     {
        ...         "date": [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 4)] * 2,
        ...         "price": [1, 4, 6, 1, 5, 2],
        ...         "stock": ["a", "a", "a", "b", "b", "b"],
        ...     }
        ... )
        >>> df.plot.line(x="date", y="price", color="stock")  # doctest: +SKIP
        """
        encodings = {}
        if x is not None:
            encodings["x"] = x
        if y is not None:
            encodings["y"] = y
        if color is not None:
            encodings["color"] = color
        if order is not None:
            encodings["order"] = order
        if tooltip is not None:
            encodings["tooltip"] = tooltip
        return (
            self.chart.mark_line()
            .encode(*args, **{**encodings, **kwargs})
            .interactive()
        )

    def point(
        self,
        x: str | Any |None= None,
        y: str | Any |None= None,
        color: str | Any|None = None,
        size: str | Any |None= None,
        tooltip: str | Any |None= None,
        *args: Any,
        **kwargs: Any,
    ) -> alt.Chart:
        """
        Draw scatter plot.

        Polars does not implement plottinng logic itself but instead defers to Altair.
        `df.plot.point(*args, **kwargs)` is shorthand for
        `alt.Chart(df).mark_point().encode(*args, **kwargs).interactive()`,
        as is intended for convenience - for full customisatibility, use a plotting
        library directly.

        .. versionchanged:: 1.4.0
            In prior versions of Polars, HvPlot was the plotting backend. If you would
            like to restore the previous plotting functionality, all you need to do
            add `import hvplot.polars` at the top of your script and replace
            `df.plot` with `df.hvplot`.

        Parameters
        ----------
        x
            Column with x-coordinates of points.
        y
            Column with y-coordinates of points.
        color
            Column to color points by.
        size
            Column which determines points' sizes.
        tooltip
            Columns to show values of when hovering over points with pointer.
        *args, **kwargs
            Additional arguments and keyword arguments passed to Altair.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "length": [1, 4, 6],
        ...         "width": [4, 5, 6],
        ...         "species": ["setosa", "setosa", "versicolor"],
        ...     }
        ... )
        >>> df.plot.point(x="length", y="width", color="species")  # doctest: +SKIP
        """
        encodings = {}
        if x is not None:
            encodings["x"] = x
        if y is not None:
            encodings["y"] = y
        if color is not None:
            encodings["color"] = color
        if size is not None:
            encodings["size"] = size
        if tooltip is not None:
            encodings["tooltip"] = tooltip
        return (
            self.chart.mark_point()
            .encode(*args, **{**encodings, **kwargs})
            .interactive()
        )

    def __getattr__(self, attr: str, *args: Any, **kwargs: Any) -> alt.Chart:
        method = self.chart.getattr(f"mark_{attr}", None)
        if method is None:
            msg = "Altair has no method 'mark_{attr}'"
            raise AttributeError(msg)
        return method().encode(*args, **kwargs).interactive()
