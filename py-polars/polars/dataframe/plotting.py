from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, Union

if TYPE_CHECKING:
    import sys

    import altair as alt
    from altair.typing import (
        ChannelColor,
        ChannelOrder,
        ChannelSize,
        ChannelTooltip,
        ChannelX,
        ChannelY,
        EncodeKwds,
    )

    from polars import DataFrame

    if sys.version_info >= (3, 10):
        from typing import TypeAlias
    else:
        from typing_extensions import TypeAlias
    if sys.version_info >= (3, 11):
        from typing import Unpack
    else:
        from typing_extensions import Unpack

    Encodings: TypeAlias = Dict[
        str,
        Union[
            ChannelX, ChannelY, ChannelColor, ChannelOrder, ChannelSize, ChannelTooltip
        ],
    ]


class DataFramePlot:
    """DataFrame.plot namespace."""

    def __init__(self, df: DataFrame) -> None:
        import altair as alt

        self._chart = alt.Chart(df)

    def bar(
        self,
        x: ChannelX | None = None,
        y: ChannelY | None = None,
        color: ChannelColor | None = None,
        tooltip: ChannelTooltip | None = None,
        /,
        **kwargs: Unpack[EncodeKwds],
    ) -> alt.Chart:
        """
        Draw bar plot.

        Polars does not implement plotting logic itself but instead defers to
        `Altair <https://altair-viz.github.io/>`_.

        `df.plot.bar(**kwargs)` is shorthand for
        `alt.Chart(df).mark_bar().encode(**kwargs).interactive()`,
        and is provided for convenience - for full customisatibility, use a plotting
        library directly.

        .. versionchanged:: 1.6.0
            In prior versions of Polars, HvPlot was the plotting backend. If you would
            like to restore the previous plotting functionality, all you need to do
            is add `import hvplot.polars` at the top of your script and replace
            `df.plot` with `df.hvplot`.

        Parameters
        ----------
        x
            Column with x-coordinates of bars.
        y
            Column with y-coordinates of bars.
        color
            Column to color bars by.
        tooltip
            Columns to show values of when hovering over bars with pointer.
        **kwargs
            Additional keyword arguments passed to Altair.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "day": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"] * 2,
        ...         "group": ["a"] * 7 + ["b"] * 7,
        ...         "value": [1, 3, 2, 4, 5, 6, 1, 1, 3, 2, 4, 5, 1, 2],
        ...     }
        ... )
        >>> df.plot.bar(
        ...     x="day", y="value", color="day", column="group"
        ... )  # doctest: +SKIP
        """
        encodings: Encodings = {}
        if x is not None:
            encodings["x"] = x
        if y is not None:
            encodings["y"] = y
        if color is not None:
            encodings["color"] = color
        if tooltip is not None:
            encodings["tooltip"] = tooltip
        return self._chart.mark_bar().encode(**encodings, **kwargs).interactive()

    def line(
        self,
        x: ChannelX | None = None,
        y: ChannelY | None = None,
        color: ChannelColor | None = None,
        order: ChannelOrder | None = None,
        tooltip: ChannelTooltip | None = None,
        /,
        **kwargs: Unpack[EncodeKwds],
    ) -> alt.Chart:
        """
        Draw line plot.

        Polars does not implement plotting logic itself but instead defers to
        `Altair <https://altair-viz.github.io/>`_.

        `alt.Chart(df).mark_line().encode(**kwargs).interactive()`,
        and is provided for convenience - for full customisatibility, use a plotting
        library directly.

        .. versionchanged:: 1.6.0
            In prior versions of Polars, HvPlot was the plotting backend. If you would
            like to restore the previous plotting functionality, all you need to do
            is add `import hvplot.polars` at the top of your script and replace
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
            Columns to show values of when hovering over lines with pointer.
        **kwargs
            Additional keyword arguments passed to Altair.

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
        encodings: Encodings = {}
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
        return self._chart.mark_line().encode(**encodings, **kwargs).interactive()

    def point(
        self,
        x: ChannelX | None = None,
        y: ChannelY | None = None,
        color: ChannelColor | None = None,
        size: ChannelSize | None = None,
        tooltip: ChannelTooltip | None = None,
        /,
        **kwargs: Unpack[EncodeKwds],
    ) -> alt.Chart:
        """
        Draw scatter plot.

        Polars does not implement plotting logic itself but instead defers to
        `Altair <https://altair-viz.github.io/>`_.

        `df.plot.point(**kwargs)` is shorthand for
        `alt.Chart(df).mark_point().encode(**kwargs).interactive()`,
        and is provided for convenience - for full customisatibility, use a plotting
        library directly.

        .. versionchanged:: 1.6.0
            In prior versions of Polars, HvPlot was the plotting backend. If you would
            like to restore the previous plotting functionality, all you need to do
            is add `import hvplot.polars` at the top of your script and replace
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
        **kwargs
            Additional keyword arguments passed to Altair.

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
        encodings: Encodings = {}
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
            self._chart.mark_point()
            .encode(
                **encodings,
                **kwargs,
            )
            .interactive()
        )

    # Alias to `point` because of how common it is.
    scatter = point

    def __getattr__(self, attr: str) -> Callable[..., alt.Chart]:
        method = getattr(self._chart, f"mark_{attr}", None)
        if method is None:
            msg = "Altair has no method 'mark_{attr}'"
            raise AttributeError(msg)
        return lambda **kwargs: method().encode(**kwargs).interactive()
