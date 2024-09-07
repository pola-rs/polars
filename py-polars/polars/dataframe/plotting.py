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


def configure_chart(
    chart: alt.Chart,
    *,
    title: str | None,
    x_axis_title: str | None,
    y_axis_title: str | None,
) -> alt.Chart:
    """
    A nice-looking default configuration, produced by Altair maintainer.

    Source: https://gist.github.com/binste/b4042fa76a89d72d45cbbb9355ec6906.
    """
    properties = {}
    if title is not None:
        properties["title"] = title
    if x_axis_title is not None:
        chart.encoding.x.title = x_axis_title
    if y_axis_title is not None:
        chart.encoding.y.title = y_axis_title
    return (
        chart.properties(**properties)
        .configure_axis(
            labelFontSize=16,
            titleFontSize=16,
            titleFontWeight="normal",
            gridColor="lightGray",
            labelAngle=0,
            labelFlush=False,
            labelPadding=5,
        )
        .configure_axisY(
            domain=False,
            ticks=False,
            labelPadding=10,
            titleAngle=0,
            titleY=-20,
            titleAlign="left",
            titlePadding=0,
        )
        .configure_axisTemporal(grid=False)
        .configure_axisDiscrete(ticks=False, labelPadding=10, grid=False)
        .configure_scale(barBandPaddingInner=0.2)
        .configure_header(labelFontSize=16, titleFontSize=16)
        .configure_legend(labelFontSize=16, titleFontSize=16, titleFontWeight="normal")
        .configure_title(
            fontSize=20,
            fontStyle="normal",
            align="left",
            anchor="start",
            orient="top",
            fontWeight=600,
            offset=10,
            subtitlePadding=3,
            subtitleFontSize=16,
        )
        .configure_view(
            strokeWidth=0, continuousHeight=350, continuousWidth=600, step=50
        )
        .configure_line(strokeWidth=3.5)
        .configure_text(fontSize=16)
        .configure_circle(size=60)
        .configure_point(size=60)
        .configure_square(size=60)
        .interactive()
    )


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
        title: str | None = None,
        x_axis_title: str | None = None,
        y_axis_title: str | None = None,
        **kwargs: Unpack[EncodeKwds],
    ) -> alt.Chart:
        """
        Draw bar plot.

        Polars defers to `Altair <https://altair-viz.github.io/>`_ for plotting, and
        this functionality is only provided for convenience.
        For configuration, we suggest reading `Chart Configuration
        <https://altair-viz.github.io/altair-tutorial/notebooks/08-Configuration.html>`_.

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
        title
            Plot title.
        x_axis_title
            Title of x-axis.
        y_axis_title
            Title of y-axis.
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
        return configure_chart(
            self._chart.mark_bar().encode(**encodings, **kwargs),
            title=title,
            x_axis_title=x_axis_title,
            y_axis_title=y_axis_title,
        )

    def line(
        self,
        x: ChannelX | None = None,
        y: ChannelY | None = None,
        color: ChannelColor | None = None,
        order: ChannelOrder | None = None,
        tooltip: ChannelTooltip | None = None,
        /,
        title: str | None = None,
        x_axis_title: str | None = None,
        y_axis_title: str | None = None,
        **kwargs: Unpack[EncodeKwds],
    ) -> alt.Chart:
        """
        Draw line plot.

        Polars defers to `Altair <https://altair-viz.github.io/>`_ for plotting, and
        this functionality is only provided for convenience.
        For configuration, we suggest reading `Chart Configuration
        <https://altair-viz.github.io/altair-tutorial/notebooks/08-Configuration.html>`_.

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
        title
            Plot title.
        x_axis_title
            Title of x-axis.
        y_axis_title
            Title of y-axis.
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
        return configure_chart(
            self._chart.mark_line().encode(**encodings, **kwargs),
            title=title,
            x_axis_title=x_axis_title,
            y_axis_title=y_axis_title,
        )

    def point(
        self,
        x: ChannelX | None = None,
        y: ChannelY | None = None,
        color: ChannelColor | None = None,
        size: ChannelSize | None = None,
        tooltip: ChannelTooltip | None = None,
        /,
        title: str | None = None,
        x_axis_title: str | None = None,
        y_axis_title: str | None = None,
        **kwargs: Unpack[EncodeKwds],
    ) -> alt.Chart:
        """
        Draw scatter plot.

        Polars defers to `Altair <https://altair-viz.github.io/>`_ for plotting, and
        this functionality is only provided for convenience.
        For configuration, we suggest reading `Chart Configuration
        <https://altair-viz.github.io/altair-tutorial/notebooks/08-Configuration.html>`_.

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
        title
            Plot title.
        x_axis_title
            Title of x-axis.
        y_axis_title
            Title of y-axis.
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
        return configure_chart(
            self._chart.mark_point().encode(
                **encodings,
                **kwargs,
            ),
            title=title,
            x_axis_title=x_axis_title,
            y_axis_title=y_axis_title,
        )

    # Alias to `point` because of how common it is.
    scatter = point

    def __getattr__(
        self,
        attr: str,
        *,
        title: str | None = None,
        x_axis_title: str | None = None,
        y_axis_title: str | None = None,
    ) -> Callable[..., alt.Chart]:
        method = getattr(self._chart, f"mark_{attr}", None)
        if method is None:
            msg = "Altair has no method 'mark_{attr}'"
            raise AttributeError(msg)
        return lambda **kwargs: configure_chart(
            method().encode(**kwargs),
            title=title,
            x_axis_title=x_axis_title,
            y_axis_title=y_axis_title,
        )
