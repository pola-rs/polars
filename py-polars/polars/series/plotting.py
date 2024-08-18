from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, Union

from polars.dependencies import altair as alt

if TYPE_CHECKING:
    import sys

    from altair.typing import (
        ChannelColor,
        ChannelOrder,
        ChannelSize,
        ChannelTooltip,
        ChannelX,
        ChannelY,
        EncodeKwds,
    )

    if sys.version_info >= (3, 10):
        from typing import TypeAlias
    else:
        from typing_extensions import TypeAlias
    if sys.version_info >= (3, 11):
        from typing import Unpack
    else:
        from typing_extensions import Unpack

    from polars import Series

    Encodings: TypeAlias = Dict[
        str,
        Union[
            ChannelX, ChannelY, ChannelColor, ChannelOrder, ChannelSize, ChannelTooltip
        ],
    ]


class Plot:
    """Series.plot namespace."""

    _accessor = "plot"

    chart: alt.Chart

    def __init__(self, s: Series) -> None:
        name = s.name or "value"
        self._df = s.to_frame(name)
        self._series_name = name

    def hist(
        self,
        /,
        **kwargs: Unpack[EncodeKwds],
    ) -> alt.Chart:
        """
        Draw histogram.

        Polars does not implement plotting logic itself but instead defers to
        `Altair <https://altair-viz.github.io/>`_.

        `s.plot.hist(**kwargs)` is shorthand for
        `alt.Chart(s.to_frame()).mark_bar().encode(x=alt.X(f'{s.name}:Q', bin=True), y='count()', **kwargs).interactive()`,
        as is intended for convenience - for full customisatibility, use a plotting
        library directly.

        .. versionchanged:: 1.6.0
            In prior versions of Polars, HvPlot was the plotting backend. If you would
            like to restore the previous plotting functionality, all you need to do
            add `import hvplot.polars` at the top of your script and replace
            `df.plot` with `df.hvplot`.

        Parameters
        ----------
        **kwargs
            Additional arguments and keyword arguments passed to Altair.

        Examples
        --------
        >>> s = pl.Series("price", [1, 3, 3, 3, 5, 2, 6, 5, 5, 5, 7])
        >>> s.plot.hist()  # doctest: +SKIP
        """  # noqa: W505
        if self._series_name == "count()":
            msg = "Cannot use `plot.hist` when Series name is `'count()'`"
            raise ValueError(msg)
        return (
            alt.Chart(self._df)
            .mark_bar()
            .encode(x=alt.X(f"{self._series_name}:Q", bin=True), y="count()", **kwargs)  # type: ignore[misc]
            .interactive()
        )

    def kde(
        self,
        /,
        **kwargs: Unpack[EncodeKwds],
    ) -> alt.Chart:
        """
        Draw kernel dentity estimate plot.

        Polars does not implement plotting logic itself but instead defers to
        `Altair <https://altair-viz.github.io/>`_.

        `s.plot.kde(**kwargs)` is shorthand for
        `alt.Chart(s.to_frame()).transform_density(s.name, as_=[s.name, 'density']).mark_area().encode(x=s.name, y='density:Q', **kwargs).interactive()`,
        as is intended for convenience - for full customisatibility, use a plotting
        library directly.

        .. versionchanged:: 1.6.0
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
            Columns to show values of when hovering over lines with pointer.
        *args, **kwargs
            Additional arguments and keyword arguments passed to Altair.

        Examples
        --------
        >>> s = pl.Series("price", [1, 3, 3, 3, 5, 2, 6, 5, 5, 5, 7])
        >>> s.plot.kde()  # doctest: +SKIP
        """  # noqa: W505
        if self._series_name == "density":
            msg = "Cannot use `plot.kde` when Series name is `'density'`"
            raise ValueError(msg)
        return (
            alt.Chart(self._df)
            .transform_density(self._series_name, as_=[self._series_name, "density"])
            .mark_area()
            .encode(x=self._series_name, y="density:Q", **kwargs)  # type: ignore[misc]
            .interactive()
        )

    def __getattr__(self, attr: str) -> Callable[..., alt.Chart]:
        if "index" in self._df.columns:
            msg = "Cannot call `plot.{attr}` when Series name is 'index'"
            raise ValueError(msg)
        method = getattr(
            alt.Chart(self._df.with_row_index("index")), f"mark_{attr}", None
        )
        if method is None:
            msg = "Altair has no method 'mark_{attr}'"
            raise AttributeError(msg)
        return (
            lambda **kwargs: method()
            .encode(x="index", y=self._series_name, **kwargs)
            .interactive()
        )
