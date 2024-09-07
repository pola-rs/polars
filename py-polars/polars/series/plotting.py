from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from polars.dataframe.plotting import configure_chart
from polars.dependencies import altair as alt

if TYPE_CHECKING:
    import sys

    from altair.typing import EncodeKwds

    if sys.version_info >= (3, 11):
        from typing import Unpack
    else:
        from typing_extensions import Unpack

    from polars import Series


class SeriesPlot:
    """Series.plot namespace."""

    _accessor = "plot"

    def __init__(self, s: Series) -> None:
        name = s.name or "value"
        self._df = s.to_frame(name)
        self._series_name = name

    def hist(
        self,
        /,
        title: str | None = None,
        x_axis_title: str | None = None,
        y_axis_title: str | None = None,
        **kwargs: Unpack[EncodeKwds],
    ) -> alt.Chart:
        """
        Draw histogram.

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
        title
            Plot title.
        x_axis_title
            Title of x-axis.
        y_axis_title
            Title of y-axis.
        **kwargs
            Additional arguments and keyword arguments passed to Altair.

        Examples
        --------
        >>> s = pl.Series("price", [1, 3, 3, 3, 5, 2, 6, 5, 5, 5, 7])
        >>> s.plot.hist()  # doctest: +SKIP
        """
        if self._series_name == "count()":
            msg = "Cannot use `plot.hist` when Series name is `'count()'`"
            raise ValueError(msg)
        return configure_chart(
            alt.Chart(self._df)
            .mark_bar()
            .encode(x=alt.X(f"{self._series_name}:Q", bin=True), y="count()", **kwargs),  # type: ignore[misc]
            title=title,
            x_axis_title=x_axis_title,
            y_axis_title=y_axis_title,
        )

    def kde(
        self,
        /,
        title: str | None = None,
        x_axis_title: str | None = None,
        y_axis_title: str | None = None,
        **kwargs: Unpack[EncodeKwds],
    ) -> alt.Chart:
        """
        Draw kernel density estimate plot.

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
        >>> s = pl.Series("price", [1, 3, 3, 3, 5, 2, 6, 5, 5, 5, 7])
        >>> s.plot.kde()  # doctest: +SKIP
        """
        if self._series_name == "density":
            msg = "Cannot use `plot.kde` when Series name is `'density'`"
            raise ValueError(msg)
        return configure_chart(
            alt.Chart(self._df)
            .transform_density(self._series_name, as_=[self._series_name, "density"])
            .mark_area()
            .encode(x=self._series_name, y="density:Q", **kwargs),  # type: ignore[misc]
            title=title,
            x_axis_title=x_axis_title,
            y_axis_title=y_axis_title,
        )

    def line(
        self,
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
        >>> s = pl.Series("price", [1, 3, 3, 3, 5, 2, 6, 5, 5, 5, 7])
        >>> s.plot.kde()  # doctest: +SKIP
        """
        if self._series_name == "index":
            msg = "Cannot call `plot.line` when Series name is 'index'"
            raise ValueError(msg)
        return configure_chart(
            alt.Chart(self._df.with_row_index())
            .mark_line()
            .encode(x="index", y=self._series_name, **kwargs),  # type: ignore[misc]
            title=title,
            x_axis_title=x_axis_title,
            y_axis_title=y_axis_title,
        )

    def __getattr__(
        self,
        attr: str,
        *,
        title: str | None = None,
        x_axis_title: str | None = None,
        y_axis_title: str | None = None,
    ) -> Callable[..., alt.Chart]:
        if self._series_name == "index":
            msg = "Cannot call `plot.{attr}` when Series name is 'index'"
            raise ValueError(msg)
        if attr == "scatter":
            # alias `scatter` to `point` because of how common it is
            attr = "point"
        method = getattr(alt.Chart(self._df.with_row_index()), f"mark_{attr}", None)
        if method is None:
            msg = "Altair has no method 'mark_{attr}'"
            raise AttributeError(msg)
        return lambda **kwargs: configure_chart(
            method().encode(x="index", y=self._series_name, **kwargs),
            title=title,
            x_axis_title=x_axis_title,
            y_axis_title=y_axis_title,
        )
