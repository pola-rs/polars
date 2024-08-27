from __future__ import annotations

from typing import TYPE_CHECKING, Callable

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
        **kwargs: Unpack[EncodeKwds],
    ) -> alt.Chart:
        """
        Draw histogram.

        Polars does not implement plotting logic itself but instead defers to
        `Altair <https://altair-viz.github.io/>`_.

        `s.plot.hist(**kwargs)` is shorthand for
        `alt.Chart(s.to_frame()).mark_bar().encode(x=alt.X(f'{s.name}:Q', bin=True), y='count()', **kwargs).interactive()`,
        and is provided for convenience - for full customisatibility, use a plotting
        library directly.

        .. versionchanged:: 1.6.0
            In prior versions of Polars, HvPlot was the plotting backend. If you would
            like to restore the previous plotting functionality, all you need to do
            is add `import hvplot.polars` at the top of your script and replace
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
        and is provided for convenience - for full customisatibility, use a plotting
        library directly.

        .. versionchanged:: 1.6.0
            In prior versions of Polars, HvPlot was the plotting backend. If you would
            like to restore the previous plotting functionality, all you need to do
            is add `import hvplot.polars` at the top of your script and replace
            `df.plot` with `df.hvplot`.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to Altair.

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

    def line(
        self,
        /,
        **kwargs: Unpack[EncodeKwds],
    ) -> alt.Chart:
        """
        Draw line plot.

        Polars does not implement plotting logic itself but instead defers to
        `Altair <https://altair-viz.github.io/>`_.

        `s.plot.line(**kwargs)` is shorthand for
        `alt.Chart(s.to_frame().with_row_index()).mark_line().encode(x='index', y=s.name, **kwargs).interactive()`,
        and is provided for convenience - for full customisatibility, use a plotting
        library directly.

        .. versionchanged:: 1.6.0
            In prior versions of Polars, HvPlot was the plotting backend. If you would
            like to restore the previous plotting functionality, all you need to do
            is add `import hvplot.polars` at the top of your script and replace
            `df.plot` with `df.hvplot`.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to Altair.

        Examples
        --------
        >>> s = pl.Series("price", [1, 3, 3, 3, 5, 2, 6, 5, 5, 5, 7])
        >>> s.plot.kde()  # doctest: +SKIP
        """  # noqa: W505
        if self._series_name == "index":
            msg = "Cannot call `plot.line` when Series name is 'index'"
            raise ValueError(msg)
        return (
            alt.Chart(self._df.with_row_index())
            .mark_line()
            .encode(x="index", y=self._series_name, **kwargs)  # type: ignore[misc]
            .interactive()
        )

    def __getattr__(self, attr: str) -> Callable[..., alt.Chart]:
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
        return (
            lambda **kwargs: method()
            .encode(x="index", y=self._series_name, **kwargs)
            .interactive()
        )
