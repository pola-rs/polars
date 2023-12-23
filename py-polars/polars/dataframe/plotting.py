from __future__ import annotations

import importlib
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from polars import DataFrame


def _get_backend(backend: str | None) -> Any:
    if backend is None:
        backend = os.environ.get("POLARS_PLOTTING_BACKEND", None)
        if backend is None:
            raise TypeError(
                "Please specify a plotting backend using the argument `backend`"
            )
    module = importlib.import_module(backend)
    if hasattr(module, "plot"):
        return module
    raise ValueError(
        f"Backend {backend} does not have a plot function. \n"
        "Please check https://github.com/data-apis/dataframe-plotting-api "
        "for more information on how to implement a plotting backend."
    )


class PlotNameSpace:
    """
    DataFrame.plot namespace.

    Polars doesn't implementing plotting logic itself. Instead, it defer to plotting
    backends which implement the `Dataframe Plotting API <https://github.com/data-apis/dataframe-plotting-api>`_.

    The idea is:

    - if you use any plotting functions or arguments as documented here, then you
      should expect consistent behaviour supported backends. Any extra keyword arguments
      you specify are passed directly to the backend itself, and so their behaviour is
      backend-dependent.
    - Polars plots are only intended as convenience methods for ease of use during
      interactive development.
    - plotting backends are developed and maintained outside of
      Polars. Backends currently known to work are:

      - plotly
      - hvplot

      but once we set out the rules, others may be on their way.

    Polars plotting methods are only intended as convenience methods for ease of
    use during interactive development. For more complex plots, it is recommended
    to use plotting libraries directly.

    """

    _accessor = "plot"

    def __init__(self, data: DataFrame) -> None:
        self._parent = data

    def _plot(self, *, kind: str = "line", backend: str | None, **kwargs: Any) -> Any:
        plot_backend = _get_backend(backend)
        return plot_backend.plot(self._parent, kind=kind, **kwargs)

    def bar(
        self,
        *,
        y: str | list[str],
        x: str | None = None,
        backend: str | None = None,
        **kwargs: Any,
    ) -> PlotNameSpace:
        """
        Produce bar plot.

        Parameters
        ----------
        x
            Column name for x axis. If not specified, then a row count with be
            used for the x axis.
        y
            Column name(s) for y axis. Can be:

            - a single column name, in which case a single line will be drawn;
            - a list of column names, in which case one line will be drawn for
              each column name in the list;
        backend
            Plotting backend to use, which must implement the
            `Dataframe Plotting API <https://github.com/data-apis/dataframe-plotting-api>`_.
            Backends currently known to be supported are:

            - 'hvplot' (requires ``hvplot>=0.9.1`` package)
            - 'plotly' (requires ``plotly>=5.16.0`` package)

            .. note::
                Plotting backends are developed and maintained by third parties,
                independently of Polars.
        kwargs
            Additional keyword arguments to pass to the plotting backend.
            Any keyword argument supported by the plotting backend can be passed -
            how it works is entirely backend-dependent.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 2, 5], "c": [7, 8, 10]})
        >>> df.plot.bar(x="a", y=["b", "c"], backend="hvplot")  # doctest: +SKIP
        """
        return self._plot(kind="bar", x=x, y=y, backend=backend, **kwargs)

    def line(
        self,
        *,
        y: str | list[str],
        x: str | None = None,
        backend: str | None = None,
        **kwargs: Any,
    ) -> PlotNameSpace:
        """
        Produce line plot.

        Parameters
        ----------
        x
            Column name for x axis. If not specified, then a row count with be
            used for the x axis.
        y
            Column name(s) for y axis. Can be:

            - a single column name, in which case a single line will be drawn;
            - a list of column names, in which case one line will be drawn for
              each column name in the list;

            If a single column contains multiple values for the same
            x-coordinate, then behaviour is backend-dependent.
        backend
            Plotting backend to use, which must implement the
            `Dataframe Plotting API <https://github.com/data-apis/dataframe-plotting-api>`_.
            Backends currently known to be supported are:

            - 'hvplot' (requires ``hvplot>=0.9.1`` package)
            - 'plotly' (requires ``plotly>=5.16.0`` package)

            .. note::
                Plotting backends are developed and maintained by third parties,
                independently of Polars.
        kwargs
            Additional keyword arguments to pass to the plotting backend.
            Any keyword argument supported by the plotting backend can be passed -
            how it works is entirely backend-dependent.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 2, 5], "c": [7, 8, 10]})
        >>> df.plot.line(x="a", y=["b", "c"], backend="plotly")  # doctest: +SKIP
        """
        return self._plot(kind="line", x=x, y=y, backend=backend, **kwargs)

    def scatter(
        self,
        *,
        y: str,
        x: str | None = None,
        backend: str | None = None,
        **kwargs: Any,
    ) -> PlotNameSpace:
        """
        Produce scatter plot.

        Parameters
        ----------
        x
            Column name for x axis. If not specified, then a row count with be
            used for the x-axis.
        y
            Column name(s) for y axis. Can be:

            - a single column name, in which case a single line will be drawn;
            - a list of column names, in which case one line will be drawn for
              each column name in the list;

            If a single column contains multiple values for the same
            x-coordinate, then behaviour is backend-dependent.
        backend
            Plotting backend to use, which must implement the
            `Dataframe Plotting API <https://github.com/data-apis/dataframe-plotting-api>`_.
            Backends currently known to be supported are:

            - 'hvplot' (requires ``hvplot>=0.9.1`` package)
            - 'plotly' (requires ``plotly>=5.16.0`` package)

            .. note::
                Plotting backends are developed and maintained by third parties,
                independently of Polars.
        kwargs
            Additional keyword arguments to pass to the plotting backend.
            Any keyword argument supported by the plotting backend can be passed -
            how it works is entirely backend-dependent.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {"a": [1, 2, 3], "b": [4, 2, 5], "c": ["setosa", "setosa", "virginica"]}
        ... )
        >>> df.plot.scatter(x="a", y="b", backend="hvplot")  # doctest: +SKIP
        """
        return self._plot(kind="scatter", x=x, y=y, backend=backend, **kwargs)
