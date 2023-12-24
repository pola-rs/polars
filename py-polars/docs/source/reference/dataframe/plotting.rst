========
Plotting
========

Polars does not implement plotting logic itself, but instead defers to
hvplot. Please see `hvplot Plotting Extensions <https://hvplot.holoviz.org/user_guide/Plotting_Extensions.html>`_
for more information and documentation.

Examples
--------
Scatter plot:

.. code-block:: python

    df = pl.DataFrame(
        {
            "length": [1, 4, 6],
            "width": [4, 5, 6],
            "species": ["setosa", "setosa", "versicolor"],
        }
    )
    df.plot.scatter(x="length", y="width", by="species")

Line plot:

.. code-block:: python

    from datetime import date
    df = pl.DataFrame(
        {
            "date": [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 3)],
            "stock_1": [1, 4, 6],
            "stock_2": [1, 5, 2],
        }
    )
    df.plot.line(x="date", y=["stock_1", "stock_2"])
