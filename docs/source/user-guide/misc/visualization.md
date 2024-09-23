# Visualization

Data in a Polars `DataFrame` can be visualized using common visualization libraries.

We illustrate plotting capabilities using the Iris dataset. We read a CSV and then
plot one column against another, colored by a yet another column.

{{code_block('user-guide/misc/visualization','dataframe',[])}}

```python exec="on" result="text" session="user-guide/misc/visualization"
--8<-- "python/user-guide/misc/visualization.py:dataframe"
```

## Built-in plotting with Altair

Polars has a `plot` method to create plots using [Altair](https://altair-viz.github.io/):

{{code_block('user-guide/misc/visualization','altair_show_plot',[])}}

```python exec="on" session="user-guide/misc/visualization"
--8<-- "python/user-guide/misc/visualization.py:altair_make_plot"
```

This is shorthand for:

```python
import altair as alt

(
    alt.Chart(df).mark_point().encode(
        x="sepal_length",
        y="sepal_width",
        color="species",
    )
    .properties(width=500)
    .configure_scale(zero=False)
)
```

and is only provided for convenience, and to signal that Altair is known to work well with
Polars.

## hvPlot

If you import `hvplot.polars`, then it registers a `hvplot`
method which you can use to create interactive plots using [hvPlot](https://hvplot.holoviz.org/).

{{code_block('user-guide/misc/visualization','hvplot_show_plot',[])}}

```python exec="on" session="user-guide/misc/visualization"
--8<-- "python/user-guide/misc/visualization.py:hvplot_make_plot"
```

## Matplotlib

To create a scatter plot we can pass columns of a `DataFrame` directly to Matplotlib as a `Series` for each column.
Matplotlib does not have explicit support for Polars objects but can accept a Polars `Series` by
converting it to a NumPy array (which is zero-copy for numeric data without null values).

Note that because the column `'species'` isn't numeric, we need to first convert it to numeric values so that
it can be passed as an argument to `c`.

{{code_block('user-guide/misc/visualization','matplotlib_show_plot',[])}}

```python exec="on" session="user-guide/misc/visualization"
--8<-- "python/user-guide/misc/visualization.py:matplotlib_make_plot"
```

## Seaborn and Plotly

[Seaborn](https://seaborn.pydata.org/) and [Plotly](https://plotly.com/) can accept a Polars `DataFrame` by leveraging the [dataframe interchange protocol](https://data-apis.org/dataframe-api/), which offers zero-copy conversion where possible. Note
that the protocol does not support all Polars data types (e.g. `List`) so your mileage may vary here.

### Seaborn

{{code_block('user-guide/misc/visualization','seaborn_show_plot',[])}}

```python exec="on" session="user-guide/misc/visualization"
--8<-- "python/user-guide/misc/visualization.py:seaborn_make_plot"
```

### Plotly

{{code_block('user-guide/misc/visualization','plotly_show_plot',[])}}

```python exec="on" session="user-guide/misc/visualization"
--8<-- "python/user-guide/misc/visualization.py:plotly_make_plot"
```
