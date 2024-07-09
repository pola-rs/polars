# Visualization

Data in a Polars `DataFrame` can be visualized using common visualization libraries.

We illustrate plotting capabilities using the Iris dataset. We scan a CSV and then do a group-by on the `species` column and get the mean of the `petal_length`.

{{code_block('user-guide/misc/visualization','dataframe',[])}}

```python exec="on" result="text" session="user-guide/misc/visualization"
--8<-- "python/user-guide/misc/visualization.py:dataframe"
```

## Built-in plotting with hvPlot

Polars has a `plot` method to create interactive plots using [hvPlot](https://hvplot.holoviz.org/).

{{code_block('user-guide/misc/visualization','hvplot_show_plot',[])}}

```python exec="on" session="user-guide/misc/visualization"
--8<-- "python/user-guide/misc/visualization.py:hvplot_make_plot"
```

## Matplotlib

To create a bar chart we can pass columns of a `DataFrame` directly to Matplotlib as a `Series` for each column. Matplotlib does not have explicit support for Polars objects but Matplotlib can accept a Polars `Series` because it can convert each Series to a numpy array, which is zero-copy for numeric
data without null values.

{{code_block('user-guide/misc/visualization','matplotlib_show_plot',[])}}

```python exec="on" session="user-guide/misc/visualization"
--8<-- "python/user-guide/misc/visualization.py:matplotlib_make_plot"
```

## Seaborn, Plotly & Altair

[Seaborn](https://seaborn.pydata.org/), [Plotly](https://plotly.com/) & [Altair](https://altair-viz.github.io/) can accept a Polars `DataFrame` by leveraging the [dataframe interchange protocol](https://data-apis.org/dataframe-api/), which offers zero-copy conversion where possible.

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

### Altair

{{code_block('user-guide/misc/visualization','altair_show_plot',[])}}

```python exec="on" session="user-guide/misc/visualization"
--8<-- "python/user-guide/misc/visualization.py:altair_make_plot"
```
