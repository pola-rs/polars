# Polars

## Installation

Installing Polars is just a simple pip install. All binaries are pre-built for python >= 3.6.

`$ pip3 install polars`

## Getting started
Below we show a simple snippet that parses a csv and does a filter followed by a groupby operation.
The eager API must feel very similar to users familiar to pandas. The lazy api is more declarative and describes what 
you want, not how you want it.

### Eager quickstart
```python
import polars as pl

df = pl.read_csv("https://j.mp/iriscsv")
df[df["sepal_length"] > 5].groupby("species").sum()
```

### Lazy quickstart
```python
(pl.scan_csv("iris.csv")
     .filter(pl.col("sepal_length") > 5)
     .groupby("species")
     .agg(pl.col("*").sum())
).collect()
```

This outputs:


<div>
   <style scoped>
      .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
      }
      .dataframe tbody tr th {
      vertical-align: top;
      }
      .dataframe thead th {
      text-align: right;
      }
   </style>
   <table border="1 "class="dataframe ">
      <thead>
         <tr>
            <th>
               species
            </th>
            <th>
               sepal_length_sum
            </th>
            <th>
               sepal_width_sum
            </th>
            <th>
               petal_length_sum
            </th>
            <th>
               petal_width_sum
            </th>
         </tr>
         <tr>
            <td>
               str
            </td>
            <td>
               f64
            </td>
            <td>
               f64
            </td>
            <td>
               f64
            </td>
            <td>
               f64
            </td>
         </tr>
      </thead>
      <tbody>
         <tr>
            <td>
               "setosa"
            </td>
            <td>
               116.9
            </td>
            <td>
               81.7
            </td>
            <td>
               33.2
            </td>
            <td>
               6.1
            </td>
         </tr>
         <tr>
            <td>
               "virginica"
            </td>
            <td>
               324.5
            </td>
            <td>
               146.2
            </td>
            <td>
               273.1
            </td>
            <td>
               99.6
            </td>
         </tr>
         <tr>
            <td>
               "versicolor"
            </td>
            <td>
               281.9
            </td>
            <td>
               131.8
            </td>
            <td>
               202.9
            </td>
            <td>
               63.3
            </td>
         </tr>
      </tbody>
   </table>
</div>

## Eager
The eager API is similar to pandas. Operations are executed directly in an imperative manner. 
The important data structures are [DataFrame's](frame.html#polars.frame.DataFrame) and [Series](series.html#polars.series.Series)

### DataFrame
Read more about the [eager DataFrame operations](frame.html#polars.frame.DataFrame).

### Series
Read more about the [eager Series operations](series.html#polars.series.Series).

## Lazy
The lazy API builds a query plan. Nothing is executed until you explicitly ask polars to execute the query 
(via `LazyFrame.collect()`, or `LazyFrame.fetch`). This provides polars with the entire context of the query and allows 
for optimizations and choosing the fastest algorithm given that context.

### LazyFrame
A `LazyFrame` is a `DataFrame` abstraction that lazily keeps track of the query plan. 
Read more about the [Lazy DataFrame operations](lazy/index.html#polars.lazy.LazyFrame).

### Expr
The arguments given to a `LazyFrame` can be constructed by building simple or complex queries. See the examples in the 
[how can I? section in the book](https://ritchie46.github.io/polars-book/how_can_i/intro.html).

The API of the [Expr can be found here](lazy/index.html#polars.lazy.Expr).

## User Guide
The [polars book](https://ritchie46.github.io/polars-book/) provides more in-depth information about polars. Reading
this will provide you with a more thorough understanding of what polars lazy has to offer, and what kind of
optimizations are done by the query optimizer.
