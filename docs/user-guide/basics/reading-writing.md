# Reading & writing

Polars supports reading and writing to all common files (e.g. csv, json, parquet), cloud storage (S3, Azure Blob, BigQuery) and databases (e.g. postgres, mysql). In the following examples we will show how to operate on most common file formats. For the following dataframe

{{code_block('user-guide/basics/reading-writing','dataframe',['DataFrame'])}}

```python exec="on" result="text" session="getting-started/reading"
--8<-- "python/user-guide/basics/reading-writing.py:dataframe"
```

#### CSV

Polars has its own fast implementation for csv reading with many flexible configuration options.

{{code_block('user-guide/basics/reading-writing','csv',['read_csv','write_csv'])}}

```python exec="on" result="text" session="getting-started/reading"
--8<-- "python/user-guide/basics/reading-writing.py:csv"
```

As we can see above, Polars made the datetimes a `string`. We can tell Polars to parse dates, when reading the csv, to ensure the date becomes a datetime. The example can be found below:

{{code_block('user-guide/basics/reading-writing','csv2',['read_csv'])}}

```python exec="on" result="text" session="getting-started/reading"
--8<-- "python/user-guide/basics/reading-writing.py:csv2"
```

#### JSON

{{code_block('user-guide/basics/reading-writing','json',['read_json','write_json'])}}

```python exec="on" result="text" session="getting-started/reading"
--8<-- "python/user-guide/basics/reading-writing.py:json"
```

#### Parquet

{{code_block('user-guide/basics/reading-writing','parquet',['read_parquet','write_parquet'])}}

```python exec="on" result="text" session="getting-started/reading"
--8<-- "python/user-guide/basics/reading-writing.py:parquet"
```

To see more examples and other data formats go to the [User Guide](../io/csv.md), section IO.
