# Parsing

Polars has native support for parsing time series data and doing more sophisticated operations such as temporal grouping and resampling.

## Datatypes

Polars has the following datetime datatypes:

- `Date`: Date representation e.g. 2014-07-08. It is internally represented as days since UNIX epoch encoded by a 32-bit signed integer.
- `Datetime`: Datetime representation e.g. 2014-07-08 07:00:00. It is internally represented as a 64 bit integer since the Unix epoch and can have different units such as ns, us, ms.
- `Duration`: A time delta type that is created when subtracting `Date/Datetime`. Similar to `timedelta` in Python.
- `Time`: Time representation, internally represented as nanoseconds since midnight.

## Parsing dates from a file

When loading from a CSV file Polars attempts to parse dates and times if the `try_parse_dates` flag is set to `True`:

{{code_block('user-guide/transformations/time-series/parsing','df',['read_csv'])}}

```python exec="on" result="text" session="user-guide/transformations/ts/parsing"
--8<-- "python/user-guide/transformations/time-series/parsing.py:setup"
--8<-- "python/user-guide/transformations/time-series/parsing.py:df"
```

On the other hand binary formats such as parquet have a schema that is respected by Polars.

## Casting strings to dates

You can also cast a column of datetimes encoded as strings to a datetime type. You do this by calling the string `str.to_date` method and passing the format of the date string:

{{code_block('user-guide/transformations/time-series/parsing','cast',['read_csv','str.to_date'])}}

```python exec="on" result="text" session="user-guide/transformations/ts/parsing"
--8<-- "python/user-guide/transformations/time-series/parsing.py:cast"
```

[The format string specification can be found here.](https://docs.rs/chrono/latest/chrono/format/strftime/index.html).

## Extracting date features from a date column

You can extract data features such as the year or day from a date column using the `.dt` namespace on a date column:

{{code_block('user-guide/transformations/time-series/parsing','extract',['dt.year'])}}

```python exec="on" result="text" session="user-guide/transformations/ts/parsing"
--8<-- "python/user-guide/transformations/time-series/parsing.py:extract"
```

## Mixed offsets

If you have mixed offsets (say, due to crossing daylight saving time),
then you can use `utc=True` and then convert to your time zone:

{{code_block('user-guide/transformations/time-series/parsing','mixed',['str.to_datetime','dt.convert_time_zone'])}}

```python exec="on" result="text" session="user-guide/transformations/ts/parsing"
--8<-- "python/user-guide/transformations/time-series/parsing.py:mixed"
```
