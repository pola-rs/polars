# Filtering

Filtering date columns works in the same way as with other types of columns using the `.filter` method.

Polars uses Python's native `datetime`, `date` and `timedelta` for equality comparisons between the datatypes `pl.Datetime`, `pl.Date` and `pl.Duration`.

In the following example we use a time series of Apple stock prices.

{{code_block('user-guide/transformations/time-series/filter','df',['read_csv'])}}

```python exec="on" result="text" session="user-guide/transformations/ts/filter"
--8<-- "python/user-guide/transformations/time-series/filter.py:df"
```

## Filtering by single dates

We can filter by a single date by casting the desired date string to a `Date` object
in a filter expression:

{{code_block('user-guide/transformations/time-series/filter','filter',['filter'])}}

```python exec="on" result="text" session="user-guide/transformations/ts/filter"
--8<-- "python/user-guide/transformations/time-series/filter.py:filter"
```

Note we are using the lowercase `datetime` method rather than the uppercase `Datetime` data type.

## Filtering by a date range

We can filter by a range of dates using the `is_between` method in a filter expression with the start and end dates:

{{code_block('user-guide/transformations/time-series/filter','range',['filter','is_between'])}}

```python exec="on" result="text" session="user-guide/transformations/ts/filter"
--8<-- "python/user-guide/transformations/time-series/filter.py:range"
```

## Filtering with negative dates

Say you are working with an archeologist and are dealing in negative dates.
Polars can parse and store them just fine, but the Python `datetime` library
does not. So for filtering, you should use attributes in the `.dt` namespace:

{{code_block('user-guide/transformations/time-series/filter','negative',['str.to_date'])}}

```python exec="on" result="text" session="user-guide/transformations/ts/filter"
--8<-- "python/user-guide/transformations/time-series/filter.py:negative"
```
