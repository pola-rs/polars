---
hide:
  - toc
---

# Time zones

!!! quote "Tom Scott"

    You really should never, ever deal with time zones if you can help it.

The `Datetime` datatype can have a time zone associated with it.
Examples of valid time zones are:

- `None`: no time zone, also known as "time zone naive".
- `UTC`: Coordinated Universal Time.
- `Asia/Kathmandu`: time zone in "area/location" format.
  See the [list of tz database time zones](https://en.wikipedia.org/wiki/List_of_tz_database_time_zones)
  to see what's available.

Caution: Fixed offsets such as +02:00, should not be used for handling time zones. It's advised to use the "Area/Location" format mentioned above, as it can manage timezones more effectively.

Note that, because a `Datetime` can only have a single time zone, it is
impossible to have a column with multiple time zones. If you are parsing data
with multiple offsets, you may want to pass `utc=True` to convert
them all to a common time zone (`UTC`), see [parsing dates and times](parsing.md).

The main methods for setting and converting between time zones are:

- `dt.convert_time_zone`: convert from one time zone to another.
- `dt.replace_time_zone`: set/unset/change time zone.
- `dt.to_local_datetime`: convert to the local datetime in a given time zone
- `dt.from_local_datetime`: convert local datetime from given time zone to new time zone

Let's look at some examples of common operations:

{{code_block('user-guide/transformations/time-series/timezones','example',['str.to_datetime','dt.replace_time_zone'])}}

```python exec="on" result="text" session="user-guide/transformations/ts/timezones"
--8<-- "python/user-guide/transformations/time-series/timezones.py:setup"
--8<-- "python/user-guide/transformations/time-series/timezones.py:example"
```

{{code_block('user-guide/transformations/time-series/timezones','example2',['dt.convert_time_zone','dt.replace_time_zone'])}}

```python exec="on" result="text" session="user-guide/transformations/ts/timezones"
--8<-- "python/user-guide/transformations/time-series/timezones.py:example2"
```

{{code_block('user-guide/transformations/time-series/timezones','example3',['dt.to_local_datetime','dt.from_local_datetime'])}}

```python exec="on" result="text" session="user-guide/transformations/ts/timezones"
--8<-- "python/user-guide/transformations/time-series/timezones.py:example3"
```
