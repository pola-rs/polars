# Resampling

We can resample by either:

- upsampling (moving data to a higher frequency)
- downsampling (moving data to a lower frequency)
- combinations of these e.g. first upsample and then downsample

## Downsampling to a lower frequency

Polars views downsampling as a special case of the **group_by** operation and you can do this with `group_by_dynamic` and `group_by_rolling` - [see the temporal group by page for examples](rolling.md).

## Upsampling to a higher frequency

Let's go through an example where we generate data at 30 minute intervals:

{{code_block('user-guide/transformations/time-series/resampling','df',['DataFrame','date_range'])}}

```python exec="on" result="text" session="user-guide/transformations/ts/resampling"
--8<-- "python/user-guide/transformations/time-series/resampling.py:setup"
--8<-- "python/user-guide/transformations/time-series/resampling.py:df"
```

Upsampling can be done by defining the new sampling interval. By upsampling we are adding in extra rows where we do not have data. As such upsampling by itself gives a DataFrame with nulls. These nulls can then be filled with a fill strategy or interpolation.

### Upsampling strategies

In this example we upsample from the original 30 minutes to 15 minutes and then use a `forward` strategy to replace the nulls with the previous non-null value:

{{code_block('user-guide/transformations/time-series/resampling','upsample',['upsample'])}}

```python exec="on" result="text" session="user-guide/transformations/ts/resampling"
--8<-- "python/user-guide/transformations/time-series/resampling.py:upsample"
```

In this example we instead fill the nulls by linear interpolation:

{{code_block('user-guide/transformations/time-series/resampling','upsample2',['upsample','interpolate','fill_null'])}}

```python exec="on" result="text" session="user-guide/transformations/ts/resampling"
--8<-- "python/user-guide/transformations/time-series/resampling.py:upsample2"
```
