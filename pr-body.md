Fixes #21898

## Summary

Adds a `strict` parameter to `pl.datetime` (default `True`). When `strict=False`, invalid date or time components produce `null` instead of raising `ComputeError`, restoring the pre-#21013 behavior as an opt-in. This is useful for users processing messy data where invalid components (e.g. `second=63`) should be silently dropped.

```python
>>> df = pl.DataFrame({"year": [2024], "month": [1], "day": [1],
...                    "hour": [1], "minute": [1], "second": [63]})
>>> df.select(pl.datetime("year", "month", "day", "hour", "minute", "second"))
ComputeError: Invalid time components (1, 1, 63, 0) supplied

>>> df.select(pl.datetime("year", "month", "day", "hour", "minute", "second", strict=False))
shape: (1, 1)
┌──────────────┐
│ datetime     │
│ ---          │
│ datetime[μs] │
╞══════════════╡
│ null         │
└──────────────┘
```

## Implementation

The flag is threaded through the full stack:

- Python wrapper (`datetime_`) and `_plr.pyi` stub
- PyO3 binding (`polars-python`) with `strict=true` default
- `DatetimeArgs` (builder field + `with_strict`)
- `TemporalFunction::DatetimeFunction` / `IRTemporalFunction::DatetimeFunction` (DSL and IR), including both conversion directions
- Physical dispatch -> `DatetimeChunked::new_from_parts`

`new_from_parts` now:

- returns `null` for invalid date/time components when `strict=False`, and raises exactly as before when `strict=True`
- validates nanoseconds in `i64` before the `u32` downcast, so out-of-range values no longer silently truncate into a valid-looking datetime
- returns `null` for nanosecond-precision datetimes outside the `i64` range when `strict=False`, and raises a clear `ComputeError` when `strict=True` (previously this path panicked)

`dt.replace` continues to raise on invalid components (passes `strict=true`), keeping its existing behavior.

`strict` only governs invalid date/time components. Ambiguous and non-existent local times are still handled by `ambiguous`.

## Tests

Added coverage for: literal invalid components with `strict=False` -> `null` (dates and times), row-wise invalid components with valid rows preserved, explicit `strict=True` still raising, all-valid literals with `strict=False`, nanosecond range boundaries (both modes), and ambiguous-time handling via `ambiguous="null"` with `strict=False`.

Closes #21898
## Note on the DSL schema hash

The new `strict` field is excluded from the `schemars` derivation of
`TemporalFunction` / `IRTemporalFunction`
(`#[cfg_attr(feature = "dsl-schema", schemars(skip))]`), keeping the
`dsl-schema-hashes.json` unchanged. Happy to regenerate the hashes instead if
you prefer the field reflected in the schema.
