import polars as pl
from datetime import date, datetime, timezone
from expression_lib import language, dist, date_util

df = pl.DataFrame(
    {
        "names": ["Richard", "Alice", "Bob"],
        "moons": ["full", "half", "red"],
        "dates": [date(2023, 1, 1), date(2024, 1, 1), date(2025, 1, 1)],
        "datetime": [datetime.now(tz=timezone.utc)] * 3,
        "dist_a": [[12, 32, 1], [], [1, -2]],
        "dist_b": [[-12, 1], [43], [876, -45, 9]],
        "start_lat": [5.6, -1245.8, 242.224],
        "start_lon": [3.1, -1.1, 128.9],
        "end_lat": [6.6, -1243.8, 240.224],
        "end_lon": [3.9, -2, 130],
    }
)

out = df.with_columns(
    pig_latin=language.pig_latinnify("names"),
    pig_latin_cap=language.pig_latinnify("names", capitalize=True),
).with_columns(
    hamming_dist=dist.hamming_distance("names", "pig_latin"),
    jaccard_sim=dist.jaccard_similarity("dist_a", "dist_b"),
    haversine=dist.haversine("start_lat", "start_lon", "end_lat", "end_lon"),
    leap_year=date_util.is_leap_year("dates"),
    new_tz=date_util.change_time_zone("datetime"),
    appended_args=language.append_args(
        "names",
        float_arg=11.234,
        integer_arg=93,
        boolean_arg=False,
        string_arg="example",
    ),
)

print(out)

# Test we can extend the expressions by importing the extension module.

import expression_lib.extension  # noqa: E402, F401

out = df.with_columns(
    pig_latin=pl.col("names").language.pig_latinnify(),
    pig_latin_cap=pl.col("names").language.pig_latinnify(capitalize=True),
).with_columns(
    hamming_dist=pl.col("names").dist.hamming_distance("pig_latin"),
    jaccard_sim=pl.col("dist_a").dist.jaccard_similarity("dist_b"),
    haversine=pl.col("start_lat").dist.haversine("start_lon", "end_lat", "end_lon"),
    leap_year=pl.col("dates").date_util.is_leap_year(),
    new_tz=pl.col("datetime").date_util.change_time_zone(),
    appended_args=pl.col("names").language.append_args(
        float_arg=11.234,
        integer_arg=93,
        boolean_arg=False,
        string_arg="example",
    ),
)

print(out)


# Tests we can return errors from FFI by passing wrong types.
try:
    out.with_columns(
        appended_args=pl.col("names").language.append_args(
            float_arg=True,
            integer_arg=True,
            boolean_arg=True,
            string_arg="example",
        )
    )
except pl.exceptions.ComputeError as e:
    assert "the plugin failed with message" in str(e)


try:
    out.with_columns(pl.col("names").panic.panic())
except pl.exceptions.ComputeError as e:
    assert "the plugin panicked" in str(e)

print("finished")
