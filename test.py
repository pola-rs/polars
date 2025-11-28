import polars as pl

# df = pl.DataFrame({"a": list(range(2 * 3 * 2))})
# q = df.select(pl.col("a").reshape((4, 1)))

# # df = pl.DataFrame({"a": [1, 2, 3, 4]})
# # s = df.to_series()
# # out = s.reshape((-1, 2))
# print(q)


import datetime
from decimal import Decimal

df = pl.DataFrame(
    [
        pl.Series("bool", [True, False, None], dtype=pl.Boolean),
        pl.Series("int8", [1, 2, None], dtype=pl.Int8),
        pl.Series("int16", [1, 2, None], dtype=pl.Int16),
        pl.Series("int32", [1, 2, None], dtype=pl.Int32),
        pl.Series("int64", [1, 2, None], dtype=pl.Int64),
        pl.Series("uint8", [1, 2, None], dtype=pl.UInt8),
        pl.Series("uint16", [1, 2, None], dtype=pl.UInt16),
        pl.Series("uint32", [1, 2, None], dtype=pl.UInt32),
        pl.Series("uint64", [1, 2, None], dtype=pl.UInt64),
        pl.Series(
            "float32",
            [1.100000023841858, 2.200000047683716, None],
            dtype=pl.Float32,
        ),
        pl.Series("float64", [1.1, 2.2, None], dtype=pl.Float64),
        pl.Series("string", ["hello", "world", None], dtype=pl.String),
        pl.Series("binary", [b"hello", b"world", None], dtype=pl.Binary),
        pl.Series(
            "decimal",
            [Decimal("1.23"), Decimal("4.56"), None],
            dtype=pl.Decimal(precision=10, scale=2),
        ),
        pl.Series(
            "date",
            [datetime.date(2023, 1, 1), datetime.date(2023, 1, 2), None],
            dtype=pl.Date,
        ),
        pl.Series(
            "datetime",
            [
                datetime.datetime(2023, 1, 1, 12, 0),
                datetime.datetime(2023, 1, 2, 13, 30),
                None,
            ],
            dtype=pl.Datetime(time_unit="us", time_zone=None),
        ),
        pl.Series(
            "time",
            [datetime.time(12, 0), datetime.time(13, 30), None],
            dtype=pl.Time,
        ),
        pl.Series(
            "duration_us",
            [datetime.timedelta(days=1), datetime.timedelta(seconds=7200), None],
            dtype=pl.Duration(time_unit="us"),
        ),
        pl.Series(
            "duration_ms",
            [datetime.timedelta(microseconds=100000), datetime.timedelta(0), None],
            dtype=pl.Duration(time_unit="ms"),
        ),
        pl.Series(
            "duration_ns",
            [
                datetime.timedelta(seconds=1),
                datetime.timedelta(microseconds=1000),
                None,
            ],
            dtype=pl.Duration(time_unit="ns"),
        ),
        pl.Series(
            "categorical", ["apple", "banana", "apple"], dtype=pl.Categorical
        ),
        pl.Series(
            "categorical_named",
            ["apple", "banana", "apple"],
            dtype=pl.Categorical(pl.Categories(name="test")),
        ),
    ]
)

print(df)

result = df.map_columns(
    pl.selectors.all(),
    lambda s: pl.Series((s.reshape((1, 1)))).reshape((1,)),
)

print(result)
