from __future__ import annotations

import io
import itertools
import operator
from dataclasses import dataclass
from decimal import Decimal as D
from math import ceil, floor
from random import choice, randrange, seed
from typing import TYPE_CHECKING, NamedTuple

import pyarrow as pa
import pytest

import polars as pl
from polars.exceptions import InvalidOperationError
from polars.testing import assert_frame_equal, assert_series_equal

if TYPE_CHECKING:
    from collections.abc import Callable

    from tests.conftest import PlMonkeyPatch


@pytest.fixture(scope="module")
def permutations_int_dec_none() -> list[tuple[D | int | None, ...]]:
    return list(
        itertools.permutations(
            [
                D("-0.01"),
                D("1.2345678"),
                D("500"),
                -1,
                None,
            ]
        )
    )


@pytest.mark.slow
def test_series_from_pydecimal_and_ints(
    permutations_int_dec_none: list[tuple[D | int | None, ...]],
) -> None:
    # TODO: check what happens if there are strings, floats arrow scalars in the list
    for data in permutations_int_dec_none:
        s = pl.Series("name", data, strict=False)
        assert s.dtype == pl.Decimal(scale=7)  # inferred scale = 7, precision = None
        assert s.dtype.is_decimal()
        assert s.name == "name"
        assert s.null_count() == 1
        for i, d in enumerate(data):
            assert s[i] == d
        assert s.to_list() == [D(x) if x is not None else None for x in data]


@pytest.mark.slow
def test_frame_from_pydecimal_and_ints(
    permutations_int_dec_none: list[tuple[D | int | None, ...]],
    plmonkeypatch: PlMonkeyPatch,
) -> None:
    class X(NamedTuple):
        a: int | D | None

    @dataclass
    class Y:
        a: int | D | None

    for data in permutations_int_dec_none:
        row_data = [(d,) for d in data]
        for cls in (X, Y):
            for ctor in (pl.DataFrame, pl.from_records):
                df = ctor(data=list(map(cls, data)))
                assert df.schema == {
                    "a": pl.Decimal(scale=7),
                }
                assert df.rows() == row_data


@pytest.mark.parametrize(
    ("input", "trim_zeros", "expected"),
    [
        ("0.00", True, "0"),
        ("0.00", False, "0.00"),
        ("-1", True, "-1"),
        ("-1.000000000000000000000000000", False, "-1.000000000000000000000000000"),
        ("0.0100", True, "0.01"),
        ("0.0100", False, "0.0100"),
        ("0.010000000000000000000000000", False, "0.010000000000000000000000000"),
        ("-1.123801239123981293891283123", True, "-1.123801239123981293891283123"),
        (
            "12345678901.234567890123458390192857685",
            True,
            "12345678901.234567890123458390192857685",
        ),
        (
            "-99999999999.999999999999999999999999999",
            True,
            "-99999999999.999999999999999999999999999",
        ),
    ],
)
def test_decimal_format(input: str, trim_zeros: bool, expected: str) -> None:
    with pl.Config(trim_decimal_zeros=trim_zeros):
        series = pl.Series([input]).str.to_decimal()
        formatted = str(series).split("\n")[-2].strip()
        assert formatted == expected


def test_init_decimal_dtype() -> None:
    s = pl.Series(
        "a", [D("-0.01"), D("1.2345678"), D("500")], dtype=pl.Decimal, strict=False
    )
    assert s.dtype.is_numeric()

    df = pl.DataFrame(
        {"a": [D("-0.01"), D("1.2345678"), D("500")]},
        schema={"a": pl.Decimal},
        strict=False,
    )
    assert df["a"].dtype.is_numeric()


def test_decimal_convert_to_float_by_schema() -> None:
    # Column Based
    df = pl.DataFrame(
        {"a": [D("1"), D("2.55"), D("45.000"), D("10.0")]}, schema={"a": pl.Float64}
    )
    expected = pl.DataFrame({"a": [1.0, 2.55, 45.0, 10.0]})
    assert_frame_equal(df, expected)

    # Row Based
    df = pl.DataFrame(
        [[D("1"), D("2.55"), D("45.000"), D("10.0")]], schema={"a": pl.Float64}
    )
    expected = pl.DataFrame({"a": [1.0, 2.55, 45.0, 10.0]})
    assert_frame_equal(df, expected)


def test_df_constructor_convert_decimal_to_float_9873() -> None:
    result = pl.DataFrame(
        [[D("45.0000")], [D("45.0000")]], schema={"a": pl.Float64}, orient="row"
    )
    expected = pl.DataFrame({"a": [45.0, 45.0]})
    assert_frame_equal(result, expected)


def test_decimal_cast() -> None:
    df = pl.DataFrame(
        {
            "decimals": [D("2"), D("2"), D("-1.5")],
        },
        strict=False,
    )

    result = df.with_columns(pl.col("decimals").cast(pl.Float32).alias("b2"))
    expected = {"decimals": [D("2"), D("2"), D("-1.5")], "b2": [2.0, 2.0, -1.5]}
    assert result.to_dict(as_series=False) == expected


def test_decimal_cast_no_scale() -> None:
    with pytest.raises(TypeError):
        pl.Series().cast(pl.Decimal)


def test_decimal_scale_precision_roundtrip(plmonkeypatch: PlMonkeyPatch) -> None:
    assert pl.from_arrow(pl.Series("dec", [D("10.0")]).to_arrow()).item() == D("10.0")


def test_string_to_decimal() -> None:
    values = [
        "40.12",
        "3420.13",
        "120134.19",
        "3212.98",
        "12.90",
        "143.09",
        "143.9",
        "-62.44",
    ]

    s = pl.Series(values).str.to_decimal()
    assert s.dtype == pl.Decimal(precision=8, scale=2)

    assert s.to_list() == [D(v) for v in values]


def test_read_csv_decimal(plmonkeypatch: PlMonkeyPatch) -> None:
    csv = """a,b
123.12,a
1.1,a
0.01,a"""

    df = pl.read_csv(csv.encode(), schema_overrides={"a": pl.Decimal(scale=2)})
    assert df.dtypes == [pl.Decimal(scale=2), pl.String]
    assert df["a"].to_list() == [
        D("123.12"),
        D("1.10"),
        D("0.01"),
    ]


def test_decimal_eq_number() -> None:
    a = pl.Series([D("1.5"), D("22.25"), D("10.0")], dtype=pl.Decimal, strict=False)
    assert_series_equal(a == 1, pl.Series([False, False, False]))
    assert_series_equal(a == 1.5, pl.Series([True, False, False]))
    assert_series_equal(a == D("1.5"), pl.Series([True, False, False]))
    assert_series_equal(a == pl.Series([D("1.5")]), pl.Series([True, False, False]))


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        (operator.le, pl.Series([None, True, True, True, True, True])),
        (operator.lt, pl.Series([None, False, False, False, True, True])),
        (operator.ge, pl.Series([None, True, True, True, False, False])),
        (operator.gt, pl.Series([None, False, False, False, False, False])),
    ],
)
def test_decimal_compare(
    op: Callable[[pl.Series, pl.Series], pl.Series], expected: pl.Series
) -> None:
    s = pl.Series(
        [None, D("1.2"), D("2.13"), D("4.99"), D("2.13"), D("1.2")],
        dtype=pl.Decimal,
        strict=False,
    )
    s2 = pl.Series(
        [None, D("1.200"), D("2.13"), D("4.99"), D("4.99"), D("2.13")], strict=False
    )

    assert_series_equal(op(s, s2), expected)


def test_decimal_arithmetic() -> None:
    dt = pl.Decimal(20, 10)
    df = pl.DataFrame(
        {
            "a": [D("0.1"), D("10.1"), D("100.01")],
            "b": [D("20.1"), D("10.19"), D("39.21")],
        },
        strict=False,
        schema={"a": dt, "b": dt},
    )

    out = df.select(
        out1=pl.col("a") + pl.col("b"),
        out2=pl.col("a") - pl.col("b"),
        out3=pl.col("a") * pl.col("b"),
        out4=pl.col("a") / pl.col("b"),
    )
    assert all(dt == pl.Decimal(38, 10) for dt in out.dtypes)

    assert out.to_dict(as_series=False) == {
        "out1": [D("20.2"), D("20.29"), D("139.22")],
        "out2": [D("-20.0"), D("-0.09"), D("60.80")],
        "out3": [D("2.01"), D("102.919"), D("3921.3921")],
        "out4": [D("0.0049751244"), D("0.9911678116"), D("2.5506248406")],
    }


def test_decimal_arithmetic_literal() -> None:
    dt = pl.Decimal(20, 10)
    df = pl.DataFrame(
        {
            "a": [D("0.1"), D("10.1"), D("100.01")],
            "b": [D("20.1"), D("10.19"), D("39.21")],
            "c": [D("412.1023"), D("2349"), D("0")],
        },
        strict=False,
        schema={"a": dt, "b": dt, "c": dt},
    )

    out = df.select(i=pl.col.a * 10, f=pl.col.b + 0.25, d=pl.col.c / D("3"))
    expected = pl.DataFrame(
        {
            "i": [D("1"), D("101"), D("1000.1")],
            "f": [20.35, 10.44, 39.46],
            "d": ["137.3674333333333333333333333", "783", "0"],
        },
        schema={"i": pl.Decimal(38, 10), "f": pl.Float64, "d": pl.Decimal(38, 10)},
    )
    assert_frame_equal(out, expected)


def test_decimal_series_value_arithmetic() -> None:
    s = pl.Series([D("0.10"), D("10.10"), D("100.01")])
    assert s.dtype == pl.Decimal(scale=2)

    out1 = s + 10
    out2 = s + D("10")
    out3 = s + D("10.0001")
    out4 = s * 2 / 3
    out5 = s / D("1.5")
    out6 = s - 5

    assert out1.dtype == pl.Decimal(scale=2)
    assert out2.dtype == pl.Decimal(scale=2)
    assert out3.dtype == pl.Decimal(scale=4)
    assert out4.dtype == pl.Decimal(scale=2)
    assert out5.dtype == pl.Decimal(scale=2)
    assert out6.dtype == pl.Decimal(scale=2)

    assert out1.to_list() == [D("10.1"), D("20.1"), D("110.01")]
    assert out2.to_list() == [D("10.1"), D("20.1"), D("110.01")]
    assert out3.to_list() == [D("10.1001"), D("20.1001"), D("110.0101")]
    assert out4.to_list() == [
        D("0.07"),
        D("6.73"),
        D("66.67"),
    ]  # TODO: do we want floor instead of round?
    assert out5.to_list() == [D("0.07"), D("6.73"), D("66.67")]
    assert out6.to_list() == [D("-4.9"), D("5.1"), D("95.01")]


def test_decimal_aggregations() -> None:
    df = pl.DataFrame(
        {
            "g": [1, 1, 2, 2],
            "a": [D("0.1"), D("10.1"), D("100.01"), D("9000.12")],
        },
        strict=False,
    )

    assert df.group_by("g").agg("a").sort("g").to_dict(as_series=False) == {
        "g": [1, 2],
        "a": [[D("0.1"), D("10.1")], [D("100.01"), D("9000.12")]],
    }

    result = df.group_by("g", maintain_order=True).agg(
        sum=pl.sum("a"),
        min=pl.min("a"),
        max=pl.max("a"),
        mean=pl.mean("a"),
        median=pl.median("a"),
    )
    expected = pl.DataFrame(
        {
            "g": [1, 2],
            "sum": [D("10.20"), D("9100.13")],
            "min": [D("0.10"), D("100.01")],
            "max": [D("10.10"), D("9000.12")],
            "mean": [5.1, 4550.065],
            "median": [5.1, 4550.065],
        }
    )
    assert_frame_equal(result, expected)

    res = df.select(
        sum=pl.sum("a"),
        min=pl.min("a"),
        max=pl.max("a"),
        mean=pl.mean("a"),
        median=pl.median("a"),
    )
    expected = pl.DataFrame(
        {
            "sum": [D("9110.33")],
            "min": [D("0.10")],
            "max": [D("9000.12")],
            "mean": [2277.5825],
            "median": [55.055],
        }
    )
    assert_frame_equal(res, expected)

    description = pl.DataFrame(
        {
            "statistic": [
                "count",
                "null_count",
                "mean",
                "std",
                "min",
                "25%",
                "50%",
                "75%",
                "max",
            ],
            "g": [4.0, 0.0, 1.5, 0.5773502691896257, 1.0, 1.0, 2.0, 2.0, 2.0],
            "a": [
                4.0,
                0.0,
                2277.5825,
                4481.916846516863,
                0.1,
                10.1,
                100.01,
                100.01,
                9000.12,
            ],
        }
    )
    assert_frame_equal(df.describe(), description)


def test_decimal_cumulative_aggregations() -> None:
    df = pl.Series("a", [D("2.2"), D("1.1"), D("3.3")]).to_frame()
    result = df.select(
        pl.col("a").cum_sum().alias("cum_sum"),
        pl.col("a").cum_min().alias("cum_min"),
        pl.col("a").cum_max().alias("cum_max"),
    )
    expected = pl.DataFrame(
        {
            "cum_sum": [D("2.2"), D("3.3"), D("6.6")],
            "cum_min": [D("2.2"), D("1.1"), D("1.1")],
            "cum_max": [D("2.2"), D("2.2"), D("3.3")],
        }
    )
    assert_frame_equal(result, expected)


def test_decimal_df_vertical_sum() -> None:
    df = pl.DataFrame({"a": [D("1.1"), D("2.2")]})
    expected = pl.DataFrame({"a": [D("3.3")]})

    assert_frame_equal(df.sum(), expected)


def test_decimal_df_vertical_agg() -> None:
    df = pl.DataFrame({"a": [D("1.0"), D("2.0"), D("3.0")]})
    expected_min = pl.DataFrame({"a": [D("1.0")]})
    expected_max = pl.DataFrame({"a": [D("3.0")]})
    assert_frame_equal(df.min(), expected_min)
    assert_frame_equal(df.max(), expected_max)


def test_decimal_in_filter() -> None:
    df = pl.DataFrame(
        {
            "foo": [1, 2, 3],
            "bar": ["6", "7", "8"],
        }
    )
    df = df.with_columns(pl.col("bar").cast(pl.Decimal(scale=0)))
    assert df.filter(pl.col("foo") > 1).to_dict(as_series=False) == {
        "foo": [2, 3],
        "bar": [D("7"), D("8")],
    }


def test_decimal_sort() -> None:
    df = pl.DataFrame(
        {
            "foo": [1, 2, 3],
            "bar": [D("3.4"), D("2.1"), D("4.5")],
            "baz": [1, 1, 2],
        }
    )
    assert df.sort("bar").to_dict(as_series=False) == {
        "foo": [2, 1, 3],
        "bar": [D("2.1"), D("3.4"), D("4.5")],
        "baz": [1, 1, 2],
    }
    assert df.sort(["bar", "foo"]).to_dict(as_series=False) == {
        "foo": [2, 1, 3],
        "bar": [D("2.1"), D("3.4"), D("4.5")],
        "baz": [1, 1, 2],
    }
    assert df.sort(["foo", "bar"]).to_dict(as_series=False) == {
        "foo": [1, 2, 3],
        "bar": [D("3.4"), D("2.1"), D("4.5")],
        "baz": [1, 1, 2],
    }

    assert df.select([pl.col("foo").sort_by("bar", descending=True).alias("s1")])[
        "s1"
    ].to_list() == [3, 1, 2]
    assert df.select([pl.col("foo").sort_by(["baz", "bar"]).alias("s2")])[
        "s2"
    ].to_list() == [2, 1, 3]


def test_decimal_unique() -> None:
    df = pl.DataFrame(
        {
            "foo": [1, 1, 2],
            "bar": [D("3.4"), D("3.4"), D("4.5")],
        }
    )
    assert df.unique().sort("bar").to_dict(as_series=False) == {
        "foo": [1, 2],
        "bar": [D("3.4"), D("4.5")],
    }


def test_decimal_write_parquet_12375() -> None:
    df = pl.DataFrame(
        {
            "hi": [True, False, True, False],
            "bye": [D(1), D(2), D(3), D(47283957238957239875)],
        },
    )
    assert df["bye"].dtype == pl.Decimal

    f = io.BytesIO()
    df.write_parquet(f)


def test_decimal_list_get_13847() -> None:
    df = pl.DataFrame({"a": [[D("1.1"), D("1.2")], [D("2.1")]]})
    out = df.select(pl.col("a").list.get(0))
    expected = pl.DataFrame({"a": [D("1.1"), D("2.1")]})
    assert_frame_equal(out, expected)


def test_decimal_explode() -> None:
    nested_decimal_df = pl.DataFrame(
        {
            "bar": [[D("3.4"), D("3.4")], [D("4.5")]],
        }
    )
    df = nested_decimal_df.explode("bar")
    expected_df = pl.DataFrame(
        {
            "bar": [D("3.4"), D("3.4"), D("4.5")],
        }
    )
    assert_frame_equal(df, expected_df)

    # test group-by head #15330
    df = pl.DataFrame(
        {
            "foo": [1, 1, 2],
            "bar": [D("3.4"), D("3.4"), D("4.5")],
        }
    )
    head_df = df.group_by("foo", maintain_order=True).head(1)
    expected_df = pl.DataFrame({"foo": [1, 2], "bar": [D("3.4"), D("4.5")]})
    assert_frame_equal(head_df, expected_df)


def test_decimal_streaming() -> None:
    seed(1)
    scale = D("1e18")
    data = [
        {"group": choice("abc"), "value": randrange(10**32) / scale} for _ in range(20)
    ]
    lf = pl.LazyFrame(data, schema_overrides={"value": pl.Decimal(scale=18)})
    assert lf.group_by("group").agg(pl.sum("value")).collect(engine="streaming").sort(
        "group"
    ).to_dict(as_series=False) == {
        "group": ["a", "b", "c"],
        "value": [
            D("244215083629512.120161049441284000"),
            D("510640422312378.070344831471216000"),
            D("161102921617598.363263936811563000"),
        ],
    }


def test_decimal_supertype() -> None:
    q = pl.LazyFrame([0.12345678]).select(
        pl.col("column_0").cast(pl.Decimal(scale=6)) * 1
    )
    assert q.collect().dtypes[0].is_decimal()


def test_decimal_raise_oob_precision() -> None:
    df = pl.DataFrame({"a": [1.0]})
    # max precision is 38.
    with pytest.raises(pl.exceptions.InvalidOperationError):
        df.select(b=pl.col("a").cast(pl.Decimal(76, 38)))


def test_decimal_dynamic_float_st() -> None:
    assert pl.LazyFrame({"a": [D("2.0"), D("0.5")]}).filter(
        pl.col("a").is_between(0.45, 0.9)
    ).collect().to_dict(as_series=False) == {"a": [D("0.5")]}


def test_decimal_strict_scale_inference_17770() -> None:
    values = [D("0.1"), D("0.10"), D("1.0121")]
    s = pl.Series(values, strict=True)
    assert s.dtype == pl.Decimal(precision=None, scale=4)
    assert s.to_list() == values


def test_decimal_round() -> None:
    dtype = pl.Decimal(3, 2)
    values = [D(f"{float(v) / 100.0:.02f}") for v in range(-150, 250, 1)]
    i_s = pl.Series("a", values, dtype)

    floor_s = pl.Series("a", [floor(v) for v in values], dtype)
    ceil_s = pl.Series("a", [ceil(v) for v in values], dtype)

    assert_series_equal(i_s.floor(), floor_s)
    assert_series_equal(i_s.ceil(), ceil_s)

    for decimals in range(10):
        got_s = i_s.round(decimals)
        expected_s = pl.Series("a", [round(v, decimals) for v in values], dtype)

        assert_series_equal(got_s, expected_s)


def test_decimal_arithmetic_schema() -> None:
    q = pl.LazyFrame({"x": [1.0]}, schema={"x": pl.Decimal(15, 2)})

    q1 = q.select(pl.col.x * pl.col.x)
    assert q1.collect_schema() == q1.collect().schema
    q1 = q.select(pl.col.x / pl.col.x)
    assert q1.collect_schema() == q1.collect().schema
    q1 = q.select(pl.col.x - pl.col.x)
    assert q1.collect_schema() == q1.collect().schema
    q1 = q.select(pl.col.x + pl.col.x)
    assert q1.collect_schema() == q1.collect().schema


def test_decimal_arithmetic_schema_float_20369() -> None:
    s = pl.Series("x", [1.0], dtype=pl.Decimal(15, 6))
    assert_series_equal((s - 1.0), pl.Series("x", [0.0]))
    assert_series_equal((3.0 - s), pl.Series("literal", [2.0]))
    assert_series_equal((3.0 / s), pl.Series("literal", [3.0]))
    assert_series_equal((s / 3.0), pl.Series("x", [0.333333]))

    assert_series_equal((s + 1.0), pl.Series("x", [2.0]))
    assert_series_equal((1.0 + s), pl.Series("literal", [2.0]))
    assert_series_equal((s * 1.0), pl.Series("x", [1.0]))
    assert_series_equal((1.0 * s), pl.Series("literal", [1.0]))


def test_decimal_arithmetic_schema_int() -> None:
    s = pl.Series("x", [1.0], dtype=pl.Decimal(15, 6))
    assert_series_equal((s - 1), pl.Series("x", [0.0], dtype=pl.Decimal(38, 6)))
    assert_series_equal((3 - s), pl.Series("literal", [2.0], dtype=pl.Decimal(38, 6)))
    assert_series_equal((3 / s), pl.Series("literal", [3.0], dtype=pl.Decimal(38, 6)))
    assert_series_equal((s / 3), pl.Series("x", [0.333333], dtype=pl.Decimal(38, 6)))

    assert_series_equal((s + 1), pl.Series("x", [2.0], dtype=pl.Decimal(38, 6)))
    assert_series_equal((1 + s), pl.Series("literal", [2.0], dtype=pl.Decimal(38, 6)))
    assert_series_equal((s * 1), pl.Series("x", [1.0], dtype=pl.Decimal(38, 6)))
    assert_series_equal((1 * s), pl.Series("literal", [1.0], dtype=pl.Decimal(38, 6)))


def test_decimal_horizontal_20482() -> None:
    b = pl.LazyFrame(
        {
            "a": [D("123.000000"), D("234.000000")],
            "b": [D("123.000000"), D("234.000000")],
        },
        schema={
            "a": pl.Decimal(18, 6),
            "b": pl.Decimal(18, 6),
        },
    )

    assert (
        b.select(
            min=pl.min_horizontal(pl.col("a"), pl.col("b")),
            max=pl.max_horizontal(pl.col("a"), pl.col("b")),
            sum=pl.sum_horizontal(pl.col("a"), pl.col("b")),
        ).collect()
    ).to_dict(as_series=False) == {
        "min": [D("123.000000"), D("234.000000")],
        "max": [D("123.000000"), D("234.000000")],
        "sum": [D("246.000000"), D("468.000000")],
    }


def test_decimal_horizontal_different_scales_16296() -> None:
    df = pl.DataFrame(
        {
            "a": [D("1.111")],
            "b": [D("2.22")],
            "c": [D("3.3")],
        },
        schema={
            "a": pl.Decimal(18, 3),
            "b": pl.Decimal(18, 2),
            "c": pl.Decimal(18, 1),
        },
    )

    assert (
        df.select(
            min=pl.min_horizontal(pl.col("a", "b", "c")),
            max=pl.max_horizontal(pl.col("a", "b", "c")),
            sum=pl.sum_horizontal(pl.col("a", "b", "c")),
        )
    ).to_dict(as_series=False) == {
        "min": [D("1.111")],
        "max": [D("3.300")],
        "sum": [D("6.631")],
    }


def test_shift_over_12957() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 1, 2, 2],
            "b": [D("1.1"), D("1.1"), D("2.2"), D("2.2")],
        }
    )
    result = df.select(
        x=pl.col("b").shift(1).over("a"),
        y=pl.col("a").shift(1).over("b"),
    )
    assert result["x"].to_list() == [None, D("1.1"), None, D("2.2")]
    assert result["y"].to_list() == [None, 1, None, 2]


def test_fill_null() -> None:
    s = pl.Series("a", [D("1.2"), None, D("1.4")])

    assert s.fill_null(D("0.0")).to_list() == [D("1.2"), D("0.0"), D("1.4")]
    assert s.fill_null(strategy="zero").to_list() == [D("1.2"), D("0.0"), D("1.4")]
    assert s.fill_null(strategy="max").to_list() == [D("1.2"), D("1.4"), D("1.4")]
    assert s.fill_null(strategy="min").to_list() == [D("1.2"), D("1.2"), D("1.4")]
    assert s.fill_null(strategy="one").to_list() == [D("1.2"), D("1.0"), D("1.4")]
    assert s.fill_null(strategy="forward").to_list() == [D("1.2"), D("1.2"), D("1.4")]
    assert s.fill_null(strategy="backward").to_list() == [D("1.2"), D("1.4"), D("1.4")]
    assert s.fill_null(strategy="mean").to_list() == [D("1.2"), D("1.3"), D("1.4")]


def test_unique() -> None:
    ser = pl.Series([D("1.1"), D("1.1"), D("2.2")])
    uniq = pl.Series([D("1.1"), D("2.2")])

    assert_series_equal(ser.unique(maintain_order=False), uniq, check_order=False)
    assert_series_equal(ser.unique(maintain_order=True), uniq)
    assert ser.n_unique() == 2
    assert ser.arg_unique().to_list() == [0, 2]


def test_groupby_agg_single_element_11232() -> None:
    data = {"g": [-1], "decimal": [-1]}
    schema = {"g": pl.Int64(), "decimal": pl.Decimal(38, 0)}
    result = (
        pl.LazyFrame(data, schema=schema)
        .group_by("g", maintain_order=True)
        .agg(pl.col("decimal").min())
        .collect()
    )
    expected = pl.DataFrame(data, schema=schema)
    assert_frame_equal(result, expected)


def test_decimal_from_large_ints_9084() -> None:
    numbers = [2963091539321097135000000000, 25658709114149718824803874]
    s = pl.Series(numbers, dtype=pl.Decimal(38, 0))
    assert s.to_list() == [D(n) for n in numbers]


def test_cast_float_to_decimal_12775() -> None:
    s = pl.Series([1.5])
    assert s.cast(pl.Decimal(scale=0)).to_list() == [D("2")]
    assert s.cast(pl.Decimal(scale=1)).to_list() == [D("1.5")]


def test_decimal_min_over_21096() -> None:
    df = pl.Series("x", [1, 2], pl.Decimal(scale=2)).to_frame()
    result = df.select(pl.col("x").min().over("x"))
    assert result["x"].to_list() == [D("1.00"), D("2.00")]


def test_decimal32_decimal64_22946() -> None:
    tbl = pa.Table.from_pydict(
        mapping={
            "colx": [D("100.1"), D("200.2"), D("300.3")],
            "coly": [D("400.4"), D("500.5"), D("600.6")],
        },
        schema=pa.schema(
            [
                ("colx", pa.decimal32(4, 1)),  # << note: decimal32
                ("coly", pa.decimal64(4, 1)),  # << note: decimal64
            ]
        ),
    )

    assert_frame_equal(
        pl.DataFrame(tbl),
        pl.DataFrame(
            [
                pl.Series(
                    "colx", [D("100.1"), D("200.2"), D("300.3")], pl.Decimal(4, 1)
                ),
                pl.Series(
                    "coly", [D("400.4"), D("500.5"), D("600.6")], pl.Decimal(4, 1)
                ),
            ]
        ),
    )


def test_decimal_cast_limit() -> None:
    fits = pl.Series([10**38 - 1, -(10**38 - 1)])
    assert_series_equal(fits.cast(pl.Decimal(38, 0)).cast(pl.Int128), fits)

    with pytest.raises(InvalidOperationError):
        fits.cast(pl.Decimal(39, 0))

    too_large1 = pl.Series([10**38])
    too_large2 = pl.Series([-(10**38)])
    with pytest.raises(InvalidOperationError):
        too_large1.cast(pl.Decimal(38, 0))
    with pytest.raises(InvalidOperationError):
        too_large2.cast(pl.Decimal(38, 0))


def test_decimal_agg() -> None:
    df = pl.DataFrame(
        {
            "g": [1, 1, 2, 2],
            "x": [1, 10, 100, 1000],
        }
    )
    ddf = df.with_columns(x=pl.col.x.cast(pl.Decimal(scale=3)))

    agg_exprs = {
        "min": pl.col.x.min(),
        "max": pl.col.x.max(),
        "mean": pl.col.x.mean(),
        "quantile": pl.col.x.quantile(0.4),
        "median": pl.col.x.median(),
        "sum": pl.col.x.sum(),
        "var": pl.col.x.var(),
        "std": pl.col.x.std(),
    }

    assert_frame_equal(
        df.select(**agg_exprs).cast(pl.Float64),
        ddf.select(**agg_exprs).cast(pl.Float64),
    )
    assert_frame_equal(
        df.group_by("g").agg(**agg_exprs).cast(pl.Float64),
        ddf.group_by("g").agg(**agg_exprs).cast(pl.Float64),
        check_row_order=False,
    )


def test_string_to_decimal_combined_prec_scale_24801() -> None:
    values = ["0.01", "10.0"]
    s = pl.Series(values).str.to_decimal()
    assert s.dtype == pl.Decimal(precision=4, scale=2)
    assert s.to_list() == [D(v) for v in values]


@pytest.mark.parametrize("maintain_order", [True, False])
def test_fallible_decimal_aggregated_with_filter(maintain_order: bool) -> None:
    df = pl.DataFrame(
        {"g": [10, 10, 20, 10], "a": [D("1.0"), D("0.0"), D("2.0"), D("1.0")]}
    )
    q = (
        df.lazy()
        .group_by("g", maintain_order=maintain_order)
        .agg(div=D("1.0") / pl.col.a.filter(pl.col.a != D("0.0")))
    )
    # must not raise an error
    out = q.collect()
    expected = pl.DataFrame({"g": [10, 20], "div": [[D("1.0"), D("1.0")], [D("0.5")]]})
    assert_frame_equal(out, expected, check_row_order=maintain_order)


def test_decimal_fits_too_large() -> None:
    with pytest.raises(pl.exceptions.InvalidOperationError):
        s = pl.Series([0, 2**128 - 10], dtype=pl.UInt128).cast(pl.Decimal(38, 0))
