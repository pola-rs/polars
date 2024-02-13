from __future__ import annotations

import io
import itertools
import operator
from dataclasses import dataclass
from decimal import Decimal as D
from typing import Any, Callable, NamedTuple

import pytest

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal


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


@pytest.mark.slow()
def test_series_from_pydecimal_and_ints(
    permutations_int_dec_none: list[tuple[D | int | None, ...]],
) -> None:
    # TODO: check what happens if there are strings, floats arrow scalars in the list
    for data in permutations_int_dec_none:
        s = pl.Series("name", data)
        assert s.dtype == pl.Decimal(scale=7)  # inferred scale = 7, precision = None
        assert s.dtype.is_decimal()
        assert s.name == "name"
        assert s.null_count() == 1
        for i, d in enumerate(data):
            assert s[i] == d
        assert s.to_list() == [D(x) if x is not None else None for x in data]


@pytest.mark.slow()
def test_frame_from_pydecimal_and_ints(
    permutations_int_dec_none: list[tuple[D | int | None, ...]], monkeypatch: Any
) -> None:
    monkeypatch.setenv("POLARS_ACTIVATE_DECIMAL", "1")

    class X(NamedTuple):
        a: int | D | None

    @dataclass
    class Y:
        a: int | D | None

    for data in permutations_int_dec_none:
        row_data = [(d,) for d in data]
        for cls in (X, Y):
            for ctor in (pl.DataFrame, pl.from_records):
                df = ctor(data=list(map(cls, data)))  # type: ignore[operator]
                assert df.schema == {
                    "a": pl.Decimal(scale=7),
                }
                assert df.rows() == row_data


@pytest.mark.parametrize(
    ("trim_zeros", "expected"),
    [
        (True, "0.01"),
        (False, "0.010000000000000000000000000"),
    ],
)
def test_to_from_pydecimal_and_format(trim_zeros: bool, expected: str) -> None:
    dec_strs = [
        "0",
        "-1",
        expected,
        "-1.123801239123981293891283123",
        "12345678901.234567890123458390192857685",
        "-99999999999.999999999999999999999999999",
    ]
    with pl.Config(trim_decimal_zeros=trim_zeros):
        formatted = (
            str(pl.Series(list(map(D, dec_strs))))
            .split("[", 1)[1]
            .split("\n", 1)[1]
            .strip()[1:-1]
            .split()
        )
        assert formatted == dec_strs


def test_init_decimal_dtype() -> None:
    s = pl.Series("a", [D("-0.01"), D("1.2345678"), D("500")], dtype=pl.Decimal)
    assert s.dtype.is_numeric()

    df = pl.DataFrame(
        {"a": [D("-0.01"), D("1.2345678"), D("500")]}, schema={"a": pl.Decimal}
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
    result = pl.DataFrame([[D("45.0000")], [D("45.0000")]], schema={"a": pl.Float64})
    expected = pl.DataFrame({"a": [45.0, 45.0]})
    assert_frame_equal(result, expected)


def test_decimal_cast() -> None:
    df = pl.DataFrame(
        {
            "decimals": [D("2"), D("2"), D("-1.5")],
        }
    )

    result = df.with_columns(pl.col("decimals").cast(pl.Float32).alias("b2"))
    expected = {"decimals": [D("2"), D("2"), D("-1.5")], "b2": [2.0, 2.0, -1.5]}
    assert result.to_dict(as_series=False) == expected


def test_decimal_cast_no_scale() -> None:
    s = pl.Series().cast(pl.Decimal)
    assert s.dtype == pl.Decimal(precision=None, scale=0)

    s = pl.Series([D("10.0")]).cast(pl.Decimal)
    assert s.dtype == pl.Decimal(precision=None, scale=1)


def test_decimal_scale_precision_roundtrip(monkeypatch: Any) -> None:
    monkeypatch.setenv("POLARS_ACTIVATE_DECIMAL", "1")
    assert pl.from_arrow(pl.Series("dec", [D("10.0")]).to_arrow()).item() == D("10.0")


def test_string_to_decimal() -> None:
    s = pl.Series(
        [
            "40.12",
            "3420.13",
            "120134.19",
            "3212.98",
            "12.90",
            "143.09",
            "143.9",
            "-62.44",
        ]
    ).str.to_decimal()
    assert s.dtype == pl.Decimal(scale=2)

    assert s.to_list() == [
        D("40.12"),
        D("3420.13"),
        D("120134.19"),
        D("3212.98"),
        D("12.90"),
        D("143.09"),
        D("143.90"),
        D("-62.44"),
    ]


def test_read_csv_decimal(monkeypatch: Any) -> None:
    monkeypatch.setenv("POLARS_ACTIVATE_DECIMAL", "1")
    csv = """a,b
123.12,a
1.1,a
0.01,a"""

    df = pl.read_csv(csv.encode(), dtypes={"a": pl.Decimal(scale=2)})
    assert df.dtypes == [pl.Decimal(precision=None, scale=2), pl.String]
    assert df["a"].to_list() == [
        D("123.12"),
        D("1.10"),
        D("0.01"),
    ]


def test_decimal_eq_number() -> None:
    a = pl.Series([D("1.5"), D("22.25"), D("10.0")], dtype=pl.Decimal)
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
        [None, D("1.2"), D("2.13"), D("4.99"), D("2.13"), D("1.2")], dtype=pl.Decimal
    )
    s2 = pl.Series([None, D("1.200"), D("2.13"), D("4.99"), D("4.99"), D("2.13")])

    assert_series_equal(op(s, s2), expected)


def test_decimal_arithmetic() -> None:
    df = pl.DataFrame(
        {
            "a": [D("0.1"), D("10.1"), D("100.01")],
            "b": [D("20.1"), D("10.19"), D("39.21")],
        }
    )
    dt = pl.Decimal(20, 10)

    out = df.select(
        out1=pl.col("a") * pl.col("b"),
        out2=pl.col("a") + pl.col("b"),
        out3=pl.col("a") / pl.col("b"),
        out4=pl.col("a") - pl.col("b"),
        out5=pl.col("a").cast(dt) / pl.col("b").cast(dt),
    )
    assert out.dtypes == [
        pl.Decimal(precision=None, scale=4),
        pl.Decimal(precision=None, scale=2),
        pl.Decimal(precision=None, scale=6),
        pl.Decimal(precision=None, scale=2),
        pl.Decimal(precision=None, scale=14),
    ]

    assert out.to_dict(as_series=False) == {
        "out1": [D("2.0100"), D("102.9190"), D("3921.3921")],
        "out2": [D("20.20"), D("20.29"), D("139.22")],
        "out3": [D("0.004975"), D("0.991167"), D("2.550624")],
        "out4": [D("-20.00"), D("-0.09"), D("60.80")],
        "out5": [D("0.00497512437810"), D("0.99116781157998"), D("2.55062484060188")],
    }


def test_decimal_series_value_arithmetic() -> None:
    s = pl.Series([D("0.10"), D("10.10"), D("100.01")])

    out1 = s + 10
    out2 = s + D("10")
    out3 = s + D("10.0001")
    out4 = s * 2 / 3
    out5 = s / D("1.5")
    out6 = s - 5

    assert out1.dtype == pl.Decimal(precision=None, scale=2)
    assert out2.dtype == pl.Decimal(precision=None, scale=2)
    assert out3.dtype == pl.Decimal(precision=None, scale=4)
    assert out4.dtype == pl.Decimal(precision=None, scale=6)
    assert out5.dtype == pl.Decimal(precision=None, scale=6)
    assert out6.dtype == pl.Decimal(precision=None, scale=2)

    assert out1.to_list() == [D("10.1"), D("20.1"), D("110.01")]
    assert out2.to_list() == [D("10.1"), D("20.1"), D("110.01")]
    assert out3.to_list() == [D("10.1001"), D("20.1001"), D("110.0101")]
    assert out4.to_list() == [
        D("0.066666"),
        D("6.733333"),
        D("66.673333"),
    ]  # TODO: do we want floor instead of round?
    assert out5.to_list() == [D("0.066666"), D("6.733333"), D("66.673333")]
    assert out6.to_list() == [D("-4.9"), D("5.1"), D("95.01")]


def test_decimal_aggregations() -> None:
    df = pl.DataFrame(
        {
            "g": [1, 1, 2, 2],
            "a": [D("0.1"), D("10.1"), D("100.01"), D("9000.12")],
        }
    )

    assert df.group_by("g", maintain_order=True).agg(
        sum=pl.sum("a"),
        min=pl.min("a"),
        max=pl.max("a"),
    ).to_dict(as_series=False) == {
        "g": [1, 2],
        "sum": [D("10.20"), D("9100.13")],
        "min": [D("0.10"), D("100.01")],
        "max": [D("10.10"), D("9000.12")],
    }

    assert df.select(
        sum=pl.sum("a"),
        min=pl.min("a"),
        max=pl.max("a"),
    ).to_dict(as_series=False) == {
        "sum": [D("9110.33")],
        "min": [D("0.10")],
        "max": [D("9000.12")],
    }


def test_decimal_in_filter() -> None:
    df = pl.DataFrame(
        {
            "foo": [1, 2, 3],
            "bar": ["6", "7", "8"],
        }
    )
    df = df.with_columns(pl.col("bar").cast(pl.Decimal))
    assert df.filter(pl.col("foo") > 1).to_dict(as_series=False) == {
        "foo": [2, 3],
        "bar": [D("7"), D("8")],
    }


def test_decimal_write_parquet_12375() -> None:
    f = io.BytesIO()
    df = pl.DataFrame(
        {"hi": [True, False, True, False], "bye": [1, 2, 3, D(47283957238957239875)]}
    )
    assert df["bye"].dtype == pl.Decimal

    df.write_parquet(f)


def test_decimal_list_get_13847() -> None:
    with pl.Config() as cfg:
        cfg.activate_decimals()
        df = pl.DataFrame({"a": [[D("1.1"), D("1.2")], [D("2.1")]]})
        out = df.select(pl.col("a").list.get(0))
        expected = pl.DataFrame({"a": [D("1.1"), D("2.1")]})
        assert_frame_equal(out, expected)
