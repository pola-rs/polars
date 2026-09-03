from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

import pytest
from dateutil.tz import tzoffset

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from polars._typing import PolarsDataType


def test_literal_scalar_list_18686() -> None:
    df = pl.DataFrame({"column1": [1, 2], "column2": ["A", "B"]})
    out = df.with_columns(lit1=pl.lit([]).cast(pl.List(pl.String)), lit2=pl.lit([]))

    assert out.to_dict(as_series=False) == {
        "column1": [1, 2],
        "column2": ["A", "B"],
        "lit1": [[], []],
        "lit2": [[], []],
    }
    assert out.schema == pl.Schema(
        [
            ("column1", pl.Int64),
            ("column2", pl.String),
            ("lit1", pl.List(pl.String)),
            ("lit2", pl.List(pl.Null)),
        ]
    )


def test_literal_integer_20807() -> None:
    for i in range(100):
        value = 2**i
        assert pl.select(pl.lit(value)).item() == value
        assert pl.select(pl.lit(-value)).item() == -value
        assert pl.select(pl.lit(value, dtype=pl.Int128)).item() == value
        assert pl.select(pl.lit(-value, dtype=pl.Int128)).item() == -value


@pytest.mark.parametrize(
    ("tz", "lit_dtype"),
    [
        (ZoneInfo("Asia/Kabul"), None),
        (ZoneInfo("Asia/Kabul"), pl.Datetime("us", "Asia/Kabul")),
        (ZoneInfo("Europe/Paris"), pl.Datetime("us", "Europe/Paris")),
        (timezone.utc, pl.Datetime("us", "UTC")),
    ],
)
def test_literal_datetime_timezone(tz: Any, lit_dtype: pl.DataType | None) -> None:
    expected_dtype = pl.Datetime("us", time_zone=str(tz))
    value = datetime(2020, 1, 1, tzinfo=tz)

    df1 = pl.DataFrame({"dt": [value]})
    df2 = pl.select(dt=pl.lit(value, dtype=lit_dtype))

    assert_frame_equal(df1, df2)
    assert df1.schema["dt"] == expected_dtype
    assert df1.item() == value


@pytest.mark.parametrize(
    ("tz", "lit_dtype", "expected_item"),
    [
        (
            # fixed offset from UTC
            tzoffset(None, 16200),
            None,
            datetime(2019, 12, 31, 19, 30, tzinfo=timezone.utc),
        ),
        (
            # fixed offset from UTC
            tzoffset("Kabul", 16200),
            None,
            datetime(2019, 12, 31, 19, 30, tzinfo=ZoneInfo("UTC")),
        ),
        (
            # fixed offset from UTC with matching timezone
            tzoffset(None, 16200),
            pl.Datetime("us", "Asia/Kabul"),
            datetime(2020, 1, 1, tzinfo=ZoneInfo("Asia/Kabul")),
        ),
        (
            # fixed offset from UTC with matching timezone
            tzoffset("Kabul", 16200),
            pl.Datetime("us", "Asia/Kabul"),
            datetime(2020, 1, 1, tzinfo=ZoneInfo("Asia/Kabul")),
        ),
    ],
)
def test_literal_datetime_timezone_utc_offset(
    tz: Any, lit_dtype: pl.DataType | None, expected_item: datetime
) -> None:
    overrides = {"schema_overrides": {"dt": lit_dtype}} if lit_dtype else {}
    value = datetime(2020, 1, 1, tzinfo=tz)

    # validate both frame and lit constructors
    df1 = pl.DataFrame({"dt": [value]}, **overrides)  # type: ignore[arg-type]
    df2 = pl.select(dt=pl.lit(value, dtype=lit_dtype))

    assert_frame_equal(df1, df2)

    expected_tz = "UTC" if lit_dtype is None else getattr(lit_dtype, "time_zone", None)
    expected_dtype = pl.Datetime("us", time_zone=expected_tz)

    for df in (df1, df2):
        assert df.schema["dt"] == expected_dtype
        assert df.item() == expected_item


def test_literal_datetime_timezone_utc_error() -> None:
    value = datetime(2020, 1, 1, tzinfo=tzoffset("Somewhere", offset=3600))

    with pytest.raises(
        TypeError,
        match=(
            r"time zone of dtype \('Pacific/Galapagos'\) differs from"
            r" time zone of value \(tzoffset\('Somewhere', 3600\)\)"
        ),
    ):
        # the offset does not correspond to the offset of the declared timezone
        pl.select(dt=pl.lit(value, dtype=pl.Datetime(time_zone="Pacific/Galapagos")))


def test_literal_object_25679() -> None:
    df = pl.DataFrame(
        data={"colx": [0, 0, 1, 2, None, 3, 3, None]},
        schema={"colx": pl.Object},
    )
    obj_zero = pl.lit(0, dtype=pl.Object())
    res = df.select(pl.col("colx").fill_null(obj_zero))

    assert res.schema == {"colx": pl.Object()}
    assert res["colx"].to_list() == [0, 0, 1, 2, 0, 3, 3, 0]


@pytest.mark.parametrize(
    ("numerator", "divisor", "floordiv", "mod"),
    [
        (10, 3, 3, 1),
        (10, 2, 5, 0),
        (1, 2, 0, 1),
        (0, 10, 0, 0),
        (1, 0, None, None),
    ],
)
@pytest.mark.parametrize(
    ("numerator_dtype", "divisor_dtype"),
    [
        (x, y)
        for x in [pl.Int8, pl.UInt8, None]
        for y in [pl.Int8, pl.UInt8, None]
        if x == y or x is None or y is None
    ],
)
def test_floordiv_mod(
    numerator: int,
    divisor: int,
    floordiv: int | None,
    mod: int | None,
    numerator_dtype: PolarsDataType | None,
    divisor_dtype: PolarsDataType | None,
) -> None:
    assert_frame_equal(
        pl.select(
            pl.lit(numerator, dtype=numerator_dtype)
            // pl.lit(divisor, dtype=divisor_dtype)
        ),
        pl.DataFrame(
            {"literal": [floordiv]},
            schema={"literal": numerator_dtype or divisor_dtype or pl.Int32},
        ),
    )
    assert_frame_equal(
        pl.select(
            pl.lit(numerator, dtype=numerator_dtype)
            % pl.lit(divisor, dtype=divisor_dtype)
        ),
        pl.DataFrame(
            {"literal": [mod]},
            schema={"literal": numerator_dtype or divisor_dtype or pl.Int32},
        ),
    )
