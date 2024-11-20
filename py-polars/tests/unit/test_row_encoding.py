from __future__ import annotations

import pytest
from hypothesis import given

import polars as pl
from polars.testing import assert_frame_equal
from polars.testing.parametric import dataframes

# @TODO: Deal with no_order
FIELD_COMBS = [
    (descending, nulls_last, False)
    for descending in [False, True]
    for nulls_last in [False, True]
]


def roundtrip_re(
    df: pl.DataFrame, fields: list[tuple[bool, bool, bool]] | None = None
) -> None:
    if fields is None:
        fields = [(False, False, False)] * df.width

    row_encoded = df._row_encode(fields)
    dtypes = [(c, df.get_column(c).dtype) for c in df.columns]
    result = row_encoded._row_decode(dtypes, fields)

    assert_frame_equal(df, result)


@given(
    df=dataframes(
        excluded_dtypes=[
            pl.Array,
            pl.Struct,
            pl.Categorical,
            pl.Enum,
        ]
    )
)
@pytest.mark.parametrize("field", FIELD_COMBS)
def test_row_encoding_parametric(
    df: pl.DataFrame, field: tuple[bool, bool, bool]
) -> None:
    roundtrip_re(df, [field] * df.width)


@pytest.mark.parametrize("field", FIELD_COMBS)
def test_nulls(field: tuple[bool, bool, bool]) -> None:
    roundtrip_re(pl.Series("a", [], pl.Null).to_frame(), [field])
    roundtrip_re(pl.Series("a", [None], pl.Null).to_frame(), [field])
    roundtrip_re(pl.Series("a", [None] * 2, pl.Null).to_frame(), [field])
    roundtrip_re(pl.Series("a", [None] * 13, pl.Null).to_frame(), [field])
    roundtrip_re(pl.Series("a", [None] * 42, pl.Null).to_frame(), [field])


@pytest.mark.parametrize("field", FIELD_COMBS)
def test_bool(field: tuple[bool, bool, bool]) -> None:
    roundtrip_re(pl.Series("a", [], pl.Boolean).to_frame(), [field])
    roundtrip_re(pl.Series("a", [False], pl.Boolean).to_frame(), [field])
    roundtrip_re(pl.Series("a", [True], pl.Boolean).to_frame(), [field])
    roundtrip_re(pl.Series("a", [False, True], pl.Boolean).to_frame(), [field])
    roundtrip_re(pl.Series("a", [True, False], pl.Boolean).to_frame(), [field])


@pytest.mark.parametrize(
    "dtype",
    [
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
    ],
)
@pytest.mark.parametrize("field", FIELD_COMBS)
def test_int(dtype: pl.DataType, field: tuple[bool, bool, bool]) -> None:
    min = pl.select(x=dtype.min()).item()  # type: ignore[attr-defined]
    max = pl.select(x=dtype.max()).item()  # type: ignore[attr-defined]

    roundtrip_re(pl.Series("a", [], dtype).to_frame(), [field])
    roundtrip_re(pl.Series("a", [0], dtype).to_frame(), [field])
    roundtrip_re(pl.Series("a", [min], dtype).to_frame(), [field])
    roundtrip_re(pl.Series("a", [max], dtype).to_frame(), [field])

    roundtrip_re(pl.Series("a", [1, 2, 3], dtype).to_frame(), [field])
    roundtrip_re(pl.Series("a", [0, 1, 2, 3], dtype).to_frame(), [field])
    roundtrip_re(pl.Series("a", [min, 0, max], dtype).to_frame(), [field])


@pytest.mark.parametrize(
    "dtype",
    [
        pl.Float32,
        pl.Float64,
    ],
)
@pytest.mark.parametrize("field", FIELD_COMBS)
def test_float(dtype: pl.DataType, field: tuple[bool, bool, bool]) -> None:
    inf = float("inf")
    inf_b = float("-inf")

    roundtrip_re(pl.Series("a", [], dtype).to_frame(), [field])
    roundtrip_re(pl.Series("a", [0.0], dtype).to_frame(), [field])
    roundtrip_re(pl.Series("a", [inf], dtype).to_frame(), [field])
    roundtrip_re(pl.Series("a", [-inf_b], dtype).to_frame(), [field])

    roundtrip_re(pl.Series("a", [1.0, 2.0, 3.0], dtype).to_frame(), [field])
    roundtrip_re(pl.Series("a", [0.0, 1.0, 2.0, 3.0], dtype).to_frame(), [field])
    roundtrip_re(pl.Series("a", [inf, 0, -inf_b], dtype).to_frame(), [field])


@pytest.mark.parametrize("field", FIELD_COMBS)
def test_str(field: tuple[bool, bool, bool]) -> None:
    roundtrip_re(pl.Series("a", [], pl.String).to_frame(), [field])
    roundtrip_re(pl.Series("a", [""], pl.String).to_frame(), [field])

    roundtrip_re(pl.Series("a", ["a", "b", "c"], pl.String).to_frame(), [field])
    roundtrip_re(pl.Series("a", ["", "a", "b", "c"], pl.String).to_frame(), [field])

    roundtrip_re(
        pl.Series("a", ["different", "length", "strings"], pl.String).to_frame(),
        [field],
    )
    roundtrip_re(
        pl.Series(
            "a", ["different", "", "length", "", "strings"], pl.String
        ).to_frame(),
        [field],
    )


@pytest.mark.parametrize("field", FIELD_COMBS)
def test_struct(field: tuple[bool, bool, bool]) -> None:
    roundtrip_re(pl.Series("a", [], pl.Struct({})).to_frame())
    roundtrip_re(pl.Series("a", [{}], pl.Struct({})).to_frame())
    roundtrip_re(
        pl.Series("a", [{"x": 1}], pl.Struct({"x": pl.Int32})).to_frame(), [field]
    )
    roundtrip_re(
        pl.Series(
            "a", [{"x": 1}, {"y": 2}], pl.Struct({"x": pl.Int32, "y": pl.Int32})
        ).to_frame(),
        [field],
    )


@pytest.mark.parametrize("field", FIELD_COMBS)
def test_list(field: tuple[bool, bool, bool]) -> None:
    dtype = pl.List(pl.Int32)
    roundtrip_re(pl.Series("a", [], dtype).to_frame(), [field])
    roundtrip_re(pl.Series("a", [[]], dtype).to_frame(), [field])
    roundtrip_re(pl.Series("a", [[1], [2]], dtype).to_frame(), [field])
    roundtrip_re(pl.Series("a", [[1, 2], [3]], dtype).to_frame(), [field])
    roundtrip_re(pl.Series("a", [[1, 2], [], [3]], dtype).to_frame(), [field])
    roundtrip_re(pl.Series("a", [None, [1, 2], None, [], [3]], dtype).to_frame(), [field])

    dtype = pl.List(pl.String)
    roundtrip_re(pl.Series("a", [], dtype).to_frame(), [field])
    roundtrip_re(pl.Series("a", [[]], dtype).to_frame(), [field])
    roundtrip_re(pl.Series("a", [[""], [""]], dtype).to_frame(), [field])
    roundtrip_re(pl.Series("a", [["abc"], ["xyzw"]], dtype).to_frame(), [field])
    roundtrip_re(pl.Series("a", [["x", "yx"], ["abc"]], dtype).to_frame(), [field])
    roundtrip_re(pl.Series("a", [["wow", "this is"], [], ["cool"]], dtype).to_frame(), [field])
    roundtrip_re(pl.Series("a", [None, ["very", "very"], None, [], ["cool"]], dtype).to_frame(), [field])
