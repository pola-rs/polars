from __future__ import annotations

import io
from datetime import date, datetime, time, timedelta, timezone
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import polars as pl
from polars.exceptions import (
    ComputeError,
    DuplicateError,
    InvalidOperationError,
    PanicException,
    UnstableWarning,
)
from polars.interchange.protocol import CompatLevel
from polars.testing import assert_frame_equal, assert_series_equal
from tests.unit.utils.pycapsule_utils import PyCapsuleStreamHolder

if TYPE_CHECKING:
    from tests.conftest import PlMonkeyPatch


def test_arrow_list_roundtrip() -> None:
    # https://github.com/pola-rs/polars/issues/1064
    tbl = pa.table({"a": [1], "b": [[1, 2]]})
    arw = pl.from_arrow(tbl).to_arrow()

    assert arw.shape == tbl.shape
    assert arw.schema.names == tbl.schema.names
    for c1, c2 in zip(arw.columns, tbl.columns, strict=True):
        assert c1.to_pylist() == c2.to_pylist()


def test_arrow_null_roundtrip() -> None:
    tbl = pa.table({"a": [None, None], "b": [[None, None], [None, None]]})
    df = pl.from_arrow(tbl)

    if isinstance(df, pl.DataFrame):
        assert df.dtypes == [pl.Null, pl.List(pl.Null)]

    arw = df.to_arrow()

    assert arw.shape == tbl.shape
    assert arw.schema.names == tbl.schema.names
    for c1, c2 in zip(arw.columns, tbl.columns, strict=True):
        assert c1.to_pylist() == c2.to_pylist()


def test_arrow_empty_dataframe() -> None:
    # 0x0 dataframe
    df = pl.DataFrame({})
    tbl = pa.table({})
    assert df.to_arrow() == tbl
    df2 = cast("pl.DataFrame", pl.from_arrow(df.to_arrow()))
    assert_frame_equal(df2, df)

    # 0 row dataframe
    df = pl.DataFrame({}, schema={"a": pl.Int32})
    tbl = pa.Table.from_batches([], pa.schema([pa.field("a", pa.int32())]))
    assert df.to_arrow() == tbl
    df2 = cast("pl.DataFrame", pl.from_arrow(df.to_arrow()))
    assert df2.schema == {"a": pl.Int32}
    assert df2.shape == (0, 1)


def test_arrow_dict_to_polars() -> None:
    pa_dict = pa.DictionaryArray.from_arrays(
        indices=np.array([0, 1, 2, 3, 1, 0, 2, 3, 3, 2]),
        dictionary=np.array(["AAA", "BBB", "CCC", "DDD"]),
    ).cast(pa.large_utf8())

    s = pl.Series(
        name="pa_dict",
        values=["AAA", "BBB", "CCC", "DDD", "BBB", "AAA", "CCC", "DDD", "DDD", "CCC"],
    )
    assert_series_equal(s, pl.Series("pa_dict", pa_dict))


def test_arrow_list_chunked_array() -> None:
    a = pa.array([[1, 2], [3, 4]])
    ca = pa.chunked_array([a, a, a])
    s = cast("pl.Series", pl.from_arrow(ca))
    assert s.dtype == pl.List


# Test that polars convert Arrays of logical types correctly to arrow
def test_arrow_array_logical() -> None:
    # cast to large string and uint8 indices because polars converts to those
    pa_data1 = (
        pa.array(["a", "b", "c", "d"])
        .dictionary_encode()
        .cast(pa.dictionary(pa.uint8(), pa.large_string()))
    )
    pa_array_logical1 = pa.FixedSizeListArray.from_arrays(pa_data1, 2)

    s1 = pl.Series(
        values=[["a", "b"], ["c", "d"]],
        dtype=pl.Array(pl.Enum(["a", "b", "c", "d"]), shape=2),
    )
    assert s1.to_arrow() == pa_array_logical1

    pa_data2 = pa.array([date(2024, 1, 1), date(2024, 1, 2)])
    pa_array_logical2 = pa.FixedSizeListArray.from_arrays(pa_data2, 1)

    s2 = pl.Series(
        values=[[date(2024, 1, 1)], [date(2024, 1, 2)]],
        dtype=pl.Array(pl.Date, shape=1),
    )
    assert s2.to_arrow() == pa_array_logical2


def test_from_dict() -> None:
    data = {"a": [1, 2], "b": [3, 4]}
    df = pl.from_dict(data)
    assert df.shape == (2, 2)
    for s1, s2 in zip(
        list(df), [pl.Series("a", [1, 2]), pl.Series("b", [3, 4])], strict=True
    ):
        assert_series_equal(s1, s2)


def test_from_dict_struct() -> None:
    data: dict[str, dict[str, list[int]] | list[int]] = {
        "a": {"b": [1, 3], "c": [2, 4]},
        "d": [5, 6],
    }
    df = pl.from_dict(data)
    assert df.shape == (2, 2)
    assert df["a"][0] == {"b": 1, "c": 2}
    assert df["a"][1] == {"b": 3, "c": 4}
    assert df.schema == {"a": pl.Struct({"b": pl.Int64, "c": pl.Int64}), "d": pl.Int64}


def test_from_dicts() -> None:
    data = [{"a": 1, "b": 4}, {"a": 2, "b": 5}, {"a": 3, "b": None}]
    df = pl.from_dicts(data)  # type: ignore[arg-type]
    assert df.shape == (3, 2)
    assert df.rows() == [(1, 4), (2, 5), (3, None)]
    assert df.schema == {"a": pl.Int64, "b": pl.Int64}


def test_from_dict_no_inference() -> None:
    schema = {"a": pl.String}
    data = [{"a": "aa"}]
    df = pl.from_dicts(data, schema_overrides=schema, infer_schema_length=0)
    assert df.schema == schema
    assert df.to_dicts() == data


def test_from_dicts_schema_override() -> None:
    schema = {
        "a": pl.String,
        "b": pl.Int64,
        "c": pl.List(pl.Struct({"x": pl.Int64, "y": pl.String, "z": pl.Float64})),
    }

    # initial data matches the expected schema
    data1 = [
        {
            "a": "l",
            "b": i,
            "c": [{"x": (j + 2), "y": "?", "z": (j % 2)} for j in range(2)],
        }
        for i in range(5)
    ]

    # extend with a mix of fields that are/not in the schema
    data2 = [{"b": i + 5, "d": "ABC", "e": "DEF"} for i in range(5)]

    for n_infer in (0, 3, 5, 8, 10, 100):
        df = pl.DataFrame(
            data=(data1 + data2),
            schema=schema,  # type: ignore[arg-type]
            infer_schema_length=n_infer,
        )
        assert df.schema == schema
        assert df.rows() == [
            ("l", 0, [{"x": 2, "y": "?", "z": 0.0}, {"x": 3, "y": "?", "z": 1.0}]),
            ("l", 1, [{"x": 2, "y": "?", "z": 0.0}, {"x": 3, "y": "?", "z": 1.0}]),
            ("l", 2, [{"x": 2, "y": "?", "z": 0.0}, {"x": 3, "y": "?", "z": 1.0}]),
            ("l", 3, [{"x": 2, "y": "?", "z": 0.0}, {"x": 3, "y": "?", "z": 1.0}]),
            ("l", 4, [{"x": 2, "y": "?", "z": 0.0}, {"x": 3, "y": "?", "z": 1.0}]),
            (None, 5, None),
            (None, 6, None),
            (None, 7, None),
            (None, 8, None),
            (None, 9, None),
        ]


def test_from_dicts_struct() -> None:
    data = [{"a": {"b": 1, "c": 2}, "d": 5}, {"a": {"b": 3, "c": 4}, "d": 6}]
    df = pl.from_dicts(data)
    assert df.shape == (2, 2)
    assert df["a"][0] == {"b": 1, "c": 2}
    assert df["a"][1] == {"b": 3, "c": 4}

    # 5649
    assert pl.from_dicts([{"a": [{"x": 1}]}, {"a": [{"y": 1}]}]).to_dict(
        as_series=False
    ) == {"a": [[{"y": None, "x": 1}], [{"y": 1, "x": None}]]}
    assert pl.from_dicts([{"a": [{"x": 1}, {"y": 2}]}, {"a": [{"y": 1}]}]).to_dict(
        as_series=False
    ) == {"a": [[{"y": None, "x": 1}, {"y": 2, "x": None}], [{"y": 1, "x": None}]]}


def test_from_records() -> None:
    data = [[1, 2, 3], [4, 5, 6]]
    df = pl.from_records(data, schema=["a", "b"])
    assert df.shape == (3, 2)
    assert df.rows() == [(1, 4), (2, 5), (3, 6)]


# https://github.com/pola-rs/polars/issues/15195
@pytest.mark.parametrize(
    "input",
    [
        pl.Series([1, 2]),
        pl.Series([{"a": 1, "b": 2}]),
        pl.DataFrame({"a": [1, 2], "b": [3, 4]}),
    ],
)
def test_from_records_non_sequence_input(input: Any) -> None:
    with pytest.raises(TypeError, match="expected data of type Sequence"):
        pl.from_records(input)


def test_from_arrow() -> None:
    data = pa.table({"a": [1, 2, 3], "b": [4, 5, 6]})
    df = pl.from_arrow(data)
    assert df.shape == (3, 2)
    assert df.rows() == [(1, 4), (2, 5), (3, 6)]  # type: ignore[union-attr]

    # if not a PyArrow type, raise a TypeError
    with pytest.raises(TypeError):
        _ = pl.from_arrow([1, 2])

    df = pl.from_arrow(
        data, schema=["a", "b"], schema_overrides={"a": pl.UInt32, "b": pl.UInt64}
    )
    assert df.rows() == [(1, 4), (2, 5), (3, 6)]  # type: ignore[union-attr]
    assert df.schema == {"a": pl.UInt32, "b": pl.UInt64}  # type: ignore[union-attr]


def test_from_arrow_with_bigquery_metadata() -> None:
    arrow_schema = pa.schema(
        [
            pa.field("id", pa.int64()).with_metadata(
                {"ARROW:extension:name": "google:sqlType:integer"}
            ),
            pa.field(
                "misc",
                pa.struct([("num", pa.int32()), ("val", pa.string())]),
            ).with_metadata({"ARROW:extension:name": "google:sqlType:struct"}),
        ]
    )
    arrow_tbl = pa.Table.from_pylist(
        [{"id": 1, "misc": None}, {"id": 2, "misc": None}],
        schema=arrow_schema,
    )

    expected_data = {"id": [1, 2], "num": [None, None], "val": [None, None]}
    expected_schema = {"id": pl.Int64, "num": pl.Int32, "val": pl.String}
    assert_frame_equal(
        pl.DataFrame(expected_data, schema=expected_schema),
        pl.from_arrow(arrow_tbl).unnest("misc"),  # type: ignore[union-attr]
    )


def test_from_optional_not_available() -> None:
    from polars._dependencies import _LazyModule

    # proxy module is created dynamically if the required module is not available
    # (see the polars._dependencies source code for additional detail/comments)

    np = _LazyModule("numpy", module_available=False)
    with pytest.raises(ImportError, match=r"np\.array requires 'numpy'"):
        pl.from_numpy(np.array([[1, 2], [3, 4]]), schema=["a", "b"])

    pa = _LazyModule("pyarrow", module_available=False)
    with pytest.raises(ImportError, match=r"pa\.table requires 'pyarrow'"):
        pl.from_arrow(pa.table({"a": [1, 2, 3], "b": [4, 5, 6]}))

    pd = _LazyModule("pandas", module_available=False)
    with pytest.raises(ImportError, match=r"pd\.Series requires 'pandas'"):
        pl.from_pandas(pd.Series([1, 2, 3]))


def test_upcast_pyarrow_dicts() -> None:
    # https://github.com/pola-rs/polars/issues/1752
    tbls = [
        pa.table(
            {
                "col_name": pa.array(
                    [f"value_{i}"], pa.dictionary(pa.int8(), pa.string())
                )
            }
        )
        for i in range(128)
    ]

    tbl = pa.concat_tables(tbls, promote_options="default")
    out = cast("pl.DataFrame", pl.from_arrow(tbl))
    assert out.shape == (128, 1)
    assert out["col_name"][0] == "value_0"
    assert out["col_name"][127] == "value_127"


def test_no_rechunk() -> None:
    table = pa.Table.from_pydict({"x": pa.chunked_array([list("ab"), list("cd")])})
    # table
    assert pl.from_arrow(table, rechunk=False).n_chunks() == 2
    # chunked array
    assert pl.from_arrow(table["x"], rechunk=False).n_chunks() == 2


def test_from_empty_arrow() -> None:
    df = cast("pl.DataFrame", pl.from_arrow(pa.table(pd.DataFrame({"a": [], "b": []}))))
    assert df.columns == ["a", "b"]
    assert df.dtypes == [pl.Float64, pl.Float64]

    # 2705
    df1 = pd.DataFrame(columns=["b"], dtype=float, index=pd.Index([]))
    tbl = pa.Table.from_pandas(df1)
    out = cast("pl.DataFrame", pl.from_arrow(tbl))
    assert out.columns == ["b", "__index_level_0__"]
    assert out.dtypes == [pl.Float64, pl.Null]
    tbl = pa.Table.from_pandas(df1, preserve_index=False)
    out = cast("pl.DataFrame", pl.from_arrow(tbl))
    assert out.columns == ["b"]
    assert out.dtypes == [pl.Float64]

    # 4568
    tbl = pa.table({"l": []}, schema=pa.schema([("l", pa.large_list(pa.uint8()))]))

    df = cast("pl.DataFrame", pl.from_arrow(tbl))
    assert df.schema["l"] == pl.List(pl.UInt8)


def test_cat_int_types_3500() -> None:
    # Create an enum / categorical / dictionary typed pyarrow array
    # Most simply done by creating a pandas categorical series first
    categorical_s = pd.Series(["a", "a", "b"], dtype="category")
    pyarrow_array = pa.Array.from_pandas(categorical_s)

    # The in-memory representation of each category can either be a signed or
    # unsigned 8-bit integer. Pandas uses Int8...
    int_dict_type = pa.dictionary(index_type=pa.int8(), value_type=pa.utf8())
    # ... while DuckDB uses UInt8
    uint_dict_type = pa.dictionary(index_type=pa.uint8(), value_type=pa.utf8())

    for t in [int_dict_type, uint_dict_type]:
        s = cast("pl.Series", pl.from_arrow(pyarrow_array.cast(t)))
        assert_series_equal(
            s, pl.Series(["a", "a", "b"]).cast(pl.Categorical), check_names=False
        )


def test_from_pyarrow_chunked_array() -> None:
    column = pa.chunked_array([[1], [2]])
    series = pl.Series("column", column)
    assert series.to_list() == [1, 2]


def test_arrow_list_null_5697() -> None:
    # Create a pyarrow table with a list[null] column.
    pa_table = pa.table([[[None]]], names=["mycol"])
    df = pl.from_arrow(pa_table)
    pa_table = df.to_arrow()
    # again to polars to test the schema
    assert pl.from_arrow(pa_table).schema == {"mycol": pl.List(pl.Null)}  # type: ignore[union-attr]


def test_from_pyarrow_map() -> None:
    pa_table = pa.table(
        [[1, 2], [[("a", "something")], [("a", "else"), ("b", "another key")]]],
        schema=pa.schema(
            [("idx", pa.int16()), ("mapping", pa.map_(pa.string(), pa.string()))]
        ),
    )

    # Convert from an empty table to trigger an ArrowSchema -> native schema
    # conversion (checks that ArrowDataType::Map is handled in Rust).
    pl.DataFrame(pa_table.slice(0, 0))

    result = pl.DataFrame(pa_table)
    assert result.to_dict(as_series=False) == {
        "idx": [1, 2],
        "mapping": [
            [{"key": "a", "value": "something"}],
            [{"key": "a", "value": "else"}, {"key": "b", "value": "another key"}],
        ],
    }


def test_from_fixed_size_binary_list() -> None:
    val = [[b"63A0B1C66575DD5708E1EB2B"]]
    arrow_array = pa.array(val, type=pa.list_(pa.binary(24)))
    s = cast("pl.Series", pl.from_arrow(arrow_array))
    assert s.dtype == pl.List(pl.Binary)
    assert s.to_list() == val


def test_dataframe_from_repr() -> None:
    # round-trip various types
    frame = (
        pl.LazyFrame(
            {
                "a": [1, 2, None],
                "b": [4.5, 5.23e13, -3.12e12],
                "c": ["x", "y", "z"],
                "d": [True, False, True],
                "e": [None, "", None],
                "f": [date(2022, 7, 5), date(2023, 2, 5), date(2023, 8, 5)],
                "g": [time(0, 0, 0, 1), time(12, 30, 45), time(23, 59, 59, 999000)],
                "h": [
                    datetime(2022, 7, 5, 10, 30, 45, 4560),
                    datetime(2023, 10, 12, 20, 3, 8, 11),
                    None,
                ],
            },
        )
        .with_columns(
            pl.col("c").cast(pl.Categorical),
            pl.col("h").cast(pl.Datetime("ns")),
        )
        .collect()
    )

    assert frame.schema == {
        "a": pl.Int64,
        "b": pl.Float64,
        "c": pl.Categorical(),
        "d": pl.Boolean,
        "e": pl.String,
        "f": pl.Date,
        "g": pl.Time,
        "h": pl.Datetime("ns"),
    }
    df = cast("pl.DataFrame", pl.from_repr(repr(frame)))
    assert_frame_equal(frame, df)

    # empty frame; confirm schema is inferred
    df = cast(
        "pl.DataFrame",
        pl.from_repr(
            """
            ┌─────┬─────┬─────┬─────┬─────┬───────┐
            │ id  ┆ q1  ┆ q2  ┆ q3  ┆ q4  ┆ total │
            │ --- ┆ --- ┆ --- ┆ --- ┆ --- ┆ ---   │
            │ str ┆ i8  ┆ i16 ┆ i32 ┆ i64 ┆ f64   │
            ╞═════╪═════╪═════╪═════╪═════╪═══════╡
            └─────┴─────┴─────┴─────┴─────┴───────┘
            """
        ),
    )
    assert df.shape == (0, 6)
    assert df.rows() == []
    assert df.schema == {
        "id": pl.String,
        "q1": pl.Int8,
        "q2": pl.Int16,
        "q3": pl.Int32,
        "q4": pl.Int64,
        "total": pl.Float64,
    }

    # empty frame with no dtypes
    df = cast(
        "pl.DataFrame",
        pl.from_repr(
            """
            ┌──────┬───────┐
            │ misc ┆ other │
            ╞══════╪═══════╡
            └──────┴───────┘
            """
        ),
    )
    assert_frame_equal(df, pl.DataFrame(schema={"misc": pl.String, "other": pl.String}))

    # empty frame with a non-standard/blank 'null' in numeric col
    df = cast(
        "pl.DataFrame",
        pl.from_repr(
            """
            ┌─────┬──────┐
            │ c1  ┆  c2  │
            │ --- ┆  --- │
            │ i32 ┆  f64 │
            ╞═════╪══════╡
            │     │ NULL │
            └─────┴──────┘
            """
        ),
    )
    assert_frame_equal(
        df,
        pl.DataFrame(
            data=[(None, None)],
            schema={"c1": pl.Int32, "c2": pl.Float64},
            orient="row",
        ),
    )

    df = cast(
        "pl.DataFrame",
        pl.from_repr(
            """
            # >>> Missing cols with old-style ellipsis, nulls, commented out
            # ┌────────────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬──────┐
            # │ dt         ┆ c1  ┆ c2  ┆ c3  ┆ ... ┆ c96 ┆ c97 ┆ c98 ┆ c99  │
            # │ ---        ┆ --- ┆ --- ┆ --- ┆     ┆ --- ┆ --- ┆ --- ┆ ---  │
            # │ date       ┆ i32 ┆ i32 ┆ i32 ┆     ┆ i64 ┆ i64 ┆ i64 ┆ i64  │
            # ╞════════════╪═════╪═════╪═════╪═════╪═════╪═════╪═════╪══════╡
            # │ 2023-03-25 ┆ 1   ┆ 2   ┆ 3   ┆ ... ┆ 96  ┆ 97  ┆ 98  ┆ 99   │
            # │ 1999-12-31 ┆ 3   ┆ 6   ┆ 9   ┆ ... ┆ 288 ┆ 291 ┆ 294 ┆ null │
            # │ null       ┆ 9   ┆ 18  ┆ 27  ┆ ... ┆ 864 ┆ 873 ┆ 882 ┆ 891  │
            # └────────────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴──────┘
            """
        ),
    )
    assert df.schema == {
        "dt": pl.Date,
        "c1": pl.Int32,
        "c2": pl.Int32,
        "c3": pl.Int32,
        "c96": pl.Int64,
        "c97": pl.Int64,
        "c98": pl.Int64,
        "c99": pl.Int64,
    }
    assert df.rows() == [
        (date(2023, 3, 25), 1, 2, 3, 96, 97, 98, 99),
        (date(1999, 12, 31), 3, 6, 9, 288, 291, 294, None),
        (None, 9, 18, 27, 864, 873, 882, 891),
    ]

    df = cast(
        "pl.DataFrame",
        pl.from_repr(
            """
            # >>> no dtypes:
            # ┌────────────┬──────┐
            # │ dt         ┆ c99  │
            # ╞════════════╪══════╡
            # │ 2023-03-25 ┆ 99   │
            # │ 1999-12-31 ┆ null │
            # │ null       ┆ 891  │
            # └────────────┴──────┘
            """
        ),
    )
    assert df.schema == {"dt": pl.Date, "c99": pl.Int64}
    assert df.rows() == [
        (date(2023, 3, 25), 99),
        (date(1999, 12, 31), None),
        (None, 891),
    ]

    df = cast(
        "pl.DataFrame",
        pl.from_repr(
            """
            In [2]: with pl.Config() as cfg:
               ...:     pl.Config.set_tbl_formatting("UTF8_FULL", rounded_corners=True)
               ...:     print(df)
               ...:
            shape: (1, 5)
            ╭───────────┬────────────┬───┬───────┬────────────────────────────────╮
            │ source_ac ┆ source_cha ┆ … ┆ ident ┆ timestamp                      │
            │ tor_id    ┆ nnel_id    ┆   ┆ ---   ┆ ---                            │
            │ ---       ┆ ---        ┆   ┆ str   ┆ datetime[μs, Asia/Tokyo]       │
            │ i32       ┆ i64        ┆   ┆       ┆                                │
            ╞═══════════╪════════════╪═══╪═══════╪════════════════════════════════╡
            │ 123456780 ┆ 9876543210 ┆ … ┆ a:b:c ┆ 2023-03-25 10:56:59.663053 JST │
            ├╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
            │ …         ┆ …          ┆ … ┆ …     ┆ …                              │
            ├╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
            │ 803065983 ┆ 2055938745 ┆ … ┆ x:y:z ┆ 2023-03-25 12:38:18.050545 JST │
            ╰───────────┴────────────┴───┴───────┴────────────────────────────────╯
            # "Een fluitje van een cent..." :)
            """
        ),
    )
    assert df.shape == (2, 4)
    assert df.schema == {
        "source_actor_id": pl.Int32,
        "source_channel_id": pl.Int64,
        "ident": pl.String,
        "timestamp": pl.Datetime("us", "Asia/Tokyo"),
    }


def test_dataframe_from_repr_24110() -> None:
    df = cast(
        "pl.DataFrame",
        pl.from_repr("""
            shape: (7, 1)
            ┌──────────────┐
            │ time_offset  │
            │ ---          │
            │ duration[μs] │
            ╞══════════════╡
            │ -2h          │
            │ 0µs          │
            │ 2h           │
            │ +2h          │
            └──────────────┘
    """),
    )
    expected = pl.DataFrame(
        {
            "time_offset": [
                timedelta(hours=-2),
                timedelta(),
                timedelta(hours=2),
                timedelta(hours=2),
            ]
        },
        schema={"time_offset": pl.Duration("us")},
    )
    assert_frame_equal(df, expected)


def test_dataframe_from_duckdb_repr() -> None:
    df = cast(
        "pl.DataFrame",
        pl.from_repr(
            """
            # misc streaming stats
            ┌────────────┬───────┬───────────────────┬───┬────────────────┬───────────────────┐
            │   As Of    │ Rank  │ Year to Date Rank │ … │ Days In Top 10 │ Streaming Seconds │
            │    date    │ int32 │      varchar      │   │     int16      │      int128       │
            ├────────────┼───────┼───────────────────┼───┼────────────────┼───────────────────┤
            │ 2025-05-09 │     1 │ 1                 │ … │             29 │  1864939402857430 │
            │ 2025-05-09 │     2 │ 2                 │ … │             15 │   658937443590045 │
            │ 2025-05-09 │     3 │ 3                 │ … │              9 │   267876522242076 │
            └────────────┴───────┴───────────────────┴───┴────────────────┴───────────────────┘
            """
        ),
    )
    expected = pl.DataFrame(
        {
            "As Of": [date(2025, 5, 9), date(2025, 5, 9), date(2025, 5, 9)],
            "Rank": [1, 2, 3],
            "Year to Date Rank": ["1", "2", "3"],
            "Days In Top 10": [29, 15, 9],
            "Streaming Seconds": [1864939402857430, 658937443590045, 267876522242076],
        },
        schema={
            "As Of": pl.Date,
            "Rank": pl.Int32,
            "Year to Date Rank": pl.String,
            "Days In Top 10": pl.Int16,
            "Streaming Seconds": pl.Int128,
        },
    )
    assert_frame_equal(expected, df)


def test_series_from_repr() -> None:
    frame = (
        pl.LazyFrame(
            {
                "a": [1, 2, None],
                "b": [4.5, 5.5, 6.5],
                "c": ["x", "y", "z"],
                "d": [True, False, True],
                "e": [None, "", None],
                "f": [date(2022, 7, 5), date(2023, 2, 5), date(2023, 8, 5)],
                "g": [time(0, 0, 0, 1), time(12, 30, 45), time(23, 59, 59, 999000)],
                "h": [
                    datetime(2022, 7, 5, 10, 30, 45, 4560),
                    datetime(2023, 10, 12, 20, 3, 8, 11),
                    None,
                ],
            },
        )
        .with_columns(
            pl.col("c").cast(pl.Categorical),
            pl.col("h").cast(pl.Datetime("ns")),
        )
        .collect()
    )

    for col in frame.columns:
        s = cast("pl.Series", pl.from_repr(repr(frame[col])))
        assert_series_equal(s, frame[col])

    s = cast(
        "pl.Series",
        pl.from_repr(
            """
            Out[3]:
            shape: (3,)
            Series: 's' [str]
            [
                "a"
                 …
                "c"
            ]
            """
        ),
    )
    assert_series_equal(s, pl.Series("s", ["a", "c"]))

    s = cast(
        "pl.Series",
        pl.from_repr(
            """
            Series: 'flt' [f32]
            [
            ]
            """
        ),
    )
    assert_series_equal(s, pl.Series("flt", [], dtype=pl.Float32))

    s = cast(
        "pl.Series",
        pl.from_repr(
            """
            Series: 'flt' [f64]
            [
                null
                +inf
                -inf
                inf
                0.0
                NaN
            ]
            >>> print("stuff")
            """
        ),
    )
    inf, nan = float("inf"), float("nan")
    assert_series_equal(
        s,
        pl.Series(
            name="flt",
            dtype=pl.Float64,
            values=[None, inf, -inf, inf, 0.0, nan],
        ),
    )


def test_dataframe_from_repr_custom_separators() -> None:
    # repr created with custom digit-grouping
    # and non-default group/decimal separators
    df = cast(
        "pl.DataFrame",
        pl.from_repr(
            """
            ┌───────────┬────────────┐
            │ x         ┆ y          │
            │ ---       ┆ ---        │
            │ i32       ┆ f64        │
            ╞═══════════╪════════════╡
            │ 123.456   ┆ -10.000,55 │
            │ -9.876    ┆ 10,0       │
            │ 9.999.999 ┆ 8,5e8      │
            └───────────┴────────────┘
            """
        ),
    )
    assert_frame_equal(
        df,
        pl.DataFrame(
            {
                "x": [123456, -9876, 9999999],
                "y": [-10000.55, 10.0, 850000000.0],
            },
            schema={"x": pl.Int32, "y": pl.Float64},
        ),
    )


def test_sliced_struct_from_arrow() -> None:
    # Create a dataset with 3 rows
    tbl = pa.Table.from_arrays(
        arrays=[
            pa.StructArray.from_arrays(
                arrays=[
                    pa.array([1, 2, 3], pa.int32()),
                    pa.array(["foo", "bar", "baz"], pa.utf8()),
                ],
                names=["a", "b"],
            )
        ],
        names=["struct_col"],
    )

    # slice the table
    # check if FFI correctly reads sliced
    result = cast("pl.DataFrame", pl.from_arrow(tbl.slice(1, 2)))
    assert result.to_dict(as_series=False) == {
        "struct_col": [{"a": 2, "b": "bar"}, {"a": 3, "b": "baz"}]
    }

    result = cast("pl.DataFrame", pl.from_arrow(tbl.slice(1, 1)))
    assert result.to_dict(as_series=False) == {"struct_col": [{"a": 2, "b": "bar"}]}


def test_from_arrow_invalid_time_zone() -> None:
    arr = pa.array(
        [datetime(2021, 1, 1, 0, 0, 0, 0)],
        type=pa.timestamp("ns", tz="this-is-not-a-time-zone"),
    )
    with pytest.raises(
        ComputeError, match=r"unable to parse time zone: 'this-is-not-a-time-zone'"
    ):
        pl.from_arrow(arr)


@pytest.mark.parametrize(
    ("fixed_offset", "etc_tz"),
    [
        ("+10:00", "Etc/GMT-10"),
        ("10:00", "Etc/GMT-10"),
        ("-10:00", "Etc/GMT+10"),
        ("+05:00", "Etc/GMT-5"),
        ("05:00", "Etc/GMT-5"),
        ("-05:00", "Etc/GMT+5"),
    ],
)
def test_from_arrow_fixed_offset(fixed_offset: str, etc_tz: str) -> None:
    arr = pa.array(
        [datetime(2021, 1, 1, 0, 0, 0, 0)],
        type=pa.timestamp("us", tz=fixed_offset),
    )
    result = cast("pl.Series", pl.from_arrow(arr))
    expected = pl.Series(
        [datetime(2021, 1, 1, tzinfo=timezone.utc)]
    ).dt.convert_time_zone(etc_tz)
    assert_series_equal(result, expected)


def test_from_avro_valid_time_zone_13032() -> None:
    arr = pa.array(
        [datetime(2021, 1, 1, 0, 0, 0, 0)], type=pa.timestamp("ns", tz="00:00")
    )
    result = cast("pl.Series", pl.from_arrow(arr))
    expected = pl.Series([datetime(2021, 1, 1)], dtype=pl.Datetime("ns", "UTC"))
    assert_series_equal(result, expected)


def test_from_numpy_different_resolution_15991() -> None:
    result = pl.Series(
        np.array(["2020-01-01"], dtype="datetime64[ns]"), dtype=pl.Datetime("us")
    )
    expected = pl.Series([datetime(2020, 1, 1)], dtype=pl.Datetime("us"))
    assert_series_equal(result, expected)


def test_from_numpy_different_resolution_invalid() -> None:
    with pytest.raises(ValueError, match="Please cast"):
        pl.Series(
            np.array(["2020-01-01"], dtype="datetime64[s]"), dtype=pl.Datetime("us")
        )


def test_compat_level(plmonkeypatch: PlMonkeyPatch) -> None:
    # change these if compat level bumped
    plmonkeypatch.setenv("POLARS_WARN_UNSTABLE", "1")
    oldest = CompatLevel.oldest()
    assert oldest is CompatLevel.oldest()  # test singleton
    assert oldest._version == 0
    with pytest.warns(UnstableWarning):
        newest = CompatLevel.newest()
    with pytest.warns(UnstableWarning):
        assert newest is CompatLevel.newest()
    assert newest._version == 1

    str_col = pl.Series(["awd"])
    bin_col = pl.Series([b"dwa"])
    assert str_col._newest_compat_level() == newest._version
    assert isinstance(str_col.to_arrow(), pa.LargeStringArray)
    assert isinstance(str_col.to_arrow(compat_level=oldest), pa.LargeStringArray)
    assert isinstance(str_col.to_arrow(compat_level=newest), pa.StringViewArray)
    assert isinstance(bin_col.to_arrow(), pa.LargeBinaryArray)
    assert isinstance(bin_col.to_arrow(compat_level=oldest), pa.LargeBinaryArray)
    assert isinstance(bin_col.to_arrow(compat_level=newest), pa.BinaryViewArray)

    df = pl.DataFrame({"str_col": str_col, "bin_col": bin_col})
    assert isinstance(df.to_arrow()["str_col"][0], pa.LargeStringScalar)
    assert isinstance(
        df.to_arrow(compat_level=oldest)["str_col"][0], pa.LargeStringScalar
    )
    assert isinstance(
        df.to_arrow(compat_level=newest)["str_col"][0], pa.StringViewScalar
    )
    assert isinstance(df.to_arrow()["bin_col"][0], pa.LargeBinaryScalar)
    assert isinstance(
        df.to_arrow(compat_level=oldest)["bin_col"][0], pa.LargeBinaryScalar
    )
    assert isinstance(
        df.to_arrow(compat_level=newest)["bin_col"][0], pa.BinaryViewScalar
    )


def test_df_pycapsule_interface() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 2, 3],
            "b": ["a", "b", "c"],
            "c": ["fooooooooooooooooooooo", "bar", "looooooooooooooooong string"],
        }
    )

    capsule_df = PyCapsuleStreamHolder(df)
    out = pa.table(capsule_df)
    assert df.shape == out.shape
    assert df.schema.names() == out.schema.names

    schema_overrides = {"a": pl.Int128}
    expected_schema = pl.Schema([("a", pl.Int128), ("b", pl.String), ("c", pl.String)])

    for arrow_obj in (
        pl.from_arrow(capsule_df),  # capsule
        out,  # table loaded from capsule
    ):
        df_res = pl.from_arrow(arrow_obj, schema_overrides=schema_overrides)
        assert expected_schema == df_res.schema  # type: ignore[union-attr]
        assert isinstance(df_res, pl.DataFrame)
        assert df.equals(df_res)


def test_misaligned_nested_arrow_19097() -> None:
    a = pl.Series("a", [1, 2, 3])
    a = a.slice(1, 2)  # by slicing we offset=1 the values
    a = a.replace(2, None)  # then we add a validity mask with offset=0
    a = a.reshape((2, 1))  # then we make it nested
    assert_series_equal(pl.Series("a", a.to_arrow()), a)


def test_arrow_roundtrip_lex_cat_20288() -> None:
    tb = pl.Series("a", ["A", "B"], pl.Categorical()).to_frame().to_arrow()
    df = pl.from_arrow(tb)
    assert isinstance(df, pl.DataFrame)
    dt = df.schema["a"]
    assert isinstance(dt, pl.Categorical)
    assert dt.ordering == "lexical"


def test_from_arrow_20271() -> None:
    df = pl.from_arrow(
        pa.table({"b": pa.DictionaryArray.from_arrays([0, 1], ["D", "E"])})
    )
    assert isinstance(df, pl.DataFrame)
    assert_series_equal(
        df.to_series(),
        pl.Series("b", ["D", "E"], pl.Categorical),
    )


def test_to_arrow_empty_chunks_20627() -> None:
    df = pl.concat(2 * [pl.Series([1])]).filter(pl.Series([False, True])).to_frame()
    assert df.to_arrow().shape == (1, 1)


def test_from_arrow_recorbatch() -> None:
    n_legs = pa.array([2, 2, 4, 4, 5, 100])
    animals = pa.array(
        ["Flamingo", "Parrot", "Dog", "Horse", "Brittle stars", "Centipede"]
    )
    names = ["n_legs", "animals"]
    record_batch = pa.RecordBatch.from_arrays([n_legs, animals], names=names)
    assert_frame_equal(
        pl.DataFrame(record_batch),
        pl.DataFrame({"n_legs": n_legs, "animals": animals}),
    )


def test_from_arrow_map_containing_timestamp_23658() -> None:
    arrow_tbl = pa.Table.from_pydict(
        {
            "column_1": [
                [
                    {
                        "field_1": [
                            {"key": 1, "value": datetime(2025, 1, 1)},
                            {"key": 2, "value": datetime(2025, 1, 2)},
                            {"key": 2, "value": None},
                        ]
                    },
                    {"field_1": []},
                    None,
                ]
            ],
        },
        schema=pa.schema(
            [
                (
                    "column_1",
                    pa.list_(
                        pa.struct(
                            [
                                ("field_1", pa.map_(pa.int32(), pa.timestamp("ms"))),
                            ]
                        )
                    ),
                )
            ]
        ),
    )

    expect = pl.DataFrame(
        {
            "column_1": [
                [
                    {
                        "field_1": [
                            {"key": 1, "value": datetime(2025, 1, 1)},
                            {"key": 2, "value": datetime(2025, 1, 2)},
                            {"key": 2, "value": None},
                        ]
                    },
                    {"field_1": []},
                    None,
                ]
            ],
        },
        schema={
            "column_1": pl.List(
                pl.Struct(
                    {
                        "field_1": pl.List(
                            pl.Struct({"key": pl.Int32, "value": pl.Datetime("ms")})
                        )
                    }
                )
            )
        },
    )

    out = pl.DataFrame(arrow_tbl)
    assert_frame_equal(out, expect)


def test_schema_constructor_from_schema_capsule() -> None:
    arrow_schema = pa.schema(
        [pa.field("test", pa.map_(pa.int32(), pa.timestamp("ms")))]
    )

    assert pl.Schema(arrow_schema) == {
        "test": pl.List(pl.Struct({"key": pl.Int32, "value": pl.Datetime("ms")}))
    }

    # Test __arrow_c_schema__ implementation on `pl.Schema`
    assert pa.schema(pl.Schema({"x": pl.Int32})) == pa.schema(
        [pa.field("x", pa.int32())]
    )

    arrow_schema = pa.schema([pa.field("a", pa.int32()), pa.field("a", pa.int32())])

    with pytest.raises(
        DuplicateError,
        match="arrow schema contained duplicate name: a",
    ):
        pl.Schema(arrow_schema)

    with pytest.raises(
        ValueError,
        match=r"object passed to pl.Schema did not return struct dtype: object: pyarrow\.Field<a: int32>, dtype: Int32",
    ):
        pl.Schema(pa.field("a", pa.int32()))

    assert pl.Schema([pa.field("a", pa.int32()), pa.field("b", pa.string())]) == {
        "a": pl.Int32,
        "b": pl.String,
    }

    with pytest.raises(
        DuplicateError,
        match=r"iterable passed to pl\.Schema contained duplicate name 'a'",
    ):
        pl.Schema([pa.field("a", pa.int32()), pa.field("a", pa.int64())])


def test_to_arrow_24142() -> None:
    df = pl.DataFrame({"a": object(), "b": "any string or bytes"})
    df.to_arrow(compat_level=CompatLevel.oldest())


def test_pycapsule_stream_interface_all_types() -> None:
    """Test all data types via Arrow C Stream PyCapsule interface."""
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

    assert_frame_equal(
        df.map_columns(
            pl.selectors.all(), lambda s: pl.Series(PyCapsuleStreamHolder(s))
        ),
        df,
    )

    assert_frame_equal(
        df.map_columns(
            pl.selectors.all(),
            lambda s: (
                pl.Series(
                    PyCapsuleStreamHolder(pl.select(pl.struct(pl.lit(s))).to_series())
                )
                .struct.unnest()
                .to_series()
            ),
        ),
        df,
    )

    assert_frame_equal(
        df.map_columns(
            pl.selectors.all(),
            lambda s: pl.Series(PyCapsuleStreamHolder(s.implode())).explode(),
        ),
        df,
    )

    assert_frame_equal(
        df.map_columns(
            pl.selectors.all(),
            lambda s: pl.Series(PyCapsuleStreamHolder(s.reshape((3, 1)))).reshape((3,)),
        ),
        df,
    )

    assert_frame_equal(pl.DataFrame(PyCapsuleStreamHolder(df)), df)
    assert_frame_equal(
        pl.DataFrame(PyCapsuleStreamHolder(df.select(pl.struct("*")))).unnest("*"), df
    )
    assert_frame_equal(
        pl.DataFrame(PyCapsuleStreamHolder(df.select(pl.all().implode()))).explode("*"),
        df,
    )
    assert_frame_equal(
        pl.DataFrame(PyCapsuleStreamHolder(df.select(pl.all().reshape((3, 1))))).select(
            pl.all().reshape((3,))
        ),
        df,
    )


def pyarrow_table_to_ipc_bytes(tbl: pa.Table) -> bytes:
    f = io.BytesIO()
    batches = tbl.to_batches()

    with pa.ipc.new_file(f, batches[0].schema) as writer:
        for batch in batches:
            writer.write_batch(batch)

    return f.getvalue()


@pytest.mark.write_disk
def test_month_day_nano_from_ffi_15969(plmonkeypatch: PlMonkeyPatch) -> None:
    import datetime

    def new_interval_scalar(months: int, days: int, nanoseconds: int) -> pa.Scalar:
        return pa.scalar((months, days, nanoseconds), type=pa.month_day_nano_interval())

    arrow_tbl = pa.Table.from_pydict(
        {
            "interval": [
                new_interval_scalar(1, 0, 0),
                new_interval_scalar(0, 1, 0),
                new_interval_scalar(0, 0, 1_000),
                new_interval_scalar(1, 1, 1_000_001_000),
                new_interval_scalar(-1, 0, 0),
                new_interval_scalar(0, -1, 0),
                new_interval_scalar(0, 0, -1_000),
                new_interval_scalar(-1, -1, -1_000_001_000),
                new_interval_scalar(3558, 0, 0),
                new_interval_scalar(-3558, 0, 0),
                new_interval_scalar(1, -1, 1_999_999_000),
            ]
        },
        schema=pa.schema([pa.field("interval", pa.month_day_nano_interval())]),
    )

    ipc_bytes = pyarrow_table_to_ipc_bytes(arrow_tbl)

    import_err_msg = (
        "could not import from `month_day_nano_interval` type. "
        "Hint: This can be imported by setting "
        "POLARS_IMPORT_INTERVAL_AS_STRUCT=1 in the environment. "
        "Note however that this is unstable functionality "
        "that may change at any time."
    )

    with pytest.raises(PanicException, match=import_err_msg):
        pl.scan_ipc(ipc_bytes).collect_schema()

    with pytest.raises(PanicException, match=import_err_msg):
        pl.scan_ipc(ipc_bytes).collect()

    with pytest.raises(PanicException, match=import_err_msg):
        pl.DataFrame(
            pa.Table.from_pydict(
                {"interval": pa.array([], type=pa.month_day_nano_interval())}
            )
        )

    with pytest.raises(ComputeError, match=import_err_msg):
        pl.Series(pa.array([], type=pa.month_day_nano_interval()))

    plmonkeypatch.setenv("POLARS_IMPORT_INTERVAL_AS_STRUCT", "1")

    expect = pl.DataFrame(
        [
            pl.Series(
                "interval",
                [
                    {"months": 1, "days": 0, "nanoseconds": datetime.timedelta(0)},
                    {"months": 0, "days": 1, "nanoseconds": datetime.timedelta(0)},
                    {
                        "months": 0,
                        "days": 0,
                        "nanoseconds": datetime.timedelta(microseconds=1),
                    },
                    {
                        "months": 1,
                        "days": 1,
                        "nanoseconds": datetime.timedelta(seconds=1, microseconds=1),
                    },
                    {"months": -1, "days": 0, "nanoseconds": datetime.timedelta(0)},
                    {"months": 0, "days": -1, "nanoseconds": datetime.timedelta(0)},
                    {
                        "months": 0,
                        "days": 0,
                        "nanoseconds": datetime.timedelta(
                            days=-1, seconds=86399, microseconds=999999
                        ),
                    },
                    {
                        "months": -1,
                        "days": -1,
                        "nanoseconds": datetime.timedelta(
                            days=-1, seconds=86398, microseconds=999999
                        ),
                    },
                    {"months": 3558, "days": 0, "nanoseconds": datetime.timedelta(0)},
                    {"months": -3558, "days": 0, "nanoseconds": datetime.timedelta(0)},
                    {
                        "months": 1,
                        "days": -1,
                        "nanoseconds": datetime.timedelta(
                            seconds=1, microseconds=999999
                        ),
                    },
                ],
                dtype=pl.Struct(
                    {
                        "months": pl.Int32,
                        "days": pl.Int32,
                        "nanoseconds": pl.Duration(time_unit="ns"),
                    }
                ),
            ),
        ]
    )

    assert_frame_equal(pl.DataFrame(arrow_tbl), expect)
    assert_series_equal(
        pl.Series(arrow_tbl.column(0)).alias("interval"), expect.to_series()
    )

    # Test IPC scan
    assert pl.scan_ipc(ipc_bytes).collect_schema() == {
        "interval": pl.Struct(
            {
                "months": pl.Int32,
                "days": pl.Int32,
                "nanoseconds": pl.Duration(time_unit="ns"),
            }
        )
    }
    assert_frame_equal(pl.scan_ipc(ipc_bytes).collect(), expect)

    assert_frame_equal(
        pl.DataFrame(
            pa.Table.from_pydict(
                {"interval": pa.array([], type=pa.month_day_nano_interval())}
            )
        ),
        pl.DataFrame(
            schema={
                "interval": pl.Struct(
                    {
                        "months": pl.Int32,
                        "days": pl.Int32,
                        "nanoseconds": pl.Duration(time_unit="ns"),
                    }
                )
            }
        ),
    )

    assert_series_equal(
        pl.Series(pa.array([], type=pa.month_day_nano_interval())),
        pl.Series(
            dtype=pl.Struct(
                {
                    "months": pl.Int32,
                    "days": pl.Int32,
                    "nanoseconds": pl.Duration(time_unit="ns"),
                }
            )
        ),
    )

    f = io.BytesIO()

    # TODO: Add Parquet round-trip test if this starts working.
    with pytest.raises(pa.ArrowNotImplementedError):
        pq.write_table(arrow_tbl, f)


def test_schema_to_arrow_15563() -> None:
    assert pl.Schema({"x": pl.String}).to_arrow() == pa.schema(
        [pa.field("x", pa.string_view())]
    )

    assert pl.Schema({"x": pl.String}).to_arrow(
        compat_level=CompatLevel.oldest()
    ) == pa.schema([pa.field("x", pa.large_string())])


def test_0_width_df_roundtrip() -> None:
    assert pl.DataFrame(height=(1 << 32) - 1).to_numpy().shape == ((1 << 32) - 1, 0)
    assert pl.DataFrame(np.zeros((10, 0))).shape == (10, 0)

    arrow_table = pl.DataFrame(height=(1 << 32) - 1).to_arrow()
    assert arrow_table.shape == ((1 << 32) - 1, 0)
    assert pl.DataFrame(arrow_table).shape == ((1 << 32) - 1, 0)

    pandas_df = pl.DataFrame(height=(1 << 32) - 1).to_pandas()
    assert pandas_df.shape == ((1 << 32) - 1, 0)
    assert pl.DataFrame(pandas_df).shape == ((1 << 32) - 1, 0)

    df = pl.DataFrame(height=5)

    assert pl.DataFrame.deserialize(df.serialize()).shape == (5, 0)
    assert pl.LazyFrame.deserialize(df.lazy().serialize()).collect().shape == (5, 0)

    for file_format in ["parquet", "ipc", "ndjson"]:
        f = io.BytesIO()
        getattr(pl.DataFrame, f"write_{file_format}")(df, f)
        f.seek(0)
        assert getattr(pl, f"read_{file_format}")(f).shape == (5, 0)

        f = io.BytesIO()
        getattr(pl.LazyFrame, f"sink_{file_format}")(df.lazy(), f)
        f.seek(0)
        assert getattr(pl, f"scan_{file_format}")(f).collect().shape == (5, 0)

    f = io.BytesIO()
    pl.LazyFrame().sink_csv(f)
    v = f.getvalue()
    assert v == b"\n"

    with pytest.raises(
        InvalidOperationError,
        match=r"cannot sink 0-width DataFrame with non-zero height \(1\) to CSV",
    ):
        pl.LazyFrame(height=1).sink_csv(io.BytesIO())
