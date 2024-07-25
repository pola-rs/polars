from __future__ import annotations

from datetime import date, datetime, time, timezone
from typing import Any, cast

import numpy as np
import pandas as pd
import pyarrow
import pyarrow as pa
import pytest

import polars as pl
from polars.exceptions import ComputeError, UnstableWarning
from polars.interchange.protocol import CompatLevel
from polars.testing import assert_frame_equal, assert_series_equal
from tests.unit.utils.pycapsule_utils import PyCapsuleStreamHolder


def test_arrow_list_roundtrip() -> None:
    # https://github.com/pola-rs/polars/issues/1064
    tbl = pa.table({"a": [1], "b": [[1, 2]]})
    arw = pl.from_arrow(tbl).to_arrow()

    assert arw.shape == tbl.shape
    assert arw.schema.names == tbl.schema.names
    for c1, c2 in zip(arw.columns, tbl.columns):
        assert c1.to_pylist() == c2.to_pylist()


def test_arrow_null_roundtrip() -> None:
    tbl = pa.table({"a": [None, None], "b": [[None, None], [None, None]]})
    df = pl.from_arrow(tbl)

    if isinstance(df, pl.DataFrame):
        assert df.dtypes == [pl.Null, pl.List(pl.Null)]

    arw = df.to_arrow()

    assert arw.shape == tbl.shape
    assert arw.schema.names == tbl.schema.names
    for c1, c2 in zip(arw.columns, tbl.columns):
        assert c1.to_pylist() == c2.to_pylist()


def test_arrow_empty_dataframe() -> None:
    # 0x0 dataframe
    df = pl.DataFrame({})
    tbl = pa.table({})
    assert df.to_arrow() == tbl
    df2 = cast(pl.DataFrame, pl.from_arrow(df.to_arrow()))
    assert_frame_equal(df2, df)

    # 0 row dataframe
    df = pl.DataFrame({}, schema={"a": pl.Int32})
    tbl = pa.Table.from_batches([], pa.schema([pa.field("a", pa.int32())]))
    assert df.to_arrow() == tbl
    df2 = cast(pl.DataFrame, pl.from_arrow(df.to_arrow()))
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
    s = cast(pl.Series, pl.from_arrow(ca))
    assert s.dtype == pl.List


def test_from_dict() -> None:
    data = {"a": [1, 2], "b": [3, 4]}
    df = pl.from_dict(data)
    assert df.shape == (2, 2)
    for s1, s2 in zip(list(df), [pl.Series("a", [1, 2]), pl.Series("b", [3, 4])]):
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
    assert df.schema == {"a": pl.Struct, "d": pl.Int64}


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


def test_from_optional_not_available() -> None:
    from polars.dependencies import _LazyModule

    # proxy module is created dynamically if the required module is not available
    # (see the polars.dependencies source code for additional detail/comments)

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
    out = cast(pl.DataFrame, pl.from_arrow(tbl))
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
    df = cast(pl.DataFrame, pl.from_arrow(pa.table(pd.DataFrame({"a": [], "b": []}))))
    assert df.columns == ["a", "b"]
    assert df.dtypes == [pl.Float64, pl.Float64]

    # 2705
    df1 = pd.DataFrame(columns=["b"], dtype=float, index=pd.Index([]))
    tbl = pa.Table.from_pandas(df1)
    out = cast(pl.DataFrame, pl.from_arrow(tbl))
    assert out.columns == ["b", "__index_level_0__"]
    assert out.dtypes == [pl.Float64, pl.Null]
    tbl = pa.Table.from_pandas(df1, preserve_index=False)
    out = cast(pl.DataFrame, pl.from_arrow(tbl))
    assert out.columns == ["b"]
    assert out.dtypes == [pl.Float64]

    # 4568
    tbl = pa.table({"l": []}, schema=pa.schema([("l", pa.large_list(pa.uint8()))]))

    df = cast(pl.DataFrame, pl.from_arrow(tbl))
    assert df.schema["l"] == pl.List(pl.UInt8)


def test_cat_int_types_3500() -> None:
    with pl.StringCache():
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
            s = cast(pl.Series, pl.from_arrow(pyarrow_array.cast(t)))
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

    result = cast(pl.DataFrame, pl.from_arrow(pa_table))
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
    s = cast(pl.Series, pl.from_arrow(arrow_array))
    assert s.dtype == pl.List(pl.Binary)
    assert s.to_list() == val


def test_dataframe_from_repr() -> None:
    # round-trip various types
    with pl.StringCache():
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

        assert frame.schema == {
            "a": pl.Int64,
            "b": pl.Float64,
            "c": pl.Categorical,
            "d": pl.Boolean,
            "e": pl.String,
            "f": pl.Date,
            "g": pl.Time,
            "h": pl.Datetime("ns"),
        }
        df = cast(pl.DataFrame, pl.from_repr(repr(frame)))
        assert_frame_equal(frame, df)

    # empty frame; confirm schema is inferred
    df = cast(
        pl.DataFrame,
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
        pl.DataFrame,
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

    # empty frame with non-standard/blank 'null'
    df = cast(
        pl.DataFrame,
        pl.from_repr(
            """
            ┌─────┬─────┐
            │ c1  ┆ c2  │
            │ --- ┆ --- │
            │ i32 ┆ f64 │
            ╞═════╪═════╡
            │     │     │
            └─────┴─────┘
            """
        ),
    )
    assert_frame_equal(
        df,
        pl.DataFrame(
            data=[(None, None)], schema={"c1": pl.Int32, "c2": pl.Float64}, orient="row"
        ),
    )

    df = cast(
        pl.DataFrame,
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
        pl.DataFrame,
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
        pl.DataFrame,
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


def test_series_from_repr() -> None:
    with pl.StringCache():
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
            s = cast(pl.Series, pl.from_repr(repr(frame[col])))
            assert_series_equal(s, frame[col])

    s = cast(
        pl.Series,
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
        pl.Series,
        pl.from_repr(
            """
            Series: 'flt' [f32]
            [
            ]
            """
        ),
    )
    assert_series_equal(s, pl.Series("flt", [], dtype=pl.Float32))


def test_dataframe_from_repr_custom_separators() -> None:
    # repr created with custom digit-grouping
    # and non-default group/decimal separators
    df = cast(
        pl.DataFrame,
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
    result = cast(pl.DataFrame, pl.from_arrow(tbl.slice(1, 2)))
    assert result.to_dict(as_series=False) == {
        "struct_col": [{"a": 2, "b": "bar"}, {"a": 3, "b": "baz"}]
    }

    result = cast(pl.DataFrame, pl.from_arrow(tbl.slice(1, 1)))
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
    result = cast(pl.Series, pl.from_arrow(arr))
    expected = pl.Series(
        [datetime(2021, 1, 1, tzinfo=timezone.utc)]
    ).dt.convert_time_zone(etc_tz)
    assert_series_equal(result, expected)


def test_from_avro_valid_time_zone_13032() -> None:
    arr = pa.array(
        [datetime(2021, 1, 1, 0, 0, 0, 0)], type=pa.timestamp("ns", tz="00:00")
    )
    result = cast(pl.Series, pl.from_arrow(arr))
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


def test_compat_level(monkeypatch: pytest.MonkeyPatch) -> None:
    # change these if compat level bumped
    monkeypatch.setenv("POLARS_WARN_UNSTABLE", "1")
    oldest = CompatLevel.oldest()
    assert oldest is CompatLevel.oldest()  # test singleton
    assert oldest._version == 0  # type: ignore[attr-defined]
    with pytest.warns(UnstableWarning):
        newest = CompatLevel.newest()
        assert newest is CompatLevel.newest()
    assert newest._version == 1  # type: ignore[attr-defined]

    str_col = pl.Series(["awd"])
    bin_col = pl.Series([b"dwa"])
    assert str_col._newest_compat_level() == newest._version  # type: ignore[attr-defined]
    assert isinstance(str_col.to_arrow(), pyarrow.LargeStringArray)
    assert isinstance(str_col.to_arrow(compat_level=oldest), pyarrow.LargeStringArray)
    assert isinstance(str_col.to_arrow(compat_level=newest), pyarrow.StringViewArray)
    assert isinstance(bin_col.to_arrow(), pyarrow.LargeBinaryArray)
    assert isinstance(bin_col.to_arrow(compat_level=oldest), pyarrow.LargeBinaryArray)
    assert isinstance(bin_col.to_arrow(compat_level=newest), pyarrow.BinaryViewArray)

    df = pl.DataFrame({"str_col": str_col, "bin_col": bin_col})
    assert isinstance(df.to_arrow()["str_col"][0], pyarrow.LargeStringScalar)
    assert isinstance(
        df.to_arrow(compat_level=oldest)["str_col"][0], pyarrow.LargeStringScalar
    )
    assert isinstance(
        df.to_arrow(compat_level=newest)["str_col"][0], pyarrow.StringViewScalar
    )
    assert isinstance(df.to_arrow()["bin_col"][0], pyarrow.LargeBinaryScalar)
    assert isinstance(
        df.to_arrow(compat_level=oldest)["bin_col"][0], pyarrow.LargeBinaryScalar
    )
    assert isinstance(
        df.to_arrow(compat_level=newest)["bin_col"][0], pyarrow.BinaryViewScalar
    )

    assert len(df.write_ipc(None).getbuffer()) == 786
    assert len(df.write_ipc(None, compat_level=oldest).getbuffer()) == 914
    assert len(df.write_ipc(None, compat_level=newest).getbuffer()) == 786
    assert len(df.write_ipc_stream(None).getbuffer()) == 544
    assert len(df.write_ipc_stream(None, compat_level=oldest).getbuffer()) == 672
    assert len(df.write_ipc_stream(None, compat_level=newest).getbuffer()) == 544


def test_df_pycapsule_interface() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 2, 3],
            "b": ["a", "b", "c"],
            "c": ["fooooooooooooooooooooo", "bar", "looooooooooooooooong string"],
        }
    )
    out = pa.table(PyCapsuleStreamHolder(df))
    assert df.shape == out.shape
    assert df.schema.names() == out.schema.names

    df2 = pl.from_arrow(out)
    assert isinstance(df2, pl.DataFrame)
    assert df.equals(df2)
