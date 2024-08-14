from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from typing import TYPE_CHECKING

import pandas as pd
import pyarrow as pa
import pytest

import polars as pl
import polars.selectors as cs
from polars.testing import assert_frame_equal, assert_series_equal

if TYPE_CHECKING:
    from polars._typing import PolarsDataType


def test_struct_to_list() -> None:
    assert pl.DataFrame(
        {"int": [1, 2], "str": ["a", "b"], "bool": [True, None], "list": [[1, 2], [3]]}
    ).select([pl.struct(pl.all()).alias("my_struct")]).to_series().to_list() == [
        {"int": 1, "str": "a", "bool": True, "list": [1, 2]},
        {"int": 2, "str": "b", "bool": None, "list": [3]},
    ]


def test_apply_unnest() -> None:
    df = (
        pl.Series([None, 2, 3, 4])
        .map_elements(
            lambda x: {"a": x, "b": x * 2, "c": True, "d": [1, 2], "e": "foo"}
        )
        .struct.unnest()
    )

    expected = pl.DataFrame(
        {
            "a": [None, 2, 3, 4],
            "b": [None, 4, 6, 8],
            "c": [None, True, True, True],
            "d": [None, [1, 2], [1, 2], [1, 2]],
            "e": [None, "foo", "foo", "foo"],
        }
    )

    assert_frame_equal(df, expected)


def test_struct_equality() -> None:
    # equal struct dimensions, equal values
    s1 = pl.Series("misc", [{"x": "a", "y": 0}, {"x": "b", "y": 0}])
    s2 = pl.Series("misc", [{"x": "a", "y": 0}, {"x": "b", "y": 0}])
    assert (s1 == s2).all()
    assert (~(s1 != s2)).all()

    # equal struct dimensions, unequal values
    s3 = pl.Series("misc", [{"x": "a", "y": 0}, {"x": "c", "y": 2}])
    s4 = pl.Series("misc", [{"x": "b", "y": 1}, {"x": "d", "y": 3}])
    assert (s3 != s4).all()
    assert (~(s3 == s4)).all()

    # unequal struct dimensions, equal values (where fields overlap)
    s5 = pl.Series("misc", [{"x": "a", "y": 0}, {"x": "b", "y": 0}])
    s6 = pl.Series("misc", [{"x": "a", "y": 0, "z": 0}, {"x": "b", "y": 0, "z": 0}])
    assert (s5 != s6).all()
    assert (~(s5 == s6)).all()

    s7 = pl.Series("misc", [{"x": "a", "y": 0}, {"x": "b", "y": 0}])
    s8 = pl.Series("misc", [{"x": "a", "y": 0}, {"x": "b", "y": 0}, {"x": "c", "y": 0}])
    assert (s7 != s8).all()
    assert (~(s7 == s8)).all()


def test_struct_equality_strict() -> None:
    s1 = pl.Struct(
        [
            pl.Field("a", pl.Int64),
            pl.Field("b", pl.Boolean),
            pl.Field("c", pl.List(pl.Int32)),
        ]
    )
    s2 = pl.Struct(
        [pl.Field("a", pl.Int64), pl.Field("b", pl.Boolean), pl.Field("c", pl.List)]
    )

    # strict
    assert s1.is_(s2) is False

    # permissive (default)
    assert s1 == s2
    assert s1 == s2


def test_struct_hashes() -> None:
    dtypes = (
        pl.Struct,
        pl.Struct([pl.Field("a", pl.Int64)]),
        pl.Struct([pl.Field("a", pl.Int64), pl.Field("b", pl.List(pl.Int64))]),
    )
    assert len({hash(tp) for tp in (dtypes)}) == 3


def test_struct_unnesting() -> None:
    df_base = pl.DataFrame({"a": [1, 2]})
    df = df_base.select(
        pl.all().alias("a_original"),
        pl.col("a")
        .map_elements(lambda x: {"a": x, "b": x * 2, "c": x % 2 == 0})
        .struct.rename_fields(["a", "a_squared", "mod2eq0"])
        .alias("foo"),
    )
    expected = pl.DataFrame(
        {
            "a_original": [1, 2],
            "a": [1, 2],
            "a_squared": [2, 4],
            "mod2eq0": [False, True],
        }
    )
    for cols in ("foo", cs.ends_with("oo")):
        out_eager = df.unnest(cols)
        assert_frame_equal(out_eager, expected)

        out_lazy = df.lazy().unnest(cols)
        assert_frame_equal(out_lazy, expected.lazy())

    out = (
        df_base.lazy()
        .select(
            pl.all().alias("a_original"),
            pl.col("a")
            .map_elements(lambda x: {"a": x, "b": x * 2, "c": x % 2 == 0})
            .struct.rename_fields(["a", "a_squared", "mod2eq0"])
            .alias("foo"),
        )
        .unnest("foo")
        .collect()
    )
    assert_frame_equal(out, expected)


def test_struct_unnest_multiple() -> None:
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4], "c": [1.0, 2.0], "d": ["a", "b"]})
    df_structs = df.select(s1=pl.struct(["a", "b"]), s2=pl.struct(["c", "d"]))

    # List input
    result = df_structs.unnest(["s1", "s2"])
    assert_frame_equal(result, df)
    assert all(tp.is_nested() for tp in df_structs.dtypes)

    # Positional input
    result = df_structs.unnest("s1", "s2")
    assert_frame_equal(result, df)


def test_struct_function_expansion() -> None:
    df = pl.DataFrame(
        {"a": [1, 2, 3, 4], "b": ["one", "two", "three", "four"], "c": [9, 8, 7, 6]}
    )
    struct_schema = {"a": pl.UInt32, "b": pl.String}
    dfs = df.with_columns(pl.struct(pl.col(["a", "b"]), schema=struct_schema))
    s = dfs["a"]

    assert isinstance(s, pl.Series)
    assert s.struct.fields == ["a", "b"]
    assert pl.Struct(struct_schema) == s.to_frame().schema["a"]

    assert_series_equal(s, pl.Series(dfs.select("a")))
    assert_frame_equal(dfs, pl.DataFrame(dfs))


def test_nested_struct() -> None:
    df = pl.DataFrame({"d": [1, 2, 3], "e": ["foo", "bar", "biz"]})
    # Nest the dataframe
    nest_l1 = df.to_struct("c").to_frame()
    # Add another column on the same level
    nest_l1 = nest_l1.with_columns(pl.col("c").is_null().alias("b"))
    # Nest the dataframe again
    nest_l2 = nest_l1.to_struct("a").to_frame()

    assert isinstance(nest_l2.dtypes[0], pl.datatypes.Struct)
    assert [f.dtype for f in nest_l2.dtypes[0].fields] == nest_l1.dtypes
    assert isinstance(nest_l1.dtypes[0], pl.datatypes.Struct)


def test_struct_to_pandas() -> None:
    pdf = pd.DataFrame([{"a": {"b": {"c": 2}}}])
    df = pl.from_pandas(pdf)

    assert isinstance(df.dtypes[0], pl.datatypes.Struct)
    assert df.to_pandas().equals(pdf)


def test_struct_logical_types_to_pandas() -> None:
    timestamp = datetime(2022, 1, 1)
    df = pd.DataFrame([{"struct": {"timestamp": timestamp}}])
    assert pl.from_pandas(df).dtypes == [pl.Struct]


def test_struct_cols() -> None:
    """Test that struct columns can be imported and work as expected."""

    def build_struct_df(data: list[dict[str, object]]) -> pl.DataFrame:
        """
        Build Polars df from list of dicts.

        Can't import directly because of issue #3145.
        """
        arrow_df = pa.Table.from_pylist(data)
        polars_df = pl.from_arrow(arrow_df)
        assert isinstance(polars_df, pl.DataFrame)
        return polars_df

    # struct column
    df = build_struct_df([{"struct_col": {"inner": 1}}])
    assert df.columns == ["struct_col"]
    assert df.schema == {"struct_col": pl.Struct}
    assert df["struct_col"].struct.field("inner").to_list() == [1]

    # struct in struct
    df = build_struct_df([{"nested_struct_col": {"struct_col": {"inner": 1}}}])
    assert df["nested_struct_col"].struct.field("struct_col").struct.field(
        "inner"
    ).to_list() == [1]

    # struct in list
    df = build_struct_df([{"list_of_struct_col": [{"inner": 1}]}])
    assert df["list_of_struct_col"][0].struct.field("inner").to_list() == [1]

    # struct in list in struct
    df = build_struct_df(
        [{"struct_list_struct_col": {"list_struct_col": [{"inner": 1}]}}]
    )
    assert df["struct_list_struct_col"].struct.field("list_struct_col")[0].struct.field(
        "inner"
    ).to_list() == [1]


def test_struct_with_validity() -> None:
    data = [{"a": {"b": 1}}, {"a": None}]
    tbl = pa.Table.from_pylist(data)
    df = pl.from_arrow(tbl)
    assert isinstance(df, pl.DataFrame)
    assert df["a"].to_list() == [{"b": 1}, None]


def test_from_dicts_struct() -> None:
    assert pl.from_dicts([{"a": 1, "b": {"a": 1, "b": 2}}]).to_series(1).to_list() == [
        {"a": 1, "b": 2}
    ]

    assert pl.from_dicts(
        [{"a": 1, "b": {"a_deep": 1, "b_deep": {"a_deeper": [1, 2, 4]}}}]
    ).to_series(1).to_list() == [{"a_deep": 1, "b_deep": {"a_deeper": [1, 2, 4]}}]

    data = [
        {"a": [{"b": 0, "c": 1}]},
        {"a": [{"b": 1, "c": 2}]},
    ]

    assert pl.from_dicts(data).to_series().to_list() == [
        [{"b": 0, "c": 1}],
        [{"b": 1, "c": 2}],
    ]


def test_list_to_struct() -> None:
    df = pl.DataFrame({"a": [[1, 2, 3], [1, 2]]})
    assert df.select([pl.col("a").list.to_struct()]).to_series().to_list() == [
        {"field_0": 1, "field_1": 2, "field_2": 3},
        {"field_0": 1, "field_1": 2, "field_2": None},
    ]

    df = pl.DataFrame({"a": [[1, 2], [1, 2, 3]]})
    assert df.select(
        pl.col("a").list.to_struct(fields=lambda idx: f"col_name_{idx}")
    ).to_series().to_list() == [
        {"col_name_0": 1, "col_name_1": 2},
        {"col_name_0": 1, "col_name_1": 2},
    ]

    df = pl.DataFrame({"a": [[1, 2], [1, 2, 3]]})
    assert df.select(
        pl.col("a").list.to_struct(n_field_strategy="max_width")
    ).to_series().to_list() == [
        {"field_0": 1, "field_1": 2, "field_2": None},
        {"field_0": 1, "field_1": 2, "field_2": 3},
    ]

    # set upper bound
    df = pl.DataFrame({"lists": [[1, 1, 1], [0, 1, 0], [1, 0, 0]]})
    assert df.lazy().select(pl.col("lists").list.to_struct(upper_bound=3)).unnest(
        "lists"
    ).sum().collect().columns == ["field_0", "field_1", "field_2"]


def test_sort_df_with_list_struct() -> None:
    assert pl.DataFrame([{"a": 1, "b": [{"c": 1}]}]).sort("a").to_dict(
        as_series=False
    ) == {
        "a": [1],
        "b": [[{"c": 1}]],
    }


def test_struct_list_head_tail() -> None:
    assert pl.DataFrame(
        {
            "list_of_struct": [
                [{"a": 1, "b": 4}, {"a": 3, "b": 6}],
                [{"a": 10, "b": 40}, {"a": 20, "b": 50}, {"a": 30, "b": 60}],
            ]
        }
    ).with_columns(
        pl.col("list_of_struct").list.head(1).alias("head"),
        pl.col("list_of_struct").list.tail(1).alias("tail"),
    ).to_dict(as_series=False) == {
        "list_of_struct": [
            [{"a": 1, "b": 4}, {"a": 3, "b": 6}],
            [{"a": 10, "b": 40}, {"a": 20, "b": 50}, {"a": 30, "b": 60}],
        ],
        "head": [[{"a": 1, "b": 4}], [{"a": 10, "b": 40}]],
        "tail": [[{"a": 3, "b": 6}], [{"a": 30, "b": 60}]],
    }


def test_struct_agg_all() -> None:
    df = pl.DataFrame(
        {
            "group": ["a", "a", "b", "b", "b"],
            "col1": [
                {"x": 1, "y": 100},
                {"x": 2, "y": 200},
                {"x": 3, "y": 300},
                {"x": 4, "y": 400},
                {"x": 5, "y": 500},
            ],
        }
    )

    assert df.group_by("group", maintain_order=True).all().to_dict(as_series=False) == {
        "group": ["a", "b"],
        "col1": [
            [{"x": 1, "y": 100}, {"x": 2, "y": 200}],
            [{"x": 3, "y": 300}, {"x": 4, "y": 400}, {"x": 5, "y": 500}],
        ],
    }


def test_struct_empty_list_creation() -> None:
    payload = [[], [{"a": 1, "b": 2}, {"a": 3, "b": 4}, {"a": 5, "b": 6}], []]

    assert pl.DataFrame({"list_struct": payload}).to_dict(as_series=False) == {
        "list_struct": [[], [{"a": 1, "b": 2}, {"a": 3, "b": 4}, {"a": 5, "b": 6}], []]
    }

    # pop first
    payload = payload[1:]
    assert pl.DataFrame({"list_struct": payload}).to_dict(as_series=False) == {
        "list_struct": [[{"a": 1, "b": 2}, {"a": 3, "b": 4}, {"a": 5, "b": 6}], []]
    }


def test_struct_arr_methods() -> None:
    df = pl.DataFrame(
        {
            "list_struct": [
                [{"a": 1, "b": 2}, {"a": 3, "b": 4}, {"a": 5, "b": 6}],
                [{"a": 1, "b": 2}, {"a": 3, "b": 4}],
                [{"a": 1, "b": 2}],
            ],
        }
    )
    assert df.select([pl.col("list_struct").list.first()]).to_dict(as_series=False) == {
        "list_struct": [{"a": 1, "b": 2}, {"a": 1, "b": 2}, {"a": 1, "b": 2}]
    }
    assert df.select([pl.col("list_struct").list.last()]).to_dict(as_series=False) == {
        "list_struct": [{"a": 5, "b": 6}, {"a": 3, "b": 4}, {"a": 1, "b": 2}]
    }
    assert df.select([pl.col("list_struct").list.get(0)]).to_dict(as_series=False) == {
        "list_struct": [{"a": 1, "b": 2}, {"a": 1, "b": 2}, {"a": 1, "b": 2}]
    }


def test_struct_concat_list() -> None:
    assert pl.DataFrame(
        {
            "list_struct1": [
                [{"a": 1, "b": 2}, {"a": 3, "b": 4}],
                [{"a": 1, "b": 2}],
            ],
            "list_struct2": [
                [{"a": 6, "b": 7}, {"a": 8, "b": 9}],
                [{"a": 6, "b": 7}],
            ],
        }
    ).with_columns(pl.col("list_struct1").list.concat("list_struct2").alias("result"))[
        "result"
    ].to_list() == [
        [{"a": 1, "b": 2}, {"a": 3, "b": 4}, {"a": 6, "b": 7}, {"a": 8, "b": 9}],
        [{"a": 1, "b": 2}, {"a": 6, "b": 7}],
    ]


def test_struct_arr_reverse() -> None:
    assert pl.DataFrame(
        {
            "list_struct": [
                [{"a": 1, "b": 2}, {"a": 3, "b": 4}, {"a": 5, "b": 6}],
                [{"a": 30, "b": 40}, {"a": 10, "b": 20}, {"a": 50, "b": 60}],
            ],
        }
    ).with_columns([pl.col("list_struct").list.reverse()]).to_dict(as_series=False) == {
        "list_struct": [
            [{"a": 5, "b": 6}, {"a": 3, "b": 4}, {"a": 1, "b": 2}],
            [{"a": 50, "b": 60}, {"a": 10, "b": 20}, {"a": 30, "b": 40}],
        ]
    }


def test_struct_comparison() -> None:
    df = pl.DataFrame(
        {
            "col1": [{"a": 1, "b": 2}, {"a": 3, "b": 4}],
            "col2": [{"a": 1, "b": 2}, {"a": 3, "b": 4}],
        }
    )
    assert df.filter(pl.col("col1") == pl.col("col2")).rows() == [
        ({"a": 1, "b": 2}, {"a": 1, "b": 2}),
        ({"a": 3, "b": 4}, {"a": 3, "b": 4}),
    ]
    # floats w/ ints
    df = pl.DataFrame(
        {
            "col1": [{"a": 1, "b": 2}, {"a": 3, "b": 4}],
            "col2": [{"a": 1.0, "b": 2}, {"a": 3.0, "b": 4}],
        }
    )
    assert df.filter(pl.col("col1") == pl.col("col2")).rows() == [
        ({"a": 1, "b": 2}, {"a": 1.0, "b": 2}),
        ({"a": 3, "b": 4}, {"a": 3.0, "b": 4}),
    ]

    df = pl.DataFrame(
        {
            "col1": [{"a": 1, "b": 2}, {"a": 3, "b": 4}],
            "col2": [{"a": 2, "b": 2}, {"a": 3, "b": 4}],
        }
    )
    assert df.filter(pl.col("col1") == pl.col("col2")).to_dict(as_series=False) == {
        "col1": [{"a": 3, "b": 4}],
        "col2": [{"a": 3, "b": 4}],
    }


def test_struct_order() -> None:
    df = pl.DataFrame({"col1": [{"a": 1, "b": 2}, {"b": 4, "a": 3}]})
    expected = {"col1": [{"a": 1, "b": 2}, {"a": 3, "b": 4}]}
    assert df.to_dict(as_series=False) == expected

    # null values should not trigger this
    assert (
        pl.Series(
            values=[
                {"a": 1, "b": None},
                {"a": 2, "b": 20},
            ],
        ).to_list()
    ) == [{"a": 1, "b": None}, {"a": 2, "b": 20}]

    assert (
        pl.Series(
            values=[
                {"a": 1, "b": 10},
                {"a": 2, "b": None},
            ],
        ).to_list()
    ) == [{"a": 1, "b": 10}, {"a": 2, "b": None}]


def test_struct_arr_eval() -> None:
    df = pl.DataFrame(
        {"col_struct": [[{"a": 1, "b": 11}, {"a": 2, "b": 12}, {"a": 1, "b": 11}]]}
    )
    assert df.with_columns(
        pl.col("col_struct").list.eval(pl.element().first()).alias("first")
    ).to_dict(as_series=False) == {
        "col_struct": [[{"a": 1, "b": 11}, {"a": 2, "b": 12}, {"a": 1, "b": 11}]],
        "first": [[{"a": 1, "b": 11}]],
    }


def test_list_of_struct_unique() -> None:
    df = pl.DataFrame(
        {"col_struct": [[{"a": 1, "b": 11}, {"a": 2, "b": 12}, {"a": 1, "b": 11}]]}
    )
    # the order is unpredictable
    unique = df.with_columns(pl.col("col_struct").list.unique().alias("unique"))[
        "unique"
    ].to_list()
    assert len(unique) == 1
    unique_el = unique[0]
    assert len(unique_el) == 2
    assert {"a": 2, "b": 12} in unique_el
    assert {"a": 1, "b": 11} in unique_el


def test_nested_explode_4026() -> None:
    df = pl.DataFrame(
        {
            "data": [
                [
                    {"account_id": 10, "values": [1, 2]},
                    {"account_id": 11, "values": [10, 20]},
                ]
            ],
            "day": ["monday"],
        }
    )

    assert df.explode("data").to_dict(as_series=False) == {
        "data": [
            {"account_id": 10, "values": [1, 2]},
            {"account_id": 11, "values": [10, 20]},
        ],
        "day": ["monday", "monday"],
    }


def test_nested_struct_sliced_append() -> None:
    s = pl.Series(
        [
            {
                "_experience": {
                    "aaid": {
                        "id": "A",
                        "namespace": {"code": "alpha"},
                    }
                }
            },
            {
                "_experience": {
                    "aaid": {
                        "id": "B",
                        "namespace": {"code": "bravo"},
                    },
                }
            },
            {
                "_experience": {
                    "aaid": {
                        "id": "D",
                        "namespace": {"code": "delta"},
                    }
                }
            },
        ]
    )
    s2 = s[1:]
    s.append(s2)

    assert s.to_list() == [
        {"_experience": {"aaid": {"id": "A", "namespace": {"code": "alpha"}}}},
        {"_experience": {"aaid": {"id": "B", "namespace": {"code": "bravo"}}}},
        {"_experience": {"aaid": {"id": "D", "namespace": {"code": "delta"}}}},
        {"_experience": {"aaid": {"id": "B", "namespace": {"code": "bravo"}}}},
        {"_experience": {"aaid": {"id": "D", "namespace": {"code": "delta"}}}},
    ]


def test_struct_group_by_field_agg_4216() -> None:
    df = pl.DataFrame([{"a": {"b": 1}, "c": 0}])

    result = df.group_by("c").agg(pl.col("a").struct.field("b").count())
    expected = {"c": [0], "b": [1]}
    assert result.to_dict(as_series=False) == expected


def test_struct_getitem() -> None:
    assert pl.Series([{"a": 1, "b": 2}]).struct["b"].name == "b"
    assert pl.Series([{"a": 1, "b": 2}]).struct[0].name == "a"
    assert pl.Series([{"a": 1, "b": 2}]).struct[1].name == "b"
    assert pl.Series([{"a": 1, "b": 2}]).struct[-1].name == "b"
    assert pl.Series([{"a": 1, "b": 2}]).to_frame().select(
        pl.col("").struct[0]
    ).to_dict(as_series=False) == {"a": [1]}


def test_struct_supertype() -> None:
    assert pl.from_dicts(
        [{"vehicle": {"auto": "car"}}, {"vehicle": {"auto": None}}]
    ).to_dict(as_series=False) == {"vehicle": [{"auto": "car"}, {"auto": None}]}


def test_struct_any_value_get_after_append() -> None:
    schema = {"a": pl.Int8, "b": pl.Int32}
    struct_def = pl.Struct(schema)

    a = pl.Series("s", [{"a": 1, "b": 2}], dtype=struct_def)
    b = pl.Series("s", [{"a": 2, "b": 3}], dtype=struct_def)
    a = a.append(b)

    assert a[0] == {"a": 1, "b": 2}
    assert a[1] == {"a": 2, "b": 3}
    assert schema == a.to_frame().unnest("s").schema


def test_struct_categorical_5843() -> None:
    df = pl.DataFrame({"foo": ["a", "b", "c", "a"]}).with_columns(
        pl.col("foo").cast(pl.Categorical)
    )
    result = df.select(pl.col("foo").value_counts(sort=True))
    assert result.to_dict(as_series=False) == {
        "foo": [
            {"foo": "a", "count": 2},
            {"foo": "b", "count": 1},
            {"foo": "c", "count": 1},
        ]
    }


def test_empty_struct() -> None:
    # List<struct>
    df = pl.DataFrame({"a": [[{}]]})
    assert df.to_dict(as_series=False) == {"a": [[None]]}

    # Struct one not empty
    df = pl.DataFrame({"a": [[{}, {"a": 10}]]})
    assert df.to_dict(as_series=False) == {"a": [[{"a": None}, {"a": 10}]]}

    # Empty struct
    df = pl.DataFrame({"a": [{}]})
    assert df.to_dict(as_series=False) == {"a": [None]}


@pytest.mark.parametrize(
    "dtype",
    [
        pl.List,
        pl.List(pl.Null),
        pl.List(pl.String),
        pl.Array(pl.Null, 32),
        pl.Array(pl.UInt8, 16),
        pl.Struct([pl.Field("", pl.Null)]),
        pl.Struct([pl.Field("x", pl.UInt32), pl.Field("y", pl.Float64)]),
    ],
)
def test_empty_series_nested_dtype(dtype: PolarsDataType) -> None:
    # various flavours of empty nested dtype
    s = pl.Series("nested", dtype=dtype)
    assert s.dtype.base_type() == dtype.base_type()
    assert s.to_list() == []


@pytest.mark.parametrize(
    "data",
    [
        [{}, {}],
        [{}, None],
        [None, {}],
        [None, None],
    ],
)
def test_empty_with_schema_struct(data: list[dict[str, object] | None]) -> None:
    # Empty structs, with schema
    struct_schema = {"a": pl.Date, "b": pl.Boolean, "c": pl.Float64}
    frame_schema = {"x": pl.Int8, "y": pl.Struct(struct_schema)}

    @dataclass
    class TestData:
        x: int
        y: dict[str, object] | None

    # test init from rows, dicts, and dataclasses
    dict_data = {"x": [10, 20], "y": data}
    dataclass_data = [
        TestData(10, data[0]),
        TestData(20, data[1]),
    ]
    for frame_data in (dict_data, dataclass_data):
        df = pl.DataFrame(
            data=frame_data,
            schema=frame_schema,  # type: ignore[arg-type]
        )
        assert df.schema == frame_schema
        assert df.unnest("y").columns == ["x", "a", "b", "c"]
        assert df.rows() == [
            (
                10,
                {"a": None, "b": None, "c": None} if data[0] is not None else None,
            ),
            (
                20,
                {"a": None, "b": None, "c": None} if data[1] is not None else None,
            ),
        ]


def test_struct_null_cast() -> None:
    dtype = pl.Struct(
        [
            pl.Field("a", pl.Int64),
            pl.Field("b", pl.String),
            pl.Field("c", pl.List(pl.Float64)),
        ]
    )
    assert (
        pl.DataFrame()
        .lazy()
        .select([pl.lit(None, dtype=pl.Null).cast(dtype, strict=True)])
        .collect()
    ).to_dict(as_series=False) == {"literal": [None]}


def test_nested_struct_in_lists_cast() -> None:
    assert pl.DataFrame(
        {
            "node_groups": [
                [{"nodes": [{"id": 1, "is_started": True}]}],
                [{"nodes": []}],
            ]
        }
    ).to_dict(as_series=False) == {
        "node_groups": [[{"nodes": [{"id": 1, "is_started": True}]}], [{"nodes": []}]]
    }


def test_struct_concat_self_no_rechunk() -> None:
    df = pl.DataFrame([{"A": {"a": 1}}])
    out = pl.concat([df, df], rechunk=False)
    assert out.dtypes == [pl.Struct([pl.Field("a", pl.Int64)])]
    assert out.to_dict(as_series=False) == {"A": [{"a": 1}, {"a": 1}]}


def test_sort_structs() -> None:
    df = pl.DataFrame(
        {
            "sex": ["m", "f", "f", "f", "m", "m", "f"],
            "age": [22, 38, 26, 24, 21, 46, 22],
        },
    )
    df_sorted_as_struct = df.select(pl.struct(["sex", "age"]).sort()).unnest("sex")
    df_expected = df.sort(by=["sex", "age"])

    assert_frame_equal(df_expected, df_sorted_as_struct)
    assert df_sorted_as_struct.to_dict(as_series=False) == {
        "sex": ["f", "f", "f", "f", "m", "m", "m"],
        "age": [22, 24, 26, 38, 21, 22, 46],
    }


def test_struct_applies_as_map() -> None:
    df = pl.DataFrame({"id": [1, 1, 2], "x": ["a", "b", "c"], "y": ["d", "e", "f"]})

    # the window function doesn't really make sense
    # but it runs the test: #7286
    assert df.select(
        pl.struct([pl.col("x"), pl.col("y") + pl.col("y")]).over("id")
    ).to_dict(as_series=False) == {
        "x": [{"x": "a", "y": "dd"}, {"x": "b", "y": "ee"}, {"x": "c", "y": "ff"}]
    }


def test_struct_is_in() -> None:
    # The dtype casts below test that struct is_in upcasts dtypes.
    s1 = (
        pl.DataFrame({"x": [4, 3, 4, 9], "y": [0, 4, 6, 2]})
        .select(pl.struct(schema={"x": pl.Int8, "y": pl.Float32}))
        .to_series()
    )
    s2 = (
        pl.DataFrame({"x": [4, 3, 5, 9], "y": [0, 7, 6, 2]})
        .select(pl.struct(["x", "y"]))
        .to_series()
    )
    assert s1.is_in(s2).to_list() == [True, False, False, True]


def test_nested_struct_logicals() -> None:
    # single nested
    payload1 = [[{"a": time(10)}], [{"a": time(10)}]]
    assert pl.Series(payload1).to_list() == payload1
    # double nested
    payload2 = [[[{"a": time(10)}]], [[{"a": time(10)}]]]
    assert pl.Series(payload2).to_list() == payload2


def test_struct_name_passed_in_agg_apply() -> None:
    struct_expr = pl.struct(
        [
            pl.col("A").min(),
            pl.col("B").search_sorted(pl.Series([3, 4])),
        ]
    ).alias("index")

    assert pl.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [1, 2, 2]}).group_by(
        "C"
    ).agg(struct_expr).sort("C", descending=True).to_dict(as_series=False) == {
        "C": [2, 1],
        "index": [
            [{"A": 2, "B": 0}, {"A": 2, "B": 0}],
            [{"A": 1, "B": 0}, {"A": 1, "B": 0}],
        ],
    }

    df = pl.DataFrame({"val": [-3, -2, -1, 0, 1, 2, 3], "k": [0] * 7})

    assert df.group_by("k").agg(
        pl.struct(
            pl.col("val").value_counts(sort=True).struct.field("val").alias("val"),
            pl.col("val").value_counts(sort=True).struct.field("count").alias("count"),
        )
    ).to_dict(as_series=False) == {
        "k": [0],
        "val": [
            [
                {"val": -3, "count": 1},
                {"val": -2, "count": 1},
                {"val": -1, "count": 1},
                {"val": 0, "count": 1},
                {"val": 1, "count": 1},
                {"val": 2, "count": 1},
                {"val": 3, "count": 1},
            ]
        ],
    }


def test_struct_null_count_strict_cast() -> None:
    s = pl.Series([{"a": None}]).cast(pl.Struct({"a": pl.Categorical}))
    assert s.dtype == pl.Struct([pl.Field("a", pl.Categorical)])
    assert s.to_list() == [{"a": None}]


def test_struct_get_field_by_index() -> None:
    df = pl.DataFrame({"val": [{"a": 1, "b": 2}]})
    expected = {"b": [2]}
    assert df.select(pl.all().struct[1]).to_dict(as_series=False) == expected


def test_struct_arithmetic_schema() -> None:
    q = pl.LazyFrame({"A": [1], "B": [2]})

    assert q.select(pl.struct("A") - pl.struct("B")).collect_schema()["A"] == pl.Struct(
        {"A": pl.Int64}
    )


def test_struct_field() -> None:
    df = pl.DataFrame(
        {
            "item": [
                {"name": "John", "age": 30, "car": None},
                {"name": "Alice", "age": 65, "car": "Volvo"},
            ]
        }
    )

    assert df.select(
        pl.col("item").struct.with_fields(
            pl.field("name").str.to_uppercase(), pl.field("car").fill_null("Mazda")
        )
    ).to_dict(as_series=False) == {
        "item": [
            {"name": "JOHN", "age": 30, "car": "Mazda"},
            {"name": "ALICE", "age": 65, "car": "Volvo"},
        ]
    }


def test_struct_field_recognized_as_renaming_expr_16480() -> None:
    q = pl.LazyFrame(
        {
            "foo": "bar",
            "my_struct": [{"x": 1, "y": 2}],
        }
    ).select(pl.col("my_struct").struct.field("x"))

    q = q.select("x")
    assert q.collect().to_dict(as_series=False) == {"x": [1]}


def test_struct_filter_chunked_16498() -> None:
    with pl.StringCache():
        N = 5
        df_orig1 = pl.DataFrame({"cat_a": ["remove"] * N, "cat_b": ["b"] * N})

        df_orig2 = pl.DataFrame({"cat_a": ["a"] * N, "cat_b": ["b"] * N})

        df = pl.concat([df_orig1, df_orig2], rechunk=False).cast(pl.Categorical)
        df = df.select(pl.struct(pl.all()).alias("s"))
        df = df.filter(pl.col("s").struct.field("cat_a") != pl.lit("remove"))
        assert df.shape == (5, 1)


def test_struct_field_dynint_nullable_16243() -> None:
    pl.select(pl.lit(None).fill_null(pl.struct(42)))


def test_struct_split_16536() -> None:
    df = pl.DataFrame({"struct": [{"a": {"a": {"a": 1}}}], "list": [[1]], "int": [1]})

    df = pl.concat([df, df, df, df], rechunk=False)
    assert df.filter(pl.col("int") == 1).shape == (4, 3)


def test_struct_wildcard_expansion_and_exclude() -> None:
    df = pl.DataFrame(
        {
            "id": [1, 2],
            "meta_data": [
                {"system_data": "to_remove", "user_data": "keep"},
                {"user_data": "keep_"},
            ],
        }
    )

    # ensure wildcard expansion is on input
    assert df.lazy().select(
        pl.col("meta_data").struct.with_fields("*")
    ).collect().schema["meta_data"].fields == [  # type: ignore[attr-defined]
        pl.Field("system_data", pl.String),
        pl.Field("user_data", pl.String),
        pl.Field("id", pl.Int64),
        pl.Field(
            "meta_data", pl.Struct({"system_data": pl.String, "user_data": pl.String})
        ),
    ]

    with pytest.raises(pl.exceptions.InvalidOperationError):
        df.lazy().select(
            pl.col("meta_data").struct.with_fields(pl.field("*").exclude("user_data"))
        ).collect()


def test_struct_chunked_gather_17603() -> None:
    df = pl.DataFrame(
        {
            "id": [0, 0, 1, 1],
            "a": [0, 1, 2, 3],
        }
    ).select("id", pl.struct("a"))
    df = pl.concat((df, df))

    assert df.select(pl.col("a").map_batches(lambda s: s).over("id")).to_dict(
        as_series=False
    ) == {
        "a": [
            {"a": 0},
            {"a": 1},
            {"a": 2},
            {"a": 3},
            {"a": 0},
            {"a": 1},
            {"a": 2},
            {"a": 3},
        ]
    }


def test_struct_out_nullability_from_arrow() -> None:
    df = pl.DataFrame(pd.DataFrame({"abc": [{"a": 1.0, "b": pd.NA}, pd.NA]}))
    res = df.select(a=pl.col("abc").struct.field("a"))
    assert res.to_dicts() == [{"a": 1.0}, {"a": None}]


def test_empty_struct_raise() -> None:
    with pytest.raises(ValueError):
        pl.struct()


def test_named_exprs() -> None:
    df = pl.DataFrame({"a": 1})
    schema = {"b": pl.Int64}
    res = df.select(pl.struct(schema=schema, b=pl.col("a")))
    assert res.to_dict(as_series=False) == {"b": [{"b": 1}]}
    assert res.schema["b"] == pl.Struct(schema)


def test_struct_outer_nullability_zip_18119() -> None:
    df = pl.Series("int", [0, 1, 2, 3], dtype=pl.Int64).to_frame()
    assert df.lazy().with_columns(
        result=pl.when(pl.col("int") >= 1).then(
            pl.struct(
                a=pl.when(pl.col("int") % 2 == 1).then(True),
                b=pl.when(pl.col("int") >= 2).then(False),
            )
        )
    ).collect().to_dict(as_series=False) == {
        "int": [0, 1, 2, 3],
        "result": [
            None,
            {"a": True, "b": None},
            {"a": None, "b": False},
            {"a": True, "b": False},
        ],
    }


def test_struct_group_by_shift_18107() -> None:
    df_in = pl.DataFrame(
        {
            "group": [1, 1, 1, 2, 2, 2],
            "id": [1, 2, 3, 4, 5, 6],
            "value": [
                {"lon": 20, "lat": 10},
                {"lon": 30, "lat": 20},
                {"lon": 40, "lat": 30},
                {"lon": 50, "lat": 40},
                {"lon": 60, "lat": 50},
                {"lon": 70, "lat": 60},
            ],
        }
    )

    assert df_in.group_by("group", maintain_order=True).agg(
        pl.col("value").shift(-1)
    ).to_dict(as_series=False) == {
        "group": [1, 2],
        "value": [
            [{"lon": 30, "lat": 20}, {"lon": 40, "lat": 30}, None],
            [{"lon": 60, "lat": 50}, {"lon": 70, "lat": 60}, None],
        ],
    }
