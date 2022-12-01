from __future__ import annotations

import typing
from datetime import datetime

import pandas as pd
import pyarrow as pa
import pytest

import polars as pl


def test_struct_various() -> None:
    df = pl.DataFrame(
        {"int": [1, 2], "str": ["a", "b"], "bool": [True, None], "list": [[1, 2], [3]]}
    )
    s = df.to_struct("my_struct")

    assert s.struct.fields == ["int", "str", "bool", "list"]
    assert s[0] == {"int": 1, "str": "a", "bool": True, "list": [1, 2]}
    assert s[1] == {"int": 2, "str": "b", "bool": None, "list": [3]}
    assert s.struct.field("list").to_list() == [[1, 2], [3]]
    assert s.struct.field("int").to_list() == [1, 2]

    assert df.to_struct("my_struct").struct.to_frame().frame_equal(df)
    assert s.struct._ipython_key_completions_() == s.struct.fields


def test_struct_to_list() -> None:
    assert pl.DataFrame(
        {"int": [1, 2], "str": ["a", "b"], "bool": [True, None], "list": [[1, 2], [3]]}
    ).select([pl.struct(pl.all()).alias("my_struct")]).to_series().to_list() == [
        {"int": 1, "str": "a", "bool": True, "list": [1, 2]},
        {"int": 2, "str": "b", "bool": None, "list": [3]},
    ]


def test_apply_to_struct() -> None:
    df = (
        pl.Series([None, 2, 3, 4])
        .apply(lambda x: {"a": x, "b": x * 2, "c": True, "d": [1, 2], "e": "foo"})
        .struct.to_frame()
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

    assert df.frame_equal(expected)


def test_rename_fields() -> None:
    df = pl.DataFrame({"int": [1, 2], "str": ["a", "b"], "bool": [True, None]})
    assert df.to_struct("my_struct").struct.rename_fields(["a", "b"]).struct.fields == [
        "a",
        "b",
    ]


def struct_unnesting() -> None:
    df = pl.DataFrame({"a": [1, 2]})
    out = df.select(
        [
            pl.all().alias("a_original"),
            pl.col("a")
            .apply(lambda x: (x, x * 2, x % 2 == 0))
            .struct.rename_fields(["a", "a_squared", "mod2eq0"])
            .alias("foo"),
        ]
    ).unnest("foo")

    expected = pl.DataFrame(
        {
            "a_original": [1, 2],
            "a": [1, 2],
            "a_squared": [2, 4],
            "mod2eq0": [False, True],
        }
    )

    assert out.frame_equal(expected)

    out = (
        df.lazy()
        .select(
            [
                pl.all().alias("a_original"),
                pl.col("a")
                .apply(lambda x: (x, x * 2, x % 2 == 0))
                .struct.rename_fields(["a", "a_squared", "mod2eq0"])
                .alias("foo"),
            ]
        )
        .unnest("foo")
        .collect()
    )
    out.frame_equal(expected)


def test_struct_function_expansion() -> None:
    df = pl.DataFrame(
        {"a": [1, 2, 3, 4], "b": ["one", "two", "three", "four"], "c": [9, 8, 7, 6]}
    )
    assert df.with_column(pl.struct(pl.col(["a", "b"])))["a"].struct.fields == [
        "a",
        "b",
    ]


def test_value_counts_expr() -> None:
    df = pl.DataFrame(
        {
            "id": ["a", "b", "b", "c", "c", "c", "d", "d"],
        }
    )

    out = (
        df.select(
            [
                pl.col("id").value_counts(sort=True),
            ]
        )
        .to_series()
        .to_list()
    )
    assert out == [
        {"id": "c", "counts": 3},
        {"id": "b", "counts": 2},
        {"id": "d", "counts": 2},
        {"id": "a", "counts": 1},
    ]


def test_value_counts_logical_type() -> None:
    # test logical type
    df = pl.DataFrame({"a": ["b", "c"]}).with_column(
        pl.col("a").cast(pl.Categorical).alias("ac")
    )
    out = df.select([pl.all().value_counts()])
    assert out["ac"].struct.field("ac").dtype == pl.Categorical
    assert out["a"].struct.field("a").dtype == pl.Utf8


def test_nested_struct() -> None:
    df = pl.DataFrame({"d": [1, 2, 3], "e": ["foo", "bar", "biz"]})
    # Nest the dataframe
    nest_l1 = df.to_struct("c").to_frame()
    # Add another column on the same level
    nest_l1 = nest_l1.with_column(pl.col("c").is_null().alias("b"))
    # Nest the dataframe again
    nest_l2 = nest_l1.to_struct("a").to_frame()

    assert isinstance(nest_l2.dtypes[0], pl.datatypes.Struct)
    assert [f.dtype for f in nest_l2.dtypes[0].fields] == nest_l1.dtypes
    assert isinstance(nest_l1.dtypes[0], pl.datatypes.Struct)


def test_eager_struct() -> None:
    with pytest.raises(pl.DuplicateError, match="multiple fields with name '' found"):
        s = pl.struct([pl.Series([1, 2, 3]), pl.Series(["a", "b", "c"])], eager=True)

    s = pl.struct(
        [pl.Series("a", [1, 2, 3]), pl.Series("b", ["a", "b", "c"])], eager=True
    )
    assert s.dtype == pl.Struct


def test_struct_to_pandas() -> None:
    df = pd.DataFrame([{"a": {"b": {"c": 2}}}])
    pl_df = pl.from_pandas(df)

    assert isinstance(pl_df.dtypes[0], pl.datatypes.Struct)

    assert pl_df.to_pandas().equals(df)


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
    assert df["a"].to_list() == [{"b": 1}, {"b": None}]


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
    assert df.select([pl.col("a").arr.to_struct()]).to_series().to_list() == [
        {"field_0": 1, "field_1": 2, "field_2": 3},
        {"field_0": 1, "field_1": 2, "field_2": None},
    ]

    df = pl.DataFrame({"a": [[1, 2], [1, 2, 3]]})
    assert df.select(
        [pl.col("a").arr.to_struct(name_generator=lambda idx: f"col_name_{idx}")]
    ).to_series().to_list() == [
        {"col_name_0": 1, "col_name_1": 2},
        {"col_name_0": 1, "col_name_1": 2},
    ]

    df = pl.DataFrame({"a": [[1, 2], [1, 2, 3]]})
    assert df.select(
        [pl.col("a").arr.to_struct(n_field_strategy="max_width")]
    ).to_series().to_list() == [
        {"field_0": 1, "field_1": 2, "field_2": None},
        {"field_0": 1, "field_1": 2, "field_2": 3},
    ]


def test_sort_df_with_list_struct() -> None:
    assert pl.DataFrame([{"a": 1, "b": [{"c": 1}]}]).sort("a").to_dict(False) == {
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
        [
            pl.col("list_of_struct").arr.head(1).alias("head"),
            pl.col("list_of_struct").arr.tail(1).alias("tail"),
        ]
    ).to_dict(
        False
    ) == {
        "list_of_struct": [
            [{"a": 1, "b": 4}, {"a": 3, "b": 6}],
            [{"a": 10, "b": 40}, {"a": 20, "b": 50}, {"a": 30, "b": 60}],
        ],
        "head": [[{"a": 1, "b": 4}], [{"a": 10, "b": 40}]],
        "tail": [[{"a": 3, "b": 6}], [{"a": 30, "b": 60}]],
    }


def test_struct_agg_list() -> None:
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

    assert df.groupby("group", maintain_order=True).agg_list().to_dict(False) == {
        "group": ["a", "b"],
        "col1": [
            [{"x": 1, "y": 100}, {"x": 2, "y": 200}],
            [{"x": 3, "y": 300}, {"x": 4, "y": 400}, {"x": 5, "y": 500}],
        ],
    }


def test_struct_empty_list_creation() -> None:
    payload = [[], [{"a": 1, "b": 2}, {"a": 3, "b": 4}, {"a": 5, "b": 6}], []]

    assert pl.DataFrame({"list_struct": payload}).to_dict(False) == {
        "list_struct": [[], [{"a": 1, "b": 2}, {"a": 3, "b": 4}, {"a": 5, "b": 6}], []]
    }

    # pop first
    payload = payload[1:]
    assert pl.DataFrame({"list_struct": payload}).to_dict(False) == {
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
    assert df.select([pl.col("list_struct").arr.first()]).to_dict(False) == {
        "list_struct": [{"a": 1, "b": 2}, {"a": 1, "b": 2}, {"a": 1, "b": 2}]
    }
    assert df.select([pl.col("list_struct").arr.last()]).to_dict(False) == {
        "list_struct": [{"a": 5, "b": 6}, {"a": 3, "b": 4}, {"a": 1, "b": 2}]
    }
    assert df.select([pl.col("list_struct").arr.get(0)]).to_dict(False) == {
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
    ).with_columns([pl.col("list_struct1").arr.concat("list_struct2").alias("result")])[
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
    ).with_columns([pl.col("list_struct").arr.reverse()]).to_dict(False) == {
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
    assert df.filter(pl.col("col1") == pl.col("col2")).to_dict(False) == {
        "col1": [{"a": 3, "b": 4}],
        "col2": [{"a": 3, "b": 4}],
    }


def test_struct_order() -> None:
    with pytest.raises(pl.ComputeError, match="struct orders must remain the same"):
        pl.DataFrame(
            {
                "col1": [{"a": 1, "b": 2}, {"b": 4, "a": 3}],
            }
        )

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


def test_struct_schema_on_append_extend_3452() -> None:
    housing1_data = [
        {
            "city": "Chicago",
            "address": "100 Main St",
            "price": 250000,
            "nbr_bedrooms": 3,
        },
        {
            "city": "New York",
            "address": "100 First Ave",
            "price": 450000,
            "nbr_bedrooms": 2,
        },
    ]

    housing2_data = [
        {
            "address": "303 Mockingbird Lane",
            "city": "Los Angeles",
            "nbr_bedrooms": 2,
            "price": 450000,
        },
        {
            "address": "404 Moldave Dr",
            "city": "Miami Beach",
            "nbr_bedrooms": 1,
            "price": 250000,
        },
    ]
    housing1, housing2 = pl.Series(housing1_data), pl.Series(housing2_data)
    with pytest.raises(
        pl.SchemaError,
        match=(
            "cannot append field with name: address to struct with field name: city,"
            " please check your schema"
        ),
    ):
        housing1.append(housing2, append_chunks=True)
    with pytest.raises(
        pl.SchemaError,
        match=(
            "cannot extend field with name: address to struct with field name: city,"
            " please check your schema"
        ),
    ):
        housing1.append(housing2, append_chunks=False)


def test_struct_arr_eval() -> None:
    df = pl.DataFrame(
        {"col_struct": [[{"a": 1, "b": 11}, {"a": 2, "b": 12}, {"a": 1, "b": 11}]]}
    )
    assert df.with_column(
        pl.col("col_struct").arr.eval(pl.element().first()).alias("first")
    ).to_dict(False) == {
        "col_struct": [[{"a": 1, "b": 11}, {"a": 2, "b": 12}, {"a": 1, "b": 11}]],
        "first": [[{"a": 1, "b": 11}]],
    }


@typing.no_type_check
def test_arr_unique() -> None:
    df = pl.DataFrame(
        {"col_struct": [[{"a": 1, "b": 11}, {"a": 2, "b": 12}, {"a": 1, "b": 11}]]}
    )
    # the order is unpredictable
    unique = df.with_column(pl.col("col_struct").arr.unique().alias("unique"))[
        "unique"
    ].to_list()
    assert len(unique) == 1
    unique_el = unique[0]
    assert len(unique_el) == 2
    assert {"a": 2, "b": 12} in unique_el
    assert {"a": 1, "b": 11} in unique_el


def test_is_in_struct() -> None:
    df = pl.DataFrame(
        {
            "struct_elem": [{"a": 1, "b": 11}, {"a": 1, "b": 90}],
            "struct_list": [
                [{"a": 1, "b": 11}, {"a": 2, "b": 12}, {"a": 3, "b": 13}],
                [{"a": 3, "b": 3}],
            ],
        }
    )

    assert df.filter(pl.col("struct_elem").is_in("struct_list")).to_dict(False) == {
        "struct_elem": [{"a": 1, "b": 11}],
        "struct_list": [[{"a": 1, "b": 11}, {"a": 2, "b": 12}, {"a": 3, "b": 13}]],
    }


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

    assert df.explode("data").to_dict(False) == {
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


def test_struct_groupby_field_agg_4216() -> None:
    df = pl.DataFrame([{"a": {"b": 1}, "c": 0}])
    assert df.groupby("c").agg(pl.col("a").struct.field("b").count()).to_dict(
        False
    ) == {"c": [0], "b": [1]}


def test_struct_getitem() -> None:
    assert pl.Series([{"a": 1, "b": 2}]).struct["b"].name == "b"
    assert pl.Series([{"a": 1, "b": 2}]).struct[0].name == "a"
    assert pl.Series([{"a": 1, "b": 2}]).struct[1].name == "b"
    assert pl.Series([{"a": 1, "b": 2}]).struct[-1].name == "b"
    assert pl.Series([{"a": 1, "b": 2}]).to_frame().select(
        [pl.col("").struct[0]]
    ).to_dict(False) == {"a": [1]}


def test_struct_broadcasting() -> None:
    df = pl.DataFrame(
        {
            "col1": [1, 2],
            "col2": [10, 20],
        }
    )

    assert (
        df.select(
            pl.struct(
                [
                    pl.lit("a").alias("a"),
                    pl.col("col1").alias("col1"),
                ]
            ).alias("my_struct")
        )
    ).to_dict(False) == {"my_struct": [{"a": "a", "col1": 1}, {"a": "a", "col1": 2}]}


def test_struct_supertype() -> None:
    assert pl.from_dicts(
        [{"vehicle": {"auto": "car"}}, {"vehicle": {"auto": None}}]
    ).to_dict(False) == {"vehicle": [{"auto": "car"}, {"auto": None}]}


def test_suffix_in_struct_creation() -> None:
    assert (
        pl.DataFrame(
            {
                "a": [1, 2],
                "b": [3, 4],
                "c": [5, 6],
            }
        ).select(pl.struct(pl.col(["a", "c"]).suffix("_foo")).alias("bar"))
    ).unnest("bar").to_dict(False) == {"a_foo": [1, 2], "c_foo": [5, 6]}


def test_concat_list_reverse_struct_fields() -> None:
    df = pl.DataFrame({"nums": [1, 2, 3, 4], "letters": ["a", "b", "c", "d"]}).select(
        [
            pl.col("nums"),
            pl.struct(["letters", "nums"]).alias("combo"),
            pl.struct(["nums", "letters"]).alias("reverse_combo"),
        ]
    )
    assert df.select(pl.concat_list(["combo", "reverse_combo"])).frame_equal(
        df.select(pl.concat_list(["combo", "combo"]))
    )
