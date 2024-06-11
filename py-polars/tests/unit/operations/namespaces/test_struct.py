from __future__ import annotations

import datetime
from collections import OrderedDict

import polars as pl
from polars.testing import assert_frame_equal


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
    assert s.struct["list"].to_list() == [[1, 2], [3]]
    assert s.struct["int"].to_list() == [1, 2]

    for s, expected_name in (
        (df.to_struct(), ""),
        (df.to_struct("my_struct"), "my_struct"),
    ):
        assert s.name == expected_name
        assert_frame_equal(s.struct.unnest(), df)
        assert s.struct._ipython_key_completions_() == s.struct.fields


def test_rename_fields() -> None:
    df = pl.DataFrame({"int": [1, 2], "str": ["a", "b"], "bool": [True, None]})
    s = df.to_struct("my_struct").struct.rename_fields(["a", "b"])
    assert s.struct.fields == ["a", "b"]


def test_struct_json_encode() -> None:
    assert pl.DataFrame(
        {"a": [{"a": [1, 2], "b": [45]}, {"a": [9, 1, 3], "b": None}]}
    ).with_columns(pl.col("a").struct.json_encode().alias("encoded")).to_dict(
        as_series=False
    ) == {
        "a": [{"a": [1, 2], "b": [45]}, {"a": [9, 1, 3], "b": None}],
        "encoded": ['{"a":[1,2],"b":[45]}', '{"a":[9,1,3],"b":null}'],
    }


def test_struct_json_encode_logical_type() -> None:
    df = pl.DataFrame(
        {
            "a": [
                {
                    "a": [datetime.date(1997, 1, 1)],
                    "b": [datetime.datetime(2000, 1, 29, 10, 30)],
                    "c": [datetime.timedelta(1, 25)],
                }
            ]
        }
    ).select(pl.col("a").struct.json_encode().alias("encoded"))
    assert df.to_dict(as_series=False) == {
        "encoded": ['{"a":["1997-01-01"],"b":["2000-01-29 10:30:00"],"c":["PT86425S"]}']
    }


def test_map_fields() -> None:
    df = pl.DataFrame({"x": {"a": 1, "b": 2}})
    assert df.schema == OrderedDict([("x", pl.Struct({"a": pl.Int64, "b": pl.Int64}))])
    df = df.select(pl.col("x").name.map_fields(lambda x: x.upper()))
    assert df.schema == OrderedDict([("x", pl.Struct({"A": pl.Int64, "B": pl.Int64}))])


def test_prefix_suffix_fields() -> None:
    df = pl.DataFrame({"x": {"a": 1, "b": 2}})

    prefix_df = df.select(pl.col("x").name.prefix_fields("p_"))
    assert prefix_df.schema == OrderedDict(
        [("x", pl.Struct({"p_a": pl.Int64, "p_b": pl.Int64}))]
    )

    suffix_df = df.select(pl.col("x").name.suffix_fields("_f"))
    assert suffix_df.schema == OrderedDict(
        [("x", pl.Struct({"a_f": pl.Int64, "b_f": pl.Int64}))]
    )


def test_struct_alias_prune_15401() -> None:
    df = pl.DataFrame({"a": []}, schema={"a": pl.Struct({"b": pl.Int8})})
    assert df.select(pl.col("a").alias("c").struct.field("b")).columns == ["b"]


def test_empty_list_eval_schema_5734() -> None:
    df = pl.DataFrame({"a": [[{"b": 1, "c": 2}]]})
    assert df.filter(False).select(
        pl.col("a").list.eval(pl.element().struct.field("b"))
    ).schema == {"a": pl.List(pl.Int64)}
