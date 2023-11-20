from __future__ import annotations

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

    assert_frame_equal(df.to_struct("my_struct").struct.unnest(), df)
    assert s.struct._ipython_key_completions_() == s.struct.fields


def test_rename_fields() -> None:
    df = pl.DataFrame({"int": [1, 2], "str": ["a", "b"], "bool": [True, None]})
    assert df.to_struct("my_struct").struct.rename_fields(["a", "b"]).struct.fields == [
        "a",
        "b",
    ]


def test_struct_json_encode() -> None:
    assert pl.DataFrame(
        {"a": [{"a": [1, 2], "b": [45]}, {"a": [9, 1, 3], "b": None}]}
    ).with_columns(pl.col("a").struct.json_encode().alias("encoded")).to_dict(
        as_series=False
    ) == {
        "a": [{"a": [1, 2], "b": [45]}, {"a": [9, 1, 3], "b": None}],
        "encoded": ['{"a":[1,2],"b":[45]}', '{"a":[9,1,3],"b":null}'],
    }
