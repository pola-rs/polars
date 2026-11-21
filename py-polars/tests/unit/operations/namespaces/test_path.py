from __future__ import annotations

import pytest

import polars as pl
from polars.exceptions import (
    ColumnNotFoundError,
    InvalidOperationError,
    StructFieldNotFoundError,
)
from polars.testing import assert_frame_equal


@pytest.fixture
def df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [1, 2],
            "a": [
                {"b": {"c": 1, "d": "x"}, "l": [10, 20, 30]},
                {"b": {"c": 2, "d": "y"}, "l": [40, 50]},
            ],
        }
    )


def select(df: pl.DataFrame, *exprs: pl.Expr, lazy: bool) -> pl.DataFrame:
    if lazy:
        return df.lazy().select(*exprs).collect()
    return df.select(*exprs)


@pytest.mark.parametrize("lazy", [False, True])
def test_path_nested_struct(df: pl.DataFrame, lazy: bool) -> None:
    out = select(df, pl.path("a.b.c"), pl.path("a.b.d"), lazy=lazy)
    expected = df.select(
        pl.col("a").struct.field("b").struct.field("c"),
        pl.col("a").struct.field("b").struct.field("d"),
    )
    assert_frame_equal(out, expected)
    assert out.columns == ["c", "d"]

    # A single name is a column.
    assert_frame_equal(select(df, pl.path("id"), lazy=lazy), df.select("id"))
    assert_frame_equal(
        select(df, pl.path("a.b"), lazy=lazy),
        df.select(pl.col("a").struct.field("b")),
    )


@pytest.mark.parametrize("lazy", [False, True])
def test_path_in_expressions(df: pl.DataFrame, lazy: bool) -> None:
    out = select(df, (pl.path("a.b.c") * 10).alias("x"), lazy=lazy)
    assert out.to_dict(as_series=False) == {"x": [10, 20]}

    lf = df.lazy().filter(pl.path("a.b.d") == "y").select("id")
    assert lf.collect().to_dict(as_series=False) == {"id": [2]}

    out = df.with_columns(pl.path("a.b.c"))
    assert out.columns == ["id", "a", "c"]
    assert out["c"].to_list() == [1, 2]

    # A list at the end of the path is just a field's value.
    out = select(df, pl.path("a.l").list.get(-1), lazy=lazy)
    assert out.to_dict(as_series=False) == {"l": [30, 50]}


def test_field_unchanged(df: pl.DataFrame) -> None:
    # `pl.field` still only names a field of the struct in `struct.with_fields`.
    with pytest.raises(InvalidOperationError, match="outside of struct context"):
        df.select(pl.field("a"))
    with pytest.raises(InvalidOperationError, match=r"field `b\.c` does not exist"):
        df.select(pl.col("a").struct.with_fields(pl.field("b.c")))


@pytest.mark.parametrize("lazy", [False, True])
def test_path_with_fields_scope(df: pl.DataFrame, lazy: bool) -> None:
    out = select(
        df,
        pl.col("a").struct.with_fields(
            (pl.path("b.c") * 10).alias("c10"), pl.path("l").list.len().alias("n")
        ),
        lazy=lazy,
    )
    assert out.schema["a"] == pl.Struct(
        {
            "b": pl.Struct({"c": pl.Int64, "d": pl.String}),
            "l": pl.List(pl.Int64),
            "c10": pl.Int64,
            "n": pl.UInt32,
        }
    )
    assert out.select(pl.col("a").struct.field("c10", "n")).to_dict(
        as_series=False
    ) == {"c10": [10, 20], "n": [3, 2]}

    # The first name is a field of the struct, not a column.
    q = df.lazy().select(pl.col("a").struct.with_fields(pl.path("id.x")))
    with pytest.raises(InvalidOperationError, match=r"field `id` of path"):
        q.collect()

    # The output name is the last field, so this replaces the top-level `c`.
    s = pl.DataFrame({"s": [{"c": 0, "b": {"c": 5}}]})
    out = s.select(pl.col("s").struct.with_fields(pl.path("b.c")))
    assert out.to_dict(as_series=False) == {"s": [{"c": 5, "b": {"c": 5}}]}


def test_path_no_quoting() -> None:
    # Backticks are reserved, not a way to quote other names.
    df = pl.DataFrame({"x.y": [{"z": 1}], "s": [{"0": 2}]})
    for path, idx in [("`x.y`.z", 0), ("s.`0`", 2)]:
        with pytest.raises(
            InvalidOperationError,
            match=rf"(?s)found `` ` `` \(at index {idx}\).*`struct\.field`",
        ):
            df.select(pl.path(path))


@pytest.mark.parametrize("lazy", [False, True])
def test_path_always_path(lazy: bool) -> None:
    # The string is always a path, even if a column has that exact name.
    df = pl.DataFrame({"a": [{"b": 1}], "a.b": [2]})
    assert select(df, pl.path("a.b"), lazy=lazy).to_dict(as_series=False) == {"b": [1]}
    e = pl.path("a.b")
    assert (e.meta.output_name(), e.meta.root_names()) == ("b", ["a"])

    # Without a column `a`, it is an error, pointing at the literal column.
    df = pl.DataFrame({"a.b": [2], "x[0]": [3]})
    hint = r"use `pl\.col\(\"a\.b\"\)`"
    with pytest.raises(ColumnNotFoundError, match=hint):
        select(df, pl.path("a.b"), lazy=lazy)
    with pytest.raises(
        InvalidOperationError,
        match=r"(?s)invalid path \"x\[0\]\".*pl\.col\(\"x\[0\]\"\)",
    ):
        select(df, pl.path("x[0]"), lazy=lazy)


@pytest.mark.parametrize("lazy", [False, True])
def test_path_with_fields_literal_name(lazy: bool) -> None:
    # Inside `with_fields`, `pl.path` is always a path and `pl.field` an exact name.
    df = pl.DataFrame({"s": [{"x.y": 1, "x": {"y": 2}}]})
    out = select(
        df,
        pl.col("s").struct.with_fields(
            pl.field("x.y").alias("lit"),
            pl.path("x.y").alias("path"),
        ),
        lazy=lazy,
    )
    assert out.to_dict(as_series=False) == {
        "s": [{"x.y": 1, "x": {"y": 2}, "lit": 1, "path": 2}]
    }

    df = pl.DataFrame({"s": [{"a[": 1}]})
    q = df.lazy().select(pl.col("s").struct.with_fields(pl.path("a[")))
    with pytest.raises(InvalidOperationError, match=r"use `pl\.field\(\"a\[\"\)`"):
        q.collect()


@pytest.mark.parametrize(
    ("path", "msg"),
    [
        ("a..b", r"expected a field name, found `\.` \(at index 2\)"),
        ("a.", r"found end of path \(at index 2\)"),
        ("", r"found end of path \(at index 0\)"),
        ("a.l[0]", r"`\[` is reserved.*\(at index 3\)"),
        ("a.l[]", r"`\[` is reserved.*\(at index 3\)"),
        ("a.l[*]", r"`\[` is reserved.*\(at index 3\)"),
        ("a]", r"found `\]` \(at index 1\)"),
        ("`a", r"found `` ` `` \(at index 0\)"),
        ("a.*", r"`\.\*` is not supported yet \(at index 2\)"),
    ],
)
def test_path_parse_errors(df: pl.DataFrame, path: str, msg: str) -> None:
    with pytest.raises(InvalidOperationError, match=msg):
        df.select(pl.path(path))
    with pytest.raises(InvalidOperationError, match=r"invalid path"):
        df.lazy().select(pl.path(path)).collect()


@pytest.mark.parametrize(
    ("path", "column"),
    [
        ("_", "_"),
        ("_1", "_1"),
        ("a_b0", "a_b0"),
        ("äö1", "äö1"),
    ],
)
def test_path_names(path: str, column: str) -> None:
    df = pl.DataFrame({"s": [{column: 1}]})
    assert df.select(pl.path(f"s.{path}")).to_dict(as_series=False) == {column: [1]}
    df = pl.DataFrame({column: [{"x": 1}]})
    assert df.select(pl.path(f"{path}.x")).to_dict(as_series=False) == {"x": [1]}


@pytest.mark.parametrize(
    ("path", "msg"),
    [
        ("a.0", r"must start with a letter or `_`.*found `0` \(at index 2\)"),
        ("0", r"must start with a letter or `_`.*found `0` \(at index 0\)"),
        ("a.1b", r"must start with a letter or `_`.*found `1` \(at index 2\)"),
        ("a.$x", r"must start with a letter or `_`.*found `\$` \(at index 2\)"),
        ("a.b-c", r"only contain letters, digits and `_`.*found `-` \(at index 3\)"),
        ("my col", r"only contain letters, digits and `_`.*found ` ` \(at index 2\)"),
        ("a.x$", r"only contain letters, digits and `_`.*found `\$` \(at index 3\)"),
    ],
)
def test_path_reserved_names(path: str, msg: str) -> None:
    # Names are a letter or `_` followed by letters, digits or `_`; anything else is
    # reserved for future syntax.
    df = pl.DataFrame({"a": [{"0": 1}], "0": [2], "my col": [3]})
    with pytest.raises(InvalidOperationError, match=msg):
        df.select(pl.path(path))


def test_path_missing(df: pl.DataFrame) -> None:
    with pytest.raises(
        StructFieldNotFoundError,
        match=r'field "z" not found in struct at "a\.b".*available fields: \["c", "d"\]',
    ):
        df.select(pl.path("a.b.z"))
    with pytest.raises(ColumnNotFoundError, match=r'"nope"'):
        df.select(pl.path("nope.b"))
    with pytest.raises(ColumnNotFoundError, match=r'"nope"'):
        df.lazy().select(pl.path("nope")).collect()
    with pytest.raises(
        InvalidOperationError, match=r'cannot select field "x" of dtype i64 at "id"'
    ):
        df.select(pl.path("id.x"))
    with pytest.raises(
        InvalidOperationError,
        match=r'cannot select field "x" of dtype list\[i64\] at "a\.l"',
    ):
        df.select(pl.path("a.l.x"))

    # Inside `with_fields`, a missing root field.
    with pytest.raises(InvalidOperationError, match=r"field `z` of path \"z\""):
        df.select(pl.col("a").struct.with_fields(pl.path("z")))


def test_path_schema(df: pl.DataFrame) -> None:
    lf = df.lazy().select(pl.path("a.b.c"), pl.path("a.b.d"), pl.path("a.l"))
    assert lf.collect_schema() == pl.Schema(
        {"c": pl.Int64(), "d": pl.String(), "l": pl.List(pl.Int64)}
    )


def test_path_arg_sort_by(df: pl.DataFrame) -> None:
    out = df.select(pl.arg_sort_by(pl.path("a.b.c"), descending=True))
    assert out.to_dict(as_series=False) == {"c": [1, 0]}

    # Inside `with_fields` too.
    s = pl.DataFrame({"s": [{"x": 2}, {"x": 1}]})
    out = s.select(
        pl.col("s").struct.with_fields(pl.arg_sort_by(pl.path("x")).alias("i"))
    )
    assert out.to_dict(as_series=False) == {"s": [{"x": 2, "i": 1}, {"x": 1, "i": 0}]}


@pytest.mark.parametrize("lazy", [False, True])
def test_path_group_by_head_tail(lazy: bool) -> None:
    df = pl.DataFrame(
        {"id": [1, 2, 3], "a": [{"b": {"c": 1}}, {"b": {"c": 2}}, {"b": {"c": 2}}]}
    )
    gb = (df.lazy() if lazy else df).group_by(pl.path("a.b.c"), maintain_order=True)
    head = gb.head(1)
    tail = gb.tail(1)
    if isinstance(head, pl.LazyFrame):
        head = head.collect()
    if isinstance(tail, pl.LazyFrame):
        tail = tail.collect()
    assert head.select("c", "id").to_dict(as_series=False) == {
        "c": [1, 2],
        "id": [1, 2],
    }
    assert tail.select("c", "id").to_dict(as_series=False) == {
        "c": [1, 2],
        "id": [1, 3],
    }


@pytest.mark.parametrize("lazy", [False, True])
def test_path_group_by_head_literal_name(lazy: bool) -> None:
    # The key is the path, also when a column is named like the path.
    df = pl.DataFrame({"a": [{"b": 1}, {"b": 1}, {"b": 2}], "a.b": [7, 8, 9]})
    gb = (df.lazy() if lazy else df).group_by(pl.path("a.b"), maintain_order=True)
    out = gb.head(1)
    if isinstance(out, pl.LazyFrame):
        out = out.collect()
    assert out.select("b", "a.b").to_dict(as_series=False) == {
        "b": [1, 2],
        "a.b": [7, 9],
    }


def test_path_meta() -> None:
    assert pl.path("a.b.c").meta.root_names() == ["a"]
    assert pl.path("a.b.c").meta.output_name() == "c"
    assert pl.path("a").meta.output_name() == "a"
    assert (pl.path("a.b") + pl.col("x")).meta.root_names() == ["a", "x"]
    assert "a" in pl.path("a.b").meta.tree_format(return_as_string=True)
    # A string that doesn't parse as a path is taken as is (the plan then fails).
    assert pl.path("a[b").meta.output_name() == "a[b"

    # Inside `with_fields`, a path starts from a struct field, not a column.
    e = pl.col("s").struct.with_fields(pl.path("x.y") + pl.col("z"))
    assert e.meta.root_names() == ["s", "z"]
    assert e.meta.output_name() == "s"

    df1 = pl.DataFrame({"a": [{"b": 2}, {"b": 1}], "v": [1, 2]})
    df2 = pl.DataFrame({"a": [{"b": 1}], "w": [3]})
    # `align_frames` aligns on the output name of the key, which must be a column; a
    # path fails the same way as the hand-written expression.
    for on in [pl.path("a.b"), pl.col("a").struct.field("b")]:
        with pytest.raises(ColumnNotFoundError, match='"b"'):
            pl.align_frames(df1, df2, on=on)


def test_path_serde(df: pl.DataFrame) -> None:
    e = pl.path("a.b.c") * 2
    out = pl.Expr.deserialize(e.meta.serialize(), format="binary")
    assert_frame_equal(df.select(out), df.select(e))


def test_path_in_list_eval() -> None:
    df = pl.DataFrame(
        {"l": [[{"x": {"y": 1}}, {"x": {"y": 2}}]], "x": [{"y": 5}], "n": [5]}
    )
    for e in [
        pl.path("n"),
        pl.path("x.y"),
        pl.element().struct.field("x").struct.field("y") + pl.path("n"),
        pl.path("__PL_ELEMENT"),
        # A missing column in a nested `eval`'s input.
        pl.path("zz").list.eval(pl.element()),
    ]:
        with pytest.raises(
            InvalidOperationError, match=r"not allowed in `eval`.*pl\.element\(\)"
        ):
            df.lazy().select(pl.col("l").list.eval(e)).collect_schema()

    # Inside `with_fields` within `eval`, a path starts from a field of the element.
    out = df.select(
        pl.col("l").list.eval(
            pl.element().struct.with_fields(pl.path("x.y").alias("z"))
        )
    )
    assert out.to_dict(as_series=False) == {
        "l": [[{"x": {"y": 1}, "z": 1}, {"x": {"y": 2}, "z": 2}]]
    }
