import pytest

import polars as pl


def test_format() -> None:
    df = pl.DataFrame({"a": ["a", "b", "c"], "b": [1, 2, 3]})

    out = df.select([pl.format("foo_{}_bar_{}", pl.col("a"), "b").alias("fmt")])
    assert out["fmt"].to_list() == ["foo_a_bar_1", "foo_b_bar_2", "foo_c_bar_3"]


def test_format_with_names() -> None:
    df = pl.DataFrame({"a": ["a", "b", "c"], "b": [1, 2, 3]})

    out = df.select([pl.format("foo_{}_bar_{b}", pl.col("a"), b="b").alias("fmt")])
    assert out["fmt"].to_list() == ["foo_a_bar_1", "foo_b_bar_2", "foo_c_bar_3"]


def test_format_with_no_placeholders() -> None:
    df = pl.DataFrame({"a": ["a", "b", "c"], "b": [1, 2, 3]})

    # Need with_columns instead of select, because all formatted strings are the same
    out = df.with_columns([pl.format("foo").alias("fmt")])
    assert out["fmt"].to_list() == ["foo", "foo", "foo"]


def test_format_with_only_placeholders() -> None:
    df = pl.DataFrame({"a": ["a", "b", "c"], "b": [1, 2, 3]})

    out = df.select([pl.format("{a}", a=pl.col("a")).alias("fmt")])
    assert out["fmt"].to_list() == ["a", "b", "c"]


def test_format_literal_brackets() -> None:
    df = pl.DataFrame({"a": ["a", "b", "c"]})

    out = df.with_columns([pl.format("test{{a}}").alias("fmt")])
    assert out["fmt"].to_list() == ["test{a}", "test{a}", "test{a}"]


def test_format_empty_literal_brackets() -> None:
    df = pl.DataFrame({"a": ["a", "b", "c"]})

    out = df.with_columns([pl.format("test{{}}").alias("fmt")])
    assert out["fmt"].to_list() == ["test{}", "test{}", "test{}"]


def test_format_raises_on_wrong_number_of_named_arguments() -> None:
    with pytest.raises(
        ValueError,
        match="Expected 2 named placeholders, but got 1 keyword arguments.",
    ):
        pl.format("foo_{a}_bar_{b}", a=pl.col("a"))


def test_format_raises_on_wrong_number_of_unnamed_arguments() -> None:
    with pytest.raises(
        ValueError,
        match="Expected 1 unnamed placeholders, but got 0 arguments.",
    ):
        pl.format("foo_{}_bar_{b}", a=pl.col("a"), b=pl.col("b"))


def test_format_specifiers_unsupported() -> None:
    with pytest.raises(
        ValueError,
        match="Formatting specifiers and conversion flags are not supported in polars.format()",
    ):
        pl.format("foo_{a:s}", a=pl.col("a"))


def test_format_conversion_flags_unsupported() -> None:
    with pytest.raises(
        ValueError,
        match="Formatting specifiers and conversion flags are not supported in polars.format()",
    ):
        pl.format("foo_{a!r}", a=pl.col("a"))
