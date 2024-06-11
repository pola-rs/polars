from typing import Any

import pytest

import polars as pl
import polars.selectors as cs


@pytest.fixture()
def df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "foo": ["A", "A", "B", "B", "C"],
            "N": [1, 2, 2, 4, 2],
            "bar": ["k", "l", "m", "m", "l"],
        }
    )


@pytest.mark.parametrize("input", [["foo", "bar"], cs.string()])
def test_partition_by(df: pl.DataFrame, input: Any) -> None:
    result = df.partition_by(input, maintain_order=True)
    expected = [
        {"foo": ["A"], "N": [1], "bar": ["k"]},
        {"foo": ["A"], "N": [2], "bar": ["l"]},
        {"foo": ["B", "B"], "N": [2, 4], "bar": ["m", "m"]},
        {"foo": ["C"], "N": [2], "bar": ["l"]},
    ]
    assert [a.to_dict(as_series=False) for a in result] == expected


def test_partition_by_include_key_false(df: pl.DataFrame) -> None:
    result = df.partition_by("foo", "bar", maintain_order=True, include_key=False)
    expected = [
        {"N": [1]},
        {"N": [2]},
        {"N": [2, 4]},
        {"N": [2]},
    ]
    assert [a.to_dict(as_series=False) for a in result] == expected


def test_partition_by_single(df: pl.DataFrame) -> None:
    result = df.partition_by("foo", maintain_order=True)
    expected = [
        {"foo": ["A", "A"], "N": [1, 2], "bar": ["k", "l"]},
        {"foo": ["B", "B"], "N": [2, 4], "bar": ["m", "m"]},
        {"foo": ["C"], "N": [2], "bar": ["l"]},
    ]
    assert [a.to_dict(as_series=False) for a in result] == expected


def test_partition_by_as_dict() -> None:
    df = pl.DataFrame({"a": ["one", "two", "one", "two"], "b": [1, 2, 3, 4]})
    result = df.partition_by(cs.all(), as_dict=True)
    result_first = result[("one", 1)]
    assert result_first.to_dict(as_series=False) == {"a": ["one"], "b": [1]}

    result = df.partition_by("a", as_dict=True)
    result_first = result[("one",)]
    assert result_first.to_dict(as_series=False) == {"a": ["one", "one"], "b": [1, 3]}


def test_partition_by_as_dict_include_keys_false() -> None:
    df = pl.DataFrame({"a": ["one", "two", "one", "two"], "b": [1, 2, 3, 4]})

    result = df.partition_by("a", include_key=False, as_dict=True)
    result_first = result[("one",)]
    assert result_first.to_dict(as_series=False) == {"b": [1, 3]}


def test_partition_by_as_dict_include_keys_false_maintain_order_false() -> None:
    df = pl.DataFrame({"a": ["one", "two", "one", "two"], "b": [1, 2, 3, 4]})
    with pytest.raises(ValueError):
        df.partition_by(["a"], maintain_order=False, include_key=False, as_dict=True)


@pytest.mark.slow()
def test_partition_by_as_dict_include_keys_false_large() -> None:
    # test with both as_dict and include_key=False
    df = pl.DataFrame(
        {
            "a": pl.int_range(0, 100, dtype=pl.UInt8, eager=True),
            "b": pl.int_range(0, 100, dtype=pl.UInt8, eager=True),
            "c": pl.int_range(0, 100, dtype=pl.UInt8, eager=True),
            "d": pl.int_range(0, 100, dtype=pl.UInt8, eager=True),
        }
    ).sample(n=100_000, with_replacement=True, shuffle=True)

    partitions = df.partition_by(["a", "b"], as_dict=True, include_key=False)
    assert all(key == value.row(0) for key, value in partitions.items())
