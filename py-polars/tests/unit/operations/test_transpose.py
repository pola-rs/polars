from datetime import date
from typing import Iterator

import pytest

import polars as pl
from polars.testing import assert_frame_equal


def test_transpose_supertype() -> None:
    assert pl.DataFrame(
        {"a": [1, 2, 3], "b": ["foo", "bar", "ham"]}
    ).transpose().to_dict(False) == {
        "column_0": ["1", "foo"],
        "column_1": ["2", "bar"],
        "column_2": ["3", "ham"],
    }


def test_transpose_struct() -> None:
    with pytest.raises(pl.ComputeError, match=r"cannot transpose with supertype: str"):
        pl.DataFrame(
            {
                "a": ["foo", "bar", "ham"],
                "b": [
                    {"a": date(2022, 1, 1), "b": True},
                    {"a": date(2022, 1, 2), "b": False},
                    {"a": date(2022, 1, 3), "b": False},
                ],
            }
        ).transpose()

    # nothing useful, but tests if we don't have UB
    assert pl.DataFrame(
        {
            "b": [
                {"a": date(2022, 1, 1), "b": True},
                {"a": date(2022, 1, 2), "b": False},
                {"a": date(2022, 1, 3), "b": False},
            ]
        }
    ).transpose().to_dict(False) == {
        "column_0": [{"a": None, "b": None}],
        "column_1": [{"a": None, "b": None}],
        "column_2": [{"a": None, "b": None}],
    }


def test_transpose_arguments() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
    expected = pl.DataFrame(
        {
            "column": ["a", "b"],
            "column_0": [1, 1],
            "column_1": [2, 2],
            "column_2": [3, 3],
        }
    )
    out = df.transpose(include_header=True)
    assert_frame_equal(expected, out)

    out = df.transpose(include_header=False, column_names=["a", "b", "c"])
    expected = pl.DataFrame(
        {
            "a": [1, 1],
            "b": [2, 2],
            "c": [3, 3],
        }
    )
    assert_frame_equal(expected, out)

    out = df.transpose(
        include_header=True, header_name="foo", column_names=["a", "b", "c"]
    )
    expected = pl.DataFrame(
        {
            "foo": ["a", "b"],
            "a": [1, 1],
            "b": [2, 2],
            "c": [3, 3],
        }
    )
    assert_frame_equal(expected, out)

    def name_generator() -> Iterator[str]:
        base_name = "my_column_"
        count = 0
        while True:
            yield f"{base_name}{count}"
            count += 1

    out = df.transpose(include_header=False, column_names=name_generator())
    expected = pl.DataFrame(
        {
            "my_column_0": [1, 1],
            "my_column_1": [2, 2],
            "my_column_2": [3, 3],
        }
    )
    assert_frame_equal(expected, out)
