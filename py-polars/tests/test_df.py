from polars import DataFrame
import pytest


def test_init():
    df = DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})

    # length mismatch
    with pytest.raises(RuntimeError):
        df = DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0, 4.0]})


def test_selection():
    df = DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0], "c": ["a", "b", "c"]})

    assert df["a"].dtype == "i64"
    assert df["b"].dtype == "f64"
    assert df["c"].dtype == "str"

    assert df[["a", "b"]].columns == ["a", "b"]
    assert df[[True, False, True]].height == 2

    assert df[[True, False, True], "b"].shape == (2, 1)
    assert df[[True, False, False], ["a", "b"]].shape == (1, 2)

    assert df[[0, 1], "b"].shape == (2, 1)
    assert df[[2], ["a", "b"]].shape == (1, 2)
