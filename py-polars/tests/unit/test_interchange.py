import sys
from typing import Any

import pandas as pd
import pytest

import polars as pl
from polars.testing import assert_frame_equal


def test_interchange() -> None:
    df = pl.DataFrame({"a": [1, 2], "b": [3.0, 4.0], "c": ["foo", "bar"]})
    dfi = df.__dataframe__()

    # Testing some random properties to make sure conversion happened correctly
    assert dfi.num_rows() == 2
    assert dfi.get_column(0).dtype[1] == 64
    assert dfi.get_column_by_name("c").get_buffers()["data"][0].bufsize == 6


def test_interchange_pyarrow_required(monkeypatch: Any) -> None:
    monkeypatch.setattr(pl.internals.dataframe.frame, "_PYARROW_AVAILABLE", False)

    df = pl.DataFrame({"a": [1, 2]})
    with pytest.raises(ImportError, match="pyarrow"):
        df.__dataframe__()


def test_interchange_pyarrow_min_version(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        pl.internals.dataframe.frame.pa,  # type: ignore[attr-defined]
        "__version__",
        "10.0.0",
    )

    df = pl.DataFrame({"a": [1, 2]})
    with pytest.raises(ImportError, match="pyarrow"):
        df.__dataframe__()


def test_interchange_categorical() -> None:
    df = pl.DataFrame({"a": ["foo", "bar"]}, schema={"a": pl.Categorical})

    # Conversion requires copy
    dfi = df.__dataframe__(allow_copy=True)
    assert dfi.get_column_by_name("a").dtype[0] == 23  # 23 signifies categorical dtype

    # If copy not allowed, throws an error
    with pytest.raises(NotImplementedError, match="categorical"):
        df.__dataframe__(allow_copy=False)


def test_from_dataframe() -> None:
    df = pl.DataFrame({"a": [1, 2], "b": [3.0, 4.0], "c": ["foo", "bar"]})
    dfi = df.__dataframe__()
    result = pl.from_dataframe(dfi)
    assert_frame_equal(result, df)


@pytest.mark.xfail(
    sys.version_info < (3, 8),
    reason="Pandas does not implement the protocol on Python 3.7",
)
def test_from_dataframe_pandas() -> None:
    data = {"a": [1, 2], "b": [3.0, 4.0], "c": ["foo", "bar"]}

    # Pandas dataframe
    df = pd.DataFrame(data)
    result = pl.from_dataframe(df)
    expected = pl.DataFrame(data)
    assert_frame_equal(result, expected)


@pytest.mark.xfail(
    sys.version_info < (3, 8),
    reason="Pandas does not implement the protocol on Python 3.7",
)
def test_from_dataframe_allow_copy() -> None:
    # Zero copy only allowed when input is already a Polars dataframe
    df = pl.DataFrame({"a": [1, 2]})
    result = pl.from_dataframe(df, allow_copy=True)
    assert_frame_equal(result, df)

    df1_pandas = pd.DataFrame({"a": [1, 2]})
    result_from_pandas = pl.from_dataframe(df1_pandas, allow_copy=False)
    assert_frame_equal(result_from_pandas, df)

    # Zero copy cannot be guaranteed for other inputs at this time
    df2_pandas = pd.DataFrame({"a": ["A", "B"]})
    with pytest.raises(RuntimeError):
        pl.from_dataframe(df2_pandas, allow_copy=False)


def test_from_dataframe_invalid_type() -> None:
    df = [[1, 2], [3, 4]]
    with pytest.raises(TypeError):
        pl.from_dataframe(df)


def test_from_dataframe_pyarrow_required(monkeypatch: Any) -> None:
    monkeypatch.setattr(pl.convert, "_PYARROW_AVAILABLE", False)

    df = pl.DataFrame({"a": [1, 2]})
    with pytest.raises(ImportError, match="pyarrow"):
        pl.from_dataframe(df.__dataframe__())

    # 'Converting' from a Polars dataframe does not hit this requirement
    result = pl.from_dataframe(df)
    assert_frame_equal(result, df)


def test_from_dataframe_pyarrow_min_version(monkeypatch: Any) -> None:
    dfi = pl.DataFrame({"a": [1, 2]}).__dataframe__()

    monkeypatch.setattr(
        pl.convert.pa,  # type: ignore[attr-defined]
        "__version__",
        "10.0.0",
    )

    with pytest.raises(ImportError, match="pyarrow"):
        pl.from_dataframe(dfi)
