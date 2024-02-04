import numpy as np
import numpy.typing as npt
import pytest

import polars as pl


def test_view() -> None:
    s = pl.Series("a", [1.0, 2.5, 3.0])
    result = s._view()
    assert isinstance(result, np.ndarray)
    assert np.all(result == np.array([1.0, 2.5, 3.0]))


def test_view_nulls() -> None:
    s = pl.Series("b", [1, 2, None])
    assert s.has_validity()
    with pytest.raises(AssertionError):
        s._view()


def test_view_nulls_sliced() -> None:
    s = pl.Series("b", [1, 2, None])
    sliced = s[:2]
    assert np.all(sliced._view() == np.array([1, 2]))
    assert not sliced.has_validity()


def test_view_ub() -> None:
    # this would be UB if the series was dropped and not passed to the view
    s = pl.Series([3, 1, 5])
    result = s.sort()._view()
    assert np.sum(result) == 9


def test_view_deprecated() -> None:
    s = pl.Series("a", [1.0, 2.5, 3.0])
    with pytest.deprecated_call():
        result = s.view()
    assert isinstance(result, np.ndarray)
    assert np.all(result == np.array([1.0, 2.5, 3.0]))


def test_numpy_disambiguation() -> None:
    a = np.array([1, 2])
    df = pl.DataFrame({"a": a})
    result = df.with_columns(b=a).to_dict(as_series=False)  # type: ignore[arg-type]
    expected = {
        "a": [1, 2],
        "b": [1, 2],
    }
    assert result == expected


def test_series_to_numpy_bool() -> None:
    s = pl.Series([True, False])
    result = s.to_numpy(use_pyarrow=False)
    assert s.to_list() == result.tolist()
    assert result.dtype == np.bool_


def test_series_to_numpy_bool_with_nulls() -> None:
    s = pl.Series([True, False, None])
    result = s.to_numpy(use_pyarrow=False)
    assert s.to_list() == result.tolist()
    assert result.dtype == np.object_


@pytest.mark.parametrize(
    ("dtype", "expected_dtype"),
    [
        (pl.Int8, np.float32),
        (pl.Int16, np.float32),
        (pl.Int32, np.float64),
        (pl.Int64, np.float64),
        (pl.UInt8, np.float32),
        (pl.UInt16, np.float32),
        (pl.UInt32, np.float64),
        (pl.UInt64, np.float64),
        (pl.Float32, np.float32),
        (pl.Float64, np.float64),
    ],
)
def test_series_to_numpy_numeric_with_nulls(
    dtype: pl.PolarsDataType, expected_dtype: npt.DTypeLike
) -> None:
    s = pl.Series([1, 2, None], dtype=dtype, strict=False)
    result = s.to_numpy(use_pyarrow=False)
    assert result.dtype == expected_dtype
