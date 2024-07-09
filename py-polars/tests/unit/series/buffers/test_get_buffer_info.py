import pytest

import polars as pl
from polars.exceptions import ComputeError
from tests.unit.conftest import NUMERIC_DTYPES


@pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
def test_get_buffer_info_numeric(dtype: pl.DataType) -> None:
    s = pl.Series([1, 2, 3], dtype=dtype)
    assert s._get_buffer_info()[0] > 0


def test_get_buffer_info_bool() -> None:
    s = pl.Series([True, False, False, True])
    assert s._get_buffer_info()[0] > 0
    assert s[1:]._get_buffer_info()[1] == 1


def test_get_buffer_info_after_rechunk() -> None:
    s = pl.Series([1, 2, 3])
    ptr = s._get_buffer_info()[0]
    assert isinstance(ptr, int)

    s2 = s.append(pl.Series([1, 2]))
    ptr2 = s2.rechunk()._get_buffer_info()[0]
    assert ptr != ptr2


def test_get_buffer_info_invalid_data_type() -> None:
    s = pl.Series(["a", "bc"])

    msg = "`_get_buffer_info` not implemented for non-physical type str; try to select a buffer first"
    with pytest.raises(TypeError, match=msg):
        s._get_buffer_info()


def test_get_buffer_info_chunked() -> None:
    s1 = pl.Series([1, 2])
    s2 = pl.Series([3, 4])
    s = pl.concat([s1, s2], rechunk=False)

    with pytest.raises(ComputeError):
        s._get_buffer_info()
