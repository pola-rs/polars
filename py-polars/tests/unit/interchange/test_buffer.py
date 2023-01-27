import pytest

import polars as pl
from polars.internals.interchange.buffer import PolarsBuffer
from polars.internals.interchange.dataframe_protocol import DlpackDeviceType


@pytest.mark.parametrize(
    ("data", "allow_copy"),
    [
        (pl.Series([1, 2]), True),
        (pl.Series([1, 2]), False),
        (pl.concat([pl.Series([1, 2]), pl.Series([1, 2])], rechunk=False), True),
    ],
)
def test_init(data, allow_copy):
    buffer = PolarsBuffer(data, allow_copy=allow_copy)
    assert buffer._data.n_chunks() == 1


def test_init_invalid_input():
    s = pl.Series([1, 2])
    data = pl.concat([s, s], rechunk=False)

    with pytest.raises(RuntimeError):
        PolarsBuffer(data, allow_copy=False)


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (pl.Series([1, 2], dtype=pl.Int8), 2),
        (pl.Series([1, 2], dtype=pl.Int64), 16),
        (pl.Series(["a", "bc", "éâç"], dtype=pl.Utf8), 9),
        (pl.Series([True, False], dtype=pl.Boolean), 2),
    ],
)
def test_bufsize(data, expected):
    buffer = PolarsBuffer(data)
    assert buffer.bufsize == expected


def test_ptr():
    data = pl.Series([1, 2])
    buffer = PolarsBuffer(data)
    result = buffer.ptr
    # Memory address is unpredictable - so we just check if an integer is returned
    assert isinstance(result, int)


@pytest.mark.skip("Not implemented yet")
def test_ptr_boolean():
    data = pl.Series([True, False])
    buffer = PolarsBuffer(data)
    result = buffer.ptr
    # Memory address is unpredictable - so we just check if an integer is returned
    assert isinstance(result, int)


def test__dlpack__():
    data = pl.Series([1, 2])
    buffer = PolarsBuffer(data)
    with pytest.raises(NotImplementedError):
        buffer.__dlpack__()


def test__dlpack_device__():
    data = pl.Series([1, 2])
    buffer = PolarsBuffer(data)
    assert buffer.__dlpack_device__() == (DlpackDeviceType.CPU, None)


def test__repr__():

    data = pl.Series([True, False, True])
    buffer = PolarsBuffer(data)
    print(buffer.__repr__())

    pass
