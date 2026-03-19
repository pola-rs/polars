from __future__ import annotations

import pytest

import polars as pl
from polars.interchange.buffer import PolarsBuffer
from polars.interchange.protocol import CopyNotAllowedError, DlpackDeviceType


@pytest.mark.parametrize(
    ("data", "allow_copy"),
    [
        (pl.Series([1, 2]), True),
        (pl.Series([1, 2]), False),
        (pl.concat([pl.Series([1, 2]), pl.Series([1, 2])], rechunk=False), True),
    ],
)
def test_init(data: pl.Series, allow_copy: bool) -> None:
    buffer = PolarsBuffer(data, allow_copy=allow_copy)
    assert buffer._data.n_chunks() == 1


def test_init_invalid_input() -> None:
    s = pl.Series([1, 2])
    data = pl.concat([s, s], rechunk=False)

    with pytest.raises(
        CopyNotAllowedError, match="non-contiguous buffer must be made contiguous"
    ):
        PolarsBuffer(data, allow_copy=False)


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (pl.Series([1, 2], dtype=pl.Int8), 2),
        (pl.Series([1, 2], dtype=pl.Int64), 16),
        (pl.Series([1.4, 2.9, 3.0], dtype=pl.Float32), 12),
        (pl.Series([97, 98, 99, 195, 169, 195, 162, 195, 167], dtype=pl.UInt8), 9),
        (pl.Series([0, 1, 0, 2, 0], dtype=pl.UInt32), 20),
        (pl.Series([True, False], dtype=pl.Boolean), 1),
        (pl.Series([True] * 8, dtype=pl.Boolean), 1),
        (pl.Series([True] * 9, dtype=pl.Boolean), 2),
        (pl.Series([True] * 9, dtype=pl.Boolean)[5:], 2),
    ],
)
def test_bufsize(data: pl.Series, expected: int) -> None:
    buffer = PolarsBuffer(data)
    assert buffer.bufsize == expected


@pytest.mark.parametrize(
    "data",
    [
        pl.Series([1, 2]),
        pl.Series([1.2, 2.9, 3.0]),
        pl.Series([True, False]),
        pl.Series([True, False])[1:],
        pl.Series([97, 98, 97], dtype=pl.UInt8),
        pl.Series([], dtype=pl.Float32),
    ],
)
def test_ptr(data: pl.Series) -> None:
    buffer = PolarsBuffer(data)
    result = buffer.ptr
    # Memory address is unpredictable, so we just check if an integer is returned
    assert isinstance(result, int)


def test__dlpack__() -> None:
    data = pl.Series([1, 2])
    buffer = PolarsBuffer(data)
    with pytest.raises(NotImplementedError):
        buffer.__dlpack__()


def test__dlpack_device__() -> None:
    data = pl.Series([1, 2])
    buffer = PolarsBuffer(data)
    assert buffer.__dlpack_device__() == (DlpackDeviceType.CPU, None)


def test__repr__() -> None:
    data = pl.Series([True, False, True])
    buffer = PolarsBuffer(data)
    print(buffer.__repr__())
