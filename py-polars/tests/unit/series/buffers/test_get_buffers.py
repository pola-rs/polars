from datetime import date
from typing import cast

import pytest

import polars as pl
from polars.testing import assert_series_equal


def test_get_buffers_only_values() -> None:
    s = pl.Series([1, 2, 3])

    result = s._get_buffers()

    assert_series_equal(result["values"], s)
    assert result["validity"] is None
    assert result["offsets"] is None


def test_get_buffers_with_validity() -> None:
    s = pl.Series([1.5, None, 3.5])

    result = s._get_buffers()

    expected_values = pl.Series([1.5, 0.0, 3.5])
    assert_series_equal(result["values"], expected_values)

    validity = cast(pl.Series, result["validity"])
    expected_validity = pl.Series([True, False, True])
    assert_series_equal(validity, expected_validity)

    assert result["offsets"] is None


def test_get_buffers_string_type() -> None:
    s = pl.Series(["a", "bc", None, "éâç", ""])

    result = s._get_buffers()

    expected_values = pl.Series(
        [97, 98, 99, 195, 169, 195, 162, 195, 167], dtype=pl.UInt8
    )
    assert_series_equal(result["values"], expected_values)

    validity = cast(pl.Series, result["validity"])
    expected_validity = pl.Series([True, True, False, True, True])
    assert_series_equal(validity, expected_validity)

    offsets = cast(pl.Series, result["offsets"])
    expected_offsets = pl.Series([0, 1, 3, 3, 9, 9], dtype=pl.Int64)
    assert_series_equal(offsets, expected_offsets)


def test_get_buffers_logical_sliced() -> None:
    s = pl.Series([date(1970, 1, 1), None, date(1970, 1, 3)])[1:]

    result = s._get_buffers()

    expected_values = pl.Series([0, 2], dtype=pl.Int32)
    assert_series_equal(result["values"], expected_values)

    validity = cast(pl.Series, result["validity"])
    expected_validity = pl.Series([False, True])
    assert_series_equal(validity, expected_validity)

    assert result["offsets"] is None


def test_get_buffers_chunked() -> None:
    s = pl.Series([1, 2, None, 4], dtype=pl.UInt8)
    s_chunked = pl.concat([s[:2], s[2:]], rechunk=False)

    result = s_chunked._get_buffers()

    expected_values = pl.Series([1, 2, 0, 4], dtype=pl.UInt8)
    assert_series_equal(result["values"], expected_values)
    assert result["values"].n_chunks() == 2

    validity = cast(pl.Series, result["validity"])
    expected_validity = pl.Series([True, True, False, True])
    assert_series_equal(validity, expected_validity)
    assert validity.n_chunks() == 2


def test_get_buffers_chunked_string_type() -> None:
    s = pl.Series(["a", "bc", None, "éâç", ""])
    s_chunked = pl.concat([s[:2], s[2:]], rechunk=False)

    result = s_chunked._get_buffers()

    expected_values = pl.Series(
        [97, 98, 99, 195, 169, 195, 162, 195, 167], dtype=pl.UInt8
    )
    assert_series_equal(result["values"], expected_values)
    assert result["values"].n_chunks() == 1

    validity = cast(pl.Series, result["validity"])
    expected_validity = pl.Series([True, True, False, True, True])
    assert_series_equal(validity, expected_validity)
    assert validity.n_chunks() == 1

    offsets = cast(pl.Series, result["offsets"])
    expected_offsets = pl.Series([0, 1, 3, 3, 9, 9], dtype=pl.Int64)
    assert_series_equal(offsets, expected_offsets)
    assert offsets.n_chunks() == 1


def test_get_buffers_unsupported_data_type() -> None:
    s = pl.Series([[1, 2], [3]])

    msg = "`_get_buffers` not implemented for `dtype` list\\[i64\\]"
    with pytest.raises(TypeError, match=msg):
        s._get_buffers()
