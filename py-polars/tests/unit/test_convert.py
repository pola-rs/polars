from __future__ import annotations

import pytest

import polars as pl


def test_schema_inference_from_rows() -> None:
    # these have to upcast to float
    result = pl.from_records([[1, 2.1, 3], [4, 5, 6.4]])
    assert result.to_dict(as_series=False) == {
        "column_0": [1.0, 2.1, 3.0],
        "column_1": [4.0, 5.0, 6.4],
    }

    result = pl.from_dicts([{"a": 1, "b": 2}, {"a": 3.1, "b": 4.5}])
    assert result.to_dict(as_series=False) == {
        "a": [1.0, 3.1],
        "b": [2.0, 4.5],
    }


def test_from_dicts_nested_nulls() -> None:
    result = pl.from_dicts([{"a": [None, None]}, {"a": [1, 2]}])
    assert result.to_dict(as_series=False) == {"a": [[None, None], [1, 2]]}


def test_from_dicts_empty() -> None:
    with pytest.raises(pl.NoDataError, match="no data, cannot infer schema"):
        pl.from_dicts([])


def test_from_dicts_all_cols_6716() -> None:
    dicts = [{"a": None} for _ in range(20)] + [{"a": "crash"}]

    with pytest.raises(
        pl.ComputeError, match="make sure that all rows have the same schema"
    ):
        pl.from_dicts(dicts, infer_schema_length=20)
    assert pl.from_dicts(dicts, infer_schema_length=None).dtypes == [pl.String]
