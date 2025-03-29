from types import MappingProxyType

import pytest

import polars as pl
from polars.exceptions import ColumnNotFoundError


def test_lazy_rename() -> None:
    lf = pl.LazyFrame({"x": [1], "y": [2]})

    result = lf.rename({"y": "x", "x": "y"}).select(["x", "y"]).collect()
    assert result.to_dict(as_series=False) == {"x": [2], "y": [1]}

    # the `strict` param controls whether we fail on columns not found in the frame
    remap_colnames = {"b": "a", "y": "x", "a": "b", "x": "y"}
    with pytest.raises(ColumnNotFoundError, match="'b' is invalid"):
        lf.rename(remap_colnames).collect()

    result = lf.rename(remap_colnames, strict=False).collect()
    assert result.to_dict(as_series=False) == {"x": [2], "y": [1]}


def test_remove_redundant_mapping_4668() -> None:
    lf = pl.LazyFrame([["a"]] * 2, ["A", "B "]).lazy()
    clean_name_dict = {x: " ".join(x.split()) for x in lf.collect_schema()}
    lf = lf.rename(clean_name_dict)
    assert lf.collect_schema().names() == ["A", "B"]


def test_rename_mapping_19400() -> None:
    # use a mapping type that is not a dict
    mapping = MappingProxyType({"a": "b", "b": "c"})

    assert pl.LazyFrame({"a": [1], "b": [2]}).rename(mapping).collect().to_dict(
        as_series=False
    ) == {"b": [1], "c": [2]}
