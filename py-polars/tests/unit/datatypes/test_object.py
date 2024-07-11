import io
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

import polars as pl
from polars.exceptions import ComputeError


def test_series_init_instantiated_object() -> None:
    s = pl.Series([object(), object()], dtype=pl.Object())
    assert isinstance(s, pl.Series)
    assert isinstance(s.dtype, pl.Object)


def test_object_empty_filter_5911() -> None:
    df = pl.DataFrame(
        data=[
            (1, "dog", {}),
        ],
        schema=[
            ("pet_id", pl.Int64),
            ("pet_type", pl.Categorical),
            ("pet_obj", pl.Object),
        ],
        orient="row",
    )

    empty_df = df.filter(pl.col("pet_type") == "cat")
    out = empty_df.select(["pet_obj"])
    assert out.dtypes == [pl.Object]
    assert out.shape == (0, 1)


def test_object_in_struct() -> None:
    np_a = np.array([1, 2, 3])
    np_b = np.array([4, 5, 6])
    df = pl.DataFrame({"A": [1, 2], "B": pl.Series([np_a, np_b], dtype=pl.Object)})

    with pytest.raises(pl.exceptions.InvalidOperationError):
        df.select([pl.struct(["B"])])


def test_nullable_object_13538() -> None:
    df = pl.DataFrame(
        data=[
            ({"a": 1},),
            ({"b": 3},),
            (None,),
        ],
        schema=[
            ("blob", pl.Object),
        ],
        orient="row",
    )

    df = df.select(
        is_null=pl.col("blob").is_null(), is_not_null=pl.col("blob").is_not_null()
    )
    assert df.to_dict(as_series=False) == {
        "is_null": [False, False, True],
        "is_not_null": [True, True, False],
    }

    df = pl.DataFrame({"col": pl.Series([0, 1, 2, None], dtype=pl.Object)})
    df = df.select(
        is_null=pl.col("col").is_null(), is_not_null=pl.col("col").is_not_null()
    )
    assert df.to_dict(as_series=False) == {
        "is_null": [False, False, False, True],
        "is_not_null": [True, True, True, False],
    }


def test_empty_sort() -> None:
    df = pl.DataFrame(
        data=[
            ({"name": "bar", "sort_key": 2},),
            ({"name": "foo", "sort_key": 1},),
        ],
        schema=[
            ("blob", pl.Object),
        ],
        orient="row",
    )
    df_filtered = df.filter(
        pl.col("blob").map_elements(
            lambda blob: blob["name"] == "baz", return_dtype=pl.Boolean
        )
    )
    df_filtered.sort(
        pl.col("blob").map_elements(
            lambda blob: blob["sort_key"], return_dtype=pl.Int64
        )
    )


def test_object_to_dicts() -> None:
    df = pl.DataFrame({"d": [{"a": 1, "b": 2, "c": 3}]}, schema={"d": pl.Object})
    assert df.to_dicts() == [{"d": {"a": 1, "b": 2, "c": 3}}]


def test_object_concat() -> None:
    df1 = pl.DataFrame(
        {"a": [1, 2, 3]},
        schema={"a": pl.Object},
    )

    df2 = pl.DataFrame(
        {"a": [1, 4, 3]},
        schema={"a": pl.Object},
    )

    catted = pl.concat([df1, df2])
    assert catted.shape == (6, 1)
    assert catted.dtypes == [pl.Object]
    assert catted.to_dict(as_series=False) == {"a": [1, 2, 3, 1, 4, 3]}


def test_object_row_construction() -> None:
    data = [
        [uuid4()],
        [uuid4()],
        [uuid4()],
    ]
    df = pl.DataFrame(
        data,
        orient="row",
    )
    assert df.dtypes == [pl.Object]
    assert df["column_0"].to_list() == [value[0] for value in data]


def test_object_apply_to_struct() -> None:
    s = pl.Series([0, 1, 2], dtype=pl.Object)
    out = s.map_elements(lambda x: {"a": str(x), "b": x})
    assert out.dtype == pl.Struct([pl.Field("a", pl.String), pl.Field("b", pl.Int64)])


def test_null_obj_str_13512() -> None:
    # https://github.com/pola-rs/polars/issues/13512

    df1 = pl.DataFrame(
        {
            "key": [1],
        }
    )
    df2 = pl.DataFrame({"key": [2], "a": pl.Series([1], dtype=pl.Object)})

    out = df1.join(df2, on="key", how="left")
    s = str(out)
    assert s == (
        "shape: (1, 2)\n"
        "┌─────┬────────┐\n"
        "│ key ┆ a      │\n"
        "│ --- ┆ ---    │\n"
        "│ i64 ┆ object │\n"
        "╞═════╪════════╡\n"
        "│ 1   ┆ null   │\n"
        "└─────┴────────┘"
    )


def test_format_object_series_14267() -> None:
    s = pl.Series([Path(), Path("abc")])
    expected = "shape: (2,)\n" "Series: '' [o][object]\n" "[\n" "\t.\n" "\tabc\n" "]"
    assert str(s) == expected


def test_object_raise_writers() -> None:
    df = pl.DataFrame({"a": object()})

    buf = io.BytesIO()

    with pytest.raises(ComputeError):
        df.write_parquet(buf)
    with pytest.raises(ComputeError):
        df.write_ipc(buf)
    with pytest.raises(ComputeError):
        df.write_json(buf)
    with pytest.raises(ComputeError):
        df.write_csv(buf)
    with pytest.raises(ComputeError):
        df.write_avro(buf)


def test_raise_list_object() -> None:
    # We don't want to support this. Unsafe enough as it is already.
    with pytest.raises(ValueError):
        pl.Series([[object()]], dtype=pl.List(pl.Object()))
