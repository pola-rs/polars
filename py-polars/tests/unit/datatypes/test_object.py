from uuid import uuid4

import numpy as np

import polars as pl


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

    out = df.select([pl.struct(["B"]).alias("foo")]).to_dict(as_series=False)
    arr = out["foo"][0]["B"]
    assert isinstance(arr, np.ndarray)
    assert (arr == np_a).sum() == 3
    arr = out["foo"][1]["B"]
    assert (arr == np_b).sum() == 3


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
    assert out.dtype == pl.Struct([pl.Field("a", pl.Utf8), pl.Field("b", pl.Int64)])
