import polars as pl
from polars.testing import assert_frame_equal


def test_drop_explode_6641() -> None:
    df = pl.DataFrame(
        {
            "chromosome": ["chr1"] * 2,
            "identifier": [["chr1:10426:10429:ACC>A"], ["chr1:10426:10429:ACC>*"]],
            "alternate": [["A"], ["T"]],
            "quality": pl.Series([None, None], dtype=pl.Float32()),
        }
    ).lazy()

    assert (
        df.explode(["identifier", "alternate"])
        .with_columns(pl.struct(["identifier", "alternate"]).alias("test"))
        .drop(["identifier", "alternate"])
        .select(pl.concat_list([pl.col("test"), pl.col("test")]))
        .collect()
    ).to_dict(False) == {
        "test": [
            [
                {"identifier": "chr1:10426:10429:ACC>A", "alternate": "A"},
                {"identifier": "chr1:10426:10429:ACC>A", "alternate": "A"},
            ],
            [
                {"identifier": "chr1:10426:10429:ACC>*", "alternate": "T"},
                {"identifier": "chr1:10426:10429:ACC>*", "alternate": "T"},
            ],
        ]
    }


def test_drop_nulls() -> None:
    df = pl.DataFrame(
        {
            "foo": [1, 2, 3],
            "bar": [6, None, 8],
            "ham": ["a", "b", "c"],
        }
    )
    result = df.drop_nulls()
    expected = pl.DataFrame(
        {
            "foo": [1, 3],
            "bar": [6, 8],
            "ham": ["a", "c"],
        }
    )
    assert_frame_equal(result, expected)

    # below we only drop entries if they are null in the column 'foo'
    result = df.drop_nulls("foo")
    assert_frame_equal(result, df)


def test_drop() -> None:
    df = pl.DataFrame({"a": [2, 1, 3], "b": ["a", "b", "c"], "c": [1, 2, 3]})
    df = df.drop(columns="a")
    assert df.shape == (3, 2)
    df = pl.DataFrame({"a": [2, 1, 3], "b": ["a", "b", "c"], "c": [1, 2, 3]})
    s = df.drop_in_place("a")
    assert s.name == "a"


def test_drop_nulls_lazy() -> None:
    df = pl.DataFrame({"nrs": [None, 1, 2, 3, None, 4, 5, None]})
    assert df.select(pl.col("nrs").drop_nulls()).to_dict(as_series=False) == {
        "nrs": [1, 2, 3, 4, 5]
    }

    df = pl.DataFrame({"foo": [1, 2, 3], "bar": [6, None, 8], "ham": ["a", "b", "c"]})
    expected = pl.DataFrame({"foo": [1, 3], "bar": [6, 8], "ham": ["a", "c"]})
    result = df.lazy().drop_nulls().collect()
    assert_frame_equal(result, expected)


def test_drop_columns() -> None:
    out = pl.DataFrame({"a": [1], "b": [2], "c": [3]}).lazy().drop(["a", "b"])
    assert out.columns == ["c"]

    out = pl.DataFrame({"a": [1], "b": [2], "c": [3]}).lazy().drop("a")
    assert out.columns == ["b", "c"]


def test_drop_nan_ignore_null_3525() -> None:
    df = pl.DataFrame({"a": [1.0, float("NaN"), 2.0, None, 3.0, 4.0]})
    assert df.select(pl.col("a").drop_nans()).to_series().to_list() == [
        1.0,
        2.0,
        None,
        3.0,
        4.0,
    ]
