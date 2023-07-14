import polars as pl
from polars import NUMERIC_DTYPES


def test_regex_exclude() -> None:
    df = pl.DataFrame({f"col_{i}": [i] for i in range(5)})

    assert df.select(pl.col("^col_.*$").exclude("col_0")).columns == [
        "col_1",
        "col_2",
        "col_3",
        "col_4",
    ]


def test_regex_in_filter() -> None:
    df = pl.DataFrame(
        {
            "nrs": [1, 2, 3, None, 5],
            "names": ["foo", "ham", "spam", "egg", None],
            "flt": [1.0, None, 3.0, 1.0, None],
        }
    )

    res = df.filter(
        pl.fold(
            acc=False, function=lambda acc, s: acc | s, exprs=(pl.col("^nrs|flt*$") < 3)
        )
    ).row(0)
    expected = (1, "foo", 1.0)
    assert res == expected


def test_regex_selection() -> None:
    ldf = pl.LazyFrame(
        {
            "foo": [1],
            "fooey": [1],
            "foobar": [1],
            "bar": [1],
        }
    )
    assert ldf.select([pl.col("^foo.*$")]).columns == ["foo", "fooey", "foobar"]


def test_exclude_selection() -> None:
    ldf = pl.LazyFrame({"a": [1], "b": [1], "c": [True]})

    assert ldf.select([pl.exclude("a")]).columns == ["b", "c"]
    assert ldf.select(pl.all().exclude(pl.Boolean)).columns == ["a", "b"]
    assert ldf.select(pl.all().exclude([pl.Boolean])).columns == ["a", "b"]
    assert ldf.select(pl.all().exclude(NUMERIC_DTYPES)).columns == ["c"]
