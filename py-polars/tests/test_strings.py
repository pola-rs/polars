import polars as pl


def test_extract_binary() -> None:
    df = pl.DataFrame({"foo": ["aron", "butler", "charly", "david"]})
    out = df.filter(pl.col("foo").str.extract("^(a)", 1) == "a").to_series()
    assert out[0] == "aron"


def test_auto_explode() -> None:
    df = pl.DataFrame(
        [pl.Series("val", ["A", "B", "C", "D"]), pl.Series("id", [1, 1, 2, 2])]
    )
    pl.col("val").str.concat(delimiter=",")
    grouped = (
        df.groupby("id")
        .agg(pl.col("val").str.concat(delimiter=",").alias("grouped"))
        .get_column("grouped")
    )
    assert grouped.dtype == pl.Utf8


def test_null_comparisons() -> None:
    s = pl.Series("s", [None, "str", "a"])
    assert (s.shift() == s).null_count() == 0
    assert (s.shift() != s).null_count() == 0


def test_extract_all_count() -> None:
    df = pl.DataFrame({"foo": ["123 bla 45 asd", "xyz 678 910t"]})
    assert (
        df.select(
            [
                pl.col("foo").str.extract_all(r"a").alias("extract"),
                pl.col("foo").str.count_match(r"a").alias("count"),
            ]
        ).to_dict(False)
        == {"extract": [["a", "a"], None], "count": [2, 0]}
    )

    assert df["foo"].str.extract_all(r"a").dtype == pl.List
    assert df["foo"].str.count_match(r"a").dtype == pl.UInt32


def test_zfill() -> None:
    df = pl.DataFrame(
        {
            "num": [-10, -1, 0, 1, 10, 100, 1000, 10000, 100000, 1000000, None],
        }
    )

    out = [
        "-0010",
        "-0001",
        "00000",
        "00001",
        "00010",
        "00100",
        "01000",
        "10000",
        "100000",
        "1000000",
        None,
    ]
    assert (
        df.with_column(pl.col("num").cast(str).str.zfill(5)).to_series().to_list()
        == out
    )
    assert df["num"].cast(str).str.zfill(5) == out


def test_ljust_and_rjust() -> None:
    df = pl.DataFrame({"a": ["foo", "longer_foo", "longest_fooooooo", "hi"]})
    assert df.select(
        [
            pl.col("a").str.rjust(10).alias("rjust"),
            pl.col("a").str.rjust(10).str.lengths().alias("rjust_len"),
            pl.col("a").str.ljust(10).alias("ljust"),
            pl.col("a").str.ljust(10).str.lengths().alias("ljust_len"),
        ]
    ).to_dict(False) == {
        "rjust": ["       foo", "longer_foo", "longest_fooooooo", "        hi"],
        "rjust_len": [10, 10, 16, 10],
        "ljust": ["foo       ", "longer_foo", "longest_fooooooo", "hi        "],
        "ljust_len": [10, 10, 16, 10],
    }
