import polars as pl


def test_contains() -> None:
    df = pl.DataFrame(
        data=[
            (1, b"some * * text"),
            (2, b"(with) special\n * chars"),
            (3, b"**etc...?$"),
        ],
        columns=["idx", "bin"],
    )
    for pattern, expected in (
        (b"e * ", [True, False, False]),
        (b"text", [True, False, False]),
        (b"special", [False, True, False]),
        (b"", [True, True, True]),
        (b"qwe", [False, False, False]),
    ):
        # series
        assert expected == df["bin"].bin.contains(pattern).to_list()
        # frame select
        assert (
            expected == df.select(pl.col("bin").bin.contains(pattern))["bin"].to_list()
        )
        # frame filter
        assert sum(expected) == len(df.filter(pl.col("bin").bin.contains(pattern)))


def test_starts_ends_with() -> None:
    assert pl.DataFrame({"a": [b"hamburger", b"nuts", b"lollypop"]}).select(
        [
            pl.col("a").bin.ends_with(b"pop").alias("pop"),
            pl.col("a").bin.starts_with(b"ham").alias("ham"),
        ]
    ).to_dict(False) == {"pop": [False, False, True], "ham": [True, False, False]}


def test_base64_encode() -> None:
    df = pl.DataFrame({"data": [b"asd", b"qwe"]})

    assert [b"YXNk", b"cXdl"] == df["data"].bin.encode("base64").to_list()


def test_base64_decode() -> None:
    df = pl.DataFrame({"data": [b"YXNk", b"cXdl"]})

    assert [b"asd", b"qwe"] == df["data"].bin.decode("base64").to_list()


def test_hex_encode() -> None:
    df = pl.DataFrame({"data": [b"asd", b"qwe"]})

    assert [b"617364", b"717765"] == df["data"].bin.encode("hex").to_list()


def test_hex_decode() -> None:
    df = pl.DataFrame({"data": [b"617364", b"717765"]})

    assert [b"asd", b"qwe"] == df["data"].bin.decode("hex").to_list()
