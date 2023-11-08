import pytest

import polars as pl
from polars.type_aliases import TransferEncoding


def test_binary_conversions() -> None:
    df = pl.DataFrame({"blob": [b"abc", None, b"cde"]}).with_columns(
        pl.col("blob").cast(pl.Utf8).alias("decoded_blob")
    )

    assert df.to_dict(as_series=False) == {
        "blob": [b"abc", None, b"cde"],
        "decoded_blob": ["abc", None, "cde"],
    }
    assert df[0, 0] == b"abc"
    assert df[1, 0] is None
    assert df.dtypes == [pl.Binary, pl.Utf8]


def test_contains() -> None:
    df = pl.DataFrame(
        data=[
            (1, b"some * * text"),
            (2, b"(with) special\n * chars"),
            (3, b"**etc...?$"),
            (4, None),
        ],
        schema=["idx", "bin"],
    )
    for pattern, expected in (
        (b"e * ", [True, False, False, None]),
        (b"text", [True, False, False, None]),
        (b"special", [False, True, False, None]),
        (b"", [True, True, True, None]),
        (b"qwe", [False, False, False, None]),
    ):
        # series
        assert expected == df["bin"].bin.contains(pattern).to_list()
        # frame select
        assert (
            expected == df.select(pl.col("bin").bin.contains(pattern))["bin"].to_list()
        )
        # frame filter
        assert sum([e for e in expected if e is True]) == len(
            df.filter(pl.col("bin").bin.contains(pattern))
        )


def test_contains_with_expr() -> None:
    df = pl.DataFrame(
        {
            "bin": [b"some * * text", b"(with) special\n * chars", b"**etc...?$", None],
            "lit1": [b"e * ", b"", b"qwe", b"None"],
            "lit2": [None, b"special\n", b"?!", None],
        }
    )

    assert df.select(
        [
            pl.col("bin").bin.contains(pl.col("lit1")).alias("contains_1"),
            pl.col("bin").bin.contains(pl.col("lit2")).alias("contains_2"),
            pl.col("bin").bin.contains(pl.lit(None)).alias("contains_3"),
        ]
    ).to_dict(as_series=False) == {
        "contains_1": [True, True, False, None],
        "contains_2": [None, True, False, None],
        "contains_3": [None, None, None, None],
    }


def test_starts_ends_with() -> None:
    assert pl.DataFrame(
        {
            "a": [b"hamburger", b"nuts", b"lollypop", None],
            "end": [b"ger", b"tg", None, b"anything"],
            "start": [b"ha", b"nga", None, b"anything"],
        }
    ).select(
        [
            pl.col("a").bin.ends_with(b"pop").alias("end_lit"),
            pl.col("a").bin.ends_with(pl.lit(None)).alias("end_none"),
            pl.col("a").bin.ends_with(pl.col("end")).alias("end_expr"),
            pl.col("a").bin.starts_with(b"ham").alias("start_lit"),
            pl.col("a").bin.ends_with(pl.lit(None)).alias("start_none"),
            pl.col("a").bin.starts_with(pl.col("start")).alias("start_expr"),
        ]
    ).to_dict(as_series=False) == {
        "end_lit": [False, False, True, None],
        "end_none": [None, None, None, None],
        "end_expr": [True, False, None, None],
        "start_lit": [True, False, False, None],
        "start_none": [None, None, None, None],
        "start_expr": [True, False, None, None],
    }


def test_base64_encode() -> None:
    df = pl.DataFrame({"data": [b"asd", b"qwe"]})

    assert ["YXNk", "cXdl"] == df["data"].bin.encode("base64").to_list()


def test_base64_decode() -> None:
    df = pl.DataFrame({"data": [b"YXNk", b"cXdl"]})

    assert [b"asd", b"qwe"] == df["data"].bin.decode("base64").to_list()


def test_hex_encode() -> None:
    df = pl.DataFrame({"data": [b"asd", b"qwe"]})

    assert ["617364", "717765"] == df["data"].bin.encode("hex").to_list()


def test_hex_decode() -> None:
    df = pl.DataFrame({"data": [b"617364", b"717765"]})

    assert [b"asd", b"qwe"] == df["data"].bin.decode("hex").to_list()


@pytest.mark.parametrize(
    "encoding",
    [
        "hex",
        "base64",
    ],
)
def test_compare_encode_between_lazy_and_eager_6814(encoding: TransferEncoding) -> None:
    df = pl.DataFrame({"x": [b"aa", b"bb", b"cc"]})
    expr = pl.col("x").bin.encode(encoding)
    result_eager = df.select(expr)
    dtype = result_eager["x"].dtype
    result_lazy = df.lazy().select(expr).select(pl.col(dtype)).collect()
    assert result_eager.frame_equal(result_lazy)


@pytest.mark.parametrize(
    "encoding",
    [
        "hex",
        "base64",
    ],
)
def test_compare_decode_between_lazy_and_eager_6814(encoding: TransferEncoding) -> None:
    df = pl.DataFrame({"x": [b"d3d3", b"abcd", b"1234"]})
    expr = pl.col("x").bin.decode(encoding)
    result_eager = df.select(expr)
    dtype = result_eager["x"].dtype
    result_lazy = df.lazy().select(expr).select(pl.col(dtype)).collect()
    assert result_eager.frame_equal(result_lazy)
