import polars as pl
from polars.testing import assert_frame_equal


def test_binary_filter() -> None:
    df = pl.DataFrame(
        {
            "name": ["a", "b", "c", "d"],
            "content": [b"aa", b"aaabbb", b"aa", b"\xc6i\xea"],
        }
    )
    assert df.filter(pl.col("content") == b"\xc6i\xea").to_dict(as_series=False) == {
        "name": ["d"],
        "content": [b"\xc6i\xea"],
    }


def test_binary_to_list() -> None:
    data = {"binary": [b"\xFD\x00\xFE\x00\xFF\x00", b"\x10\x00\x20\x00\x30\x00"]}
    schema = {"binary": pl.Binary}

    df = pl.DataFrame(data, schema).with_columns(
        pl.col("binary").cast(pl.List(pl.UInt8))
    )

    expected = pl.DataFrame(
        {"binary": [[253, 0, 254, 0, 255, 0], [16, 0, 32, 0, 48, 0]]},
        schema={"binary": pl.List(pl.UInt8)},
    )
    assert_frame_equal(df, expected)
