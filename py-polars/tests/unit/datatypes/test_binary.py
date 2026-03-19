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
    data = {"binary": [b"\xfd\x00\xfe\x00\xff\x00", b"\x10\x00\x20\x00\x30\x00"]}
    schema = {"binary": pl.Binary}

    print(pl.DataFrame(data, schema))
    df = pl.DataFrame(data, schema).with_columns(
        pl.col("binary").cast(pl.List(pl.UInt8))
    )

    expected = pl.DataFrame(
        {"binary": [[253, 0, 254, 0, 255, 0], [16, 0, 32, 0, 48, 0]]},
        schema={"binary": pl.List(pl.UInt8)},
    )
    print(df)
    assert_frame_equal(df, expected)


def test_string_to_binary() -> None:
    s = pl.Series("data", ["", None, "\x01\x02"])

    assert s.cast(pl.Binary).to_list() == [b"", None, b"\x01\x02"]
    assert s.cast(pl.Binary).cast(pl.Utf8).to_list() == ["", None, "\x01\x02"]
