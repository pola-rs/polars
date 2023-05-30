import polars as pl


def test_binary_filter() -> None:
    df = pl.DataFrame(
        {
            "name": ["a", "b", "c", "d"],
            "content": [b"aa", b"aaabbb", b"aa", b"\xc6i\xea"],
        }
    )
    assert df.filter(pl.col("content") == b"\xc6i\xea").to_dict(False) == {
        "name": ["d"],
        "content": [b"\xc6i\xea"],
    }
