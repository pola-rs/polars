import polars as pl


def test_read_web_file() -> None:
    url = "https://raw.githubusercontent.com/pola-rs/polars/master/examples/datasets/foods1.csv"  # noqa: E501
    df = pl.read_csv(url)
    assert df.shape == (27, 4)
