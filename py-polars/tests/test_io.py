import io
from utils import get_complete_df
import polars as pl


def test_to_from_buffer():
    df = get_complete_df()
    df = df.drop("strings_nulls")

    for to_fn, from_fn in zip(
        [df.to_parquet, df.to_csv], [df.read_parquet, df.read_csv]
    ):
        f = io.BytesIO()
        to_fn(f)
        f.seek(0)

        df_1 = from_fn(f)
        assert df.frame_equal(df_1, null_equal=True)


def test_read_web_file():
    url = "https://raw.githubusercontent.com/ritchie46/polars/master/examples/aggregate_multiple_files_in_chunks/datasets/foods1.csv"
    df = pl.read_csv(url)
    assert df.shape == (27, 4)
