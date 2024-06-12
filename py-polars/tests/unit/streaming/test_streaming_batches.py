import polars as pl


def test_stream_data_in_batches():
    df = pl.DataFrame({"col_1": [0] * 5 + [1] * 5})

    dfs = list(df.lazy().collect_batches(
        streaming=True
    ))

    assert len(dfs) == 11
