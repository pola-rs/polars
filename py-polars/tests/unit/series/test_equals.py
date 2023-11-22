from datetime import datetime

import polars as pl


def test_equals() -> None:
    s1 = pl.Series("a", [1.0, 2.0, None], pl.Float64)
    s2 = pl.Series("a", [1, 2, None], pl.Int64)

    assert s1.equals(s2) is True
    assert s1.equals(s2, strict=True) is False
    assert s1.equals(s2, null_equal=False) is False

    df = pl.DataFrame(
        {"dtm": [datetime(2222, 2, 22, 22, 22, 22)]},
        schema_overrides={"dtm": pl.Datetime(time_zone="UTC")},
    ).with_columns(
        s3=pl.col("dtm").dt.convert_time_zone("Europe/London"),
        s4=pl.col("dtm").dt.convert_time_zone("Asia/Tokyo"),
    )
    s3 = df["s3"].rename("b")
    s4 = df["s4"].rename("b")

    assert s3.equals(s4) is False
    assert s3.equals(s4, strict=True) is False
    assert s3.equals(s4, null_equal=False) is False
    assert s3.dt.convert_time_zone("Asia/Tokyo").equals(s4) is True
