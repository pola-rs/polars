import io
import os

import polars as pl


def test_csv_statistics_offset() -> None:
    # this would fail if the statistics sample did not also sample
    # from the end of the file
    # the lines at the end have larger rows as the numbers increase
    csv = "\n".join(str(x) for x in range(5_000))
    assert pl.read_csv(io.StringIO(csv), n_rows=5000).height == 4999


def test_csv_scan_categorical() -> None:
    N = 5_000
    if os.name != "nt":
        pl.DataFrame({"x": ["A"] * N}).write_csv("/tmp/test_csv_scan_categorical.csv")
        df = pl.scan_csv(
            "/tmp/test_csv_scan_categorical.csv", dtypes={"x": pl.Categorical}
        ).collect()
        assert df["x"].dtype == pl.Categorical
