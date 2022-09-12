import io

import polars as pl


def test_csv_statistics_offset() -> None:
    # this would fail if the statistics sample did not also sample
    # from the end of the file
    # the lines at the end have larger rows as the numbers increase
    csv = "\n".join(str(x) for x in range(5_000))
    assert pl.read_csv(io.StringIO(csv), n_rows=5000).height == 4999
