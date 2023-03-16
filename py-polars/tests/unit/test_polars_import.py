import subprocess
import sys

import pytest

import polars as pl


def _import_timings() -> bytes:
    # assemble suitable command to get polars module import timing;
    # run in a separate process to ensure clean timing results.
    cmd = f'{sys.executable} -X importtime -c "import polars"'
    return (
        subprocess.run(cmd, shell=True, capture_output=True)
        .stderr.replace(b"import time:", b"")
        .replace(b" ", b"")
        .strip()
    )


def _import_timings_as_frame(best_of: int) -> pl.DataFrame:
    # create master frame as minimum of 'best_of' timings.
    import_timings = [
        pl.read_csv(
            source=_import_timings(),
            separator="|",
            has_header=True,
            new_columns=["own_time", "cumulative_time", "import"],
        )
        for _ in range(best_of)
    ]
    return (
        sorted(
            import_timings,
            key=lambda tm: int(tm["cumulative_time"].max()),  # type: ignore[arg-type]
        )[0]
        .select("import", "own_time", "cumulative_time")
        .sort(by=["cumulative_time"], descending=True)
    )


@pytest.mark.skipif(sys.platform == "win32", reason="Unreliable on Windows")
def test_polars_import() -> None:
    # note: take the fastest of three runs to reduce noise.
    df_import = _import_timings_as_frame(best_of=3)

    # ensure that we have not broken lazy-loading (numpy, pandas, pyarrow, etc).
    lazy_modules = [dep for dep in pl.dependencies.__all__ if not dep.startswith("_")]
    for mod in lazy_modules:
        assert (
            not df_import["import"].str.starts_with(mod).any()
        ), f"lazy-loading regression: found {mod!r} at import time"

    # ensure that we do not have an import speed regression.
    with pl.Config() as cfg:
        cfg.set_tbl_rows(25)
        cfg.set_tbl_hide_dataframe_shape()

        total_import_time = df_import["cumulative_time"].max()
        assert isinstance(total_import_time, int)

        if_err = f"Possible import speed regression; took {total_import_time//1_000}ms"
        assert total_import_time < 200_000, f"{if_err}\n{df_import}"
