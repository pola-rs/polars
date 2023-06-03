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
        ).with_columns(
            pl.col(["own_time", "cumulative_time"]).str.strip().cast(pl.Int32)
        )
        for _ in range(best_of)
    ]
    return (
        sorted(
            import_timings,
            key=lambda tm: int(tm["cumulative_time"].max()),  # type: ignore[arg-type]
        )[0]
        .reverse()
        .select("import", "own_time", "cumulative_time")
    )


@pytest.mark.skipif(sys.platform == "win32", reason="Unreliable on Windows")
def test_polars_import() -> None:
    # note: take the fastest of several runs to reduce noise.
    df_import = _import_timings_as_frame(best_of=3)

    with pl.Config() as cfg:
        # get a complete view of what's going on in case of failure
        cfg.set_tbl_rows(250)
        cfg.set_fmt_str_lengths(100)
        cfg.set_tbl_hide_dataframe_shape()

        # ensure that we have not broken lazy-loading (numpy, pandas, pyarrow, etc).
        lazy_modules = [
            dep for dep in pl.dependencies.__all__ if not dep.startswith("_")
        ]
        for mod in lazy_modules:
            not_imported = not df_import["import"].str.starts_with(mod).any()
            if_err = f"lazy-loading regression: found {mod!r} at import time"
            assert not_imported, f"{if_err}\n{df_import}"

        # ensure that we do not have an import speed regression.
        polars_import = df_import.filter(pl.col("import").str.strip() == "polars")
        polars_import_time = polars_import["cumulative_time"].item()
        assert isinstance(polars_import_time, int)

        if_err = f"Possible import speed regression; took {polars_import_time//1_000}ms"
        assert polars_import_time < 200_000, f"{if_err}\n{df_import}"
