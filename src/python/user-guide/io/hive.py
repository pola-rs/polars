# --8<-- [start:init_paths]
import polars as pl
from pathlib import Path

dfs = [
    pl.DataFrame({"x": [1, 2]}),
    pl.DataFrame({"x": [3, 4, 5]}),
    pl.DataFrame({"x": [6, 7]}),
    pl.DataFrame({"x": [8, 9, 10, 11]}),
]

parts = [
    "year=2023/month=11",
    "year=2023/month=12",
    "year=2024/month=01",
    "year=2024/month=02",
]

for df, part in zip(dfs, parts):
    path = Path("docs/assets/data/hive/") / part / "data.parquet"
    Path(path).parent.mkdir(exist_ok=True, parents=True)
    df.write_parquet(path)

    path = Path("docs/assets/data/hive_mixed/") / part / "data.parquet"
    Path(path).parent.mkdir(exist_ok=True, parents=True)
    df.write_parquet(path)

# Make sure the file is not empty because path expansion ignores empty files.
Path("docs/assets/data/hive_mixed/description.txt").write_text("A")


def print_paths(path: str) -> None:
    def dir_recurse(path: Path):
        if path.is_dir():
            for p in path.iterdir():
                yield from dir_recurse(p)
        else:
            yield path

    df = (
        pl.Series(
            "File path",
            (str(x) for x in dir_recurse(Path(path))),
            dtype=pl.String,
        )
        .sort()
        .to_frame()
    )

    with pl.Config(
        tbl_hide_column_data_types=True,
        tbl_hide_dataframe_shape=True,
        fmt_str_lengths=999,
    ):
        print(df)


print_paths("docs/assets/data/hive/")
# --8<-- [end:init_paths]

# --8<-- [start:show_mixed_paths]
print_paths("docs/assets/data/hive_mixed/")
# --8<-- [end:show_mixed_paths]

# --8<-- [start:scan_dir]
import polars as pl

df = pl.scan_parquet("docs/assets/data/hive/").collect()

with pl.Config(tbl_rows=99):
    print(df)
# --8<-- [end:scan_dir]

# --8<-- [start:scan_dir_err]
from pathlib import Path

try:
    pl.scan_parquet("docs/assets/data/hive_mixed/").collect()
except Exception as e:
    print(e)

# --8<-- [end:scan_dir_err]

# --8<-- [start:scan_glob]
df = pl.scan_parquet(
    # Glob to match all files ending in `.parquet`
    "docs/assets/data/hive_mixed/**/*.parquet",
    hive_partitioning=True,
).collect()

with pl.Config(tbl_rows=99):
    print(df)

# --8<-- [end:scan_glob]

# --8<-- [start:scan_file_no_hive]
df = pl.scan_parquet(
    [
        "docs/assets/data/hive/year=2024/month=01/data.parquet",
        "docs/assets/data/hive/year=2024/month=02/data.parquet",
    ],
).collect()

print(df)

# --8<-- [end:scan_file_no_hive]

# --8<-- [start:scan_file_hive]
df = pl.scan_parquet(
    [
        "docs/assets/data/hive/year=2024/month=01/data.parquet",
        "docs/assets/data/hive/year=2024/month=02/data.parquet",
    ],
    hive_partitioning=True,
).collect()

print(df)

# --8<-- [end:scan_file_hive]

# --8<-- [start:write_parquet_partitioned_show_data]
df = pl.DataFrame({"a": [1, 1, 2, 2, 3], "b": [1, 1, 1, 2, 2], "c": 1})
print(df)
# --8<-- [end:write_parquet_partitioned_show_data]

# --8<-- [start:write_parquet_partitioned]
df.write_parquet("docs/assets/data/hive_write/", partition_by=["a", "b"])
# --8<-- [end:write_parquet_partitioned]

# --8<-- [start:write_parquet_partitioned_show_paths]
print_paths("docs/assets/data/hive_write/")
# --8<-- [end:write_parquet_partitioned_show_paths]
