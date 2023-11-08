"""
# --8<-- [start:recommendation]
from multiprocessing import get_context


def my_fun(s):
    print(s)


with get_context("spawn").Pool() as pool:
    pool.map(my_fun, ["input1", "input2", ...])

# --8<-- [end:recommendation]

# --8<-- [start:example1]
import multiprocessing
import polars as pl


def test_sub_process(df: pl.DataFrame, job_id):
    df_filtered = df.filter(pl.col("a") > 0)
    print(f"Filtered (job_id: {job_id})", df_filtered, sep="\n")


def create_dataset():
    return pl.DataFrame({"a": [0, 2, 3, 4, 5], "b": [0, 4, 5, 56, 4]})


def setup():
    # some setup work
    df = create_dataset()
    df.write_parquet("/tmp/test.parquet")


def main():
    test_df = pl.read_parquet("/tmp/test.parquet")

    for i in range(0, 5):
        proc = multiprocessing.get_context("spawn").Process(
            target=test_sub_process, args=(test_df, i)
        )
        proc.start()
        proc.join()

        print(f"Executed sub process {i}")


if __name__ == "__main__":
    setup()
    main()

# --8<-- [end:example1]
"""
# --8<-- [start:example2]
import multiprocessing
import polars as pl


def test_sub_process(df: pl.DataFrame, job_id):
    df_filtered = df.filter(pl.col("a") > 0)
    print(f"Filtered (job_id: {job_id})", df_filtered, sep="\n")


def create_dataset():
    return pl.DataFrame({"a": [0, 2, 3, 4, 5], "b": [0, 4, 5, 56, 4]})


def main():
    test_df = create_dataset()

    for i in range(0, 5):
        proc = multiprocessing.get_context("fork").Process(
            target=test_sub_process, args=(test_df, i)
        )
        proc.start()
        proc.join()

        print(f"Executed sub process {i}")


if __name__ == "__main__":
    main()

# --8<-- [end:example2]
