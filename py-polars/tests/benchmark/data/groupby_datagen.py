"""
Script to generate data for benchmarking group-by operations.

Data generation logic was adapted from the H2O.ai benchmark.
The original R script is located here:
https://github.com/h2oai/db-benchmark/blob/master/_data/groupby-datagen.R

Examples
--------
10 million rows, 100 groups, no nulls, random order:
$ python groupby_datagen.py 1e7 1e2 --null-percentage 0

100 million rows, 10 groups, 5% nulls, sorted:
$ python groupby_datagen.py 1e8 1e1 --null-percentage 5 --sorted
"""

import argparse

import numpy as np
from numpy.random import default_rng

import polars as pl

SEED = 0
rng = default_rng(seed=SEED)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate data for benchmarking group-by operations"
    )

    parser.add_argument("rows", type=float, help="Number of rows")
    parser.add_argument("groups", type=float, help="Number of groups")
    parser.add_argument(
        "-n",
        "--null-percentage",
        type=int,
        default=0,
        choices=range(1, 101),
        metavar="[0-100]",
        help="Percentage of null values",
    )
    parser.add_argument(
        "-s",
        "--sort",
        action="store_true",
        help="Sort the data by group",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output filename",
    )

    args = parser.parse_args()

    # Convert arguments to appropriate types
    N = int(args.rows)
    K = int(args.groups)
    null_ratio = args.null_percentage / 100

    print(
        f"Producing data of {N} rows, {K} K groups factors, {args.null_percentage}% nulls, sorted: {args.sort}"
    )

    df = pl.DataFrame(
        {
            "id1": rng.choice([f"id{str(i).zfill(3)}" for i in range(1, K + 1)], N),
            "id2": rng.choice([f"id{str(i).zfill(3)}" for i in range(1, K + 1)], N),
            "id3": rng.choice(
                [f"id{str(i).zfill(10)}" for i in range(1, int(N / K) + 1)], N
            ),
            "id4": rng.choice(range(1, K + 1), N),
            "id5": rng.choice(range(1, K + 1), N),
            "id6": rng.choice(range(1, int(N / K) + 1), N),
            "v1": rng.choice(range(1, 6), N),
            "v2": rng.choice(range(1, 16), N),
            "v3": np.round(rng.uniform(0, 100, N), 6),
        }
    )

    if null_ratio > 0.0:
        print("Setting nulls")
        df = set_nulls(df, null_ratio)

    if args.sort:
        print("Sorting data")
        df = df.sort(c for c in df.columns if c.startswith("id"))

    write_data(df, args)


def set_nulls(df: pl.DataFrame, null_ratio: float) -> pl.DataFrame:
    """Set null values according to the given ratio."""

    def set_nulls_var(s: pl.Series, ratio: float) -> pl.Series:
        """Set Series values to null according to the given ratio."""
        len = s.len()
        n_null = int(ratio * len)
        if n_null == 0:
            return s

        indices = rng.choice(len, size=n_null, replace=False)
        return s.scatter(indices, None)

    def set_nulls_group(s: pl.Series, ratio: float) -> pl.Series:
        """Set Series unique values to null according to the given ratio."""
        uniques = s.unique()
        n_null = int(ratio * uniques.len())
        if n_null == 0:
            return s

        to_replace = rng.choice(uniques, size=n_null, replace=False)
        return (
            s.to_frame()
            .select(
                pl.when(pl.col(s.name).is_in(to_replace))
                .then(None)
                .otherwise(pl.col(s.name))
                .alias(s.name)
            )
            .to_series()
        )

    return df.with_columns(
        set_nulls_group(s, null_ratio)
        if s.name.startswith("id")
        else set_nulls_var(s, null_ratio)
        for s in df.get_columns()
    )


def write_data(df: pl.DataFrame, args: argparse.Namespace) -> None:
    def format_int(i: int) -> str:
        base, exp = f"{i:e}".split("e")
        base = f"{float(base):g}"
        exp = int(exp)
        return f"{base}e{exp}"

    if args.output is not None:
        filename = args.output
    else:
        filename = f"G1_{format_int(args.rows)}_{format_int(args.groups)}_{args.null_percentage}_{int(args.sort)}.csv"

    print(f"Writing data to {filename}")
    df.write_csv(filename)


if __name__ == "__main__":
    main()
