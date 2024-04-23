"""
Script to generate data for benchmarking group-by operations.

Data generation logic was adapted from the H2O.ai benchmark.
The original R script is located here:
https://github.com/h2oai/db-benchmark/blob/master/_data/groupby-datagen.R

Examples
--------
10 million rows, 100 groups, no nulls, random order:
$ python datagen_groupby.py 1e7 1e2 --null-percentage 0

100 million rows, 10 groups, 5% nulls, sorted:
$ python datagen_groupby.py 1e8 1e1 --null-percentage 5 --sorted
"""

import argparse
import logging

import numpy as np
from numpy.random import default_rng

import polars as pl

logging.basicConfig(level=logging.INFO)

SEED = 0
rng = default_rng(seed=SEED)

__all__ = ["generate_group_by_data"]


def generate_group_by_data(
    n_rows: int, n_groups: int, null_ratio: float = 0.0, *, sort: bool = False
) -> pl.DataFrame:
    """Generate data for benchmarking group-by operations."""
    logging.info("Generating data...")
    df = _generate_data(n_rows, n_groups)

    if null_ratio > 0.0:
        logging.info("Setting nulls...")
        df = _set_nulls(df, null_ratio)

    if sort:
        logging.info("Sorting data...")
        df = df.sort(c for c in df.columns if c.startswith("id"))

    logging.info("Done generating data.")
    return df


def _generate_data(n_rows: int, n_groups: int) -> pl.DataFrame:
    N = n_rows
    K = n_groups

    group_str_small = [f"id{str(i).zfill(3)}" for i in range(1, K + 1)]
    group_str_large = [f"id{str(i).zfill(10)}" for i in range(1, int(N / K) + 1)]
    group_int_small = range(1, K + 1)
    group_int_large = range(1, int(N / K) + 1)

    var_int_small = range(1, 6)
    var_int_large = range(1, 16)
    var_float = rng.uniform(0, 100, N)

    return pl.DataFrame(
        {
            "id1": rng.choice(group_str_small, N),
            "id2": rng.choice(group_str_small, N),
            "id3": rng.choice(group_str_large, N),
            "id4": rng.choice(group_int_small, N),
            "id5": rng.choice(group_int_small, N),
            "id6": rng.choice(group_int_large, N),
            "v1": rng.choice(var_int_small, N),
            "v2": rng.choice(var_int_large, N),
            "v3": np.round(var_float, 6),
        },
        schema={
            "id1": pl.String,
            "id2": pl.String,
            "id3": pl.String,
            "id4": pl.Int32,
            "id5": pl.Int32,
            "id6": pl.Int32,
            "v1": pl.Int32,
            "v2": pl.Int32,
            "v3": pl.Float64,
        },
    )


def _set_nulls(df: pl.DataFrame, null_ratio: float) -> pl.DataFrame:
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
    n_rows = int(args.rows)
    n_groups = int(args.groups)
    null_ratio = args.null_percentage / 100
    sort = args.sort

    logging.info(
        f"Generating data: {n_rows} rows, {n_groups} groups, {null_ratio} null ratio, sorted: {args.sort}"
    )

    df = generate_group_by_data(n_rows, n_groups, null_ratio=null_ratio, sort=sort)
    write_data(df, args)


def write_data(df: pl.DataFrame, args: argparse.Namespace) -> None:
    def format_int(i: int) -> str:
        base, exp = f"{i:e}".split("e")
        return f"{float(base):g}e{int(exp)}"

    if args.output is not None:
        filename = args.output
    else:
        filename = f"G1_{format_int(args.rows)}_{format_int(args.groups)}_{args.null_percentage}_{int(args.sort)}.csv"

    logging.info(f"Writing data to {filename}")
    df.write_csv(filename)
    logging.info("Done writing data.")


if __name__ == "__main__":
    main()
