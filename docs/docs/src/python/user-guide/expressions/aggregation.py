# --8<-- [start:setup]
import polars as pl
from datetime import date

# --8<-- [end:setup]

# --8<-- [start:dataframe]
url = "https://theunitedstates.io/congress-legislators/legislators-historical.csv"

dtypes = {
    "first_name": pl.Categorical,
    "gender": pl.Categorical,
    "type": pl.Categorical,
    "state": pl.Categorical,
    "party": pl.Categorical,
}

dataset = pl.read_csv(url, dtypes=dtypes).with_columns(
    pl.col("birthday").str.strptime(pl.Date, strict=False)
)
# --8<-- [end:dataframe]

# --8<-- [start:basic]
q = (
    dataset.lazy()
    .groupby("first_name")
    .agg(
        pl.count(),
        pl.col("gender"),
        pl.first("last_name"),
    )
    .sort("count", descending=True)
    .limit(5)
)

df = q.collect()
print(df)
# --8<-- [end:basic]

# --8<-- [start:conditional]
q = (
    dataset.lazy()
    .groupby("state")
    .agg(
        (pl.col("party") == "Anti-Administration").sum().alias("anti"),
        (pl.col("party") == "Pro-Administration").sum().alias("pro"),
    )
    .sort("pro", descending=True)
    .limit(5)
)

df = q.collect()
print(df)
# --8<-- [end:conditional]

# --8<-- [start:nested]
q = (
    dataset.lazy()
    .groupby("state", "party")
    .agg(pl.count("party").alias("count"))
    .filter(
        (pl.col("party") == "Anti-Administration")
        | (pl.col("party") == "Pro-Administration")
    )
    .sort("count", descending=True)
    .limit(5)
)

df = q.collect()
print(df)
# --8<-- [end:nested]


# --8<-- [start:filter]
def compute_age() -> pl.Expr:
    return date(2021, 1, 1).year - pl.col("birthday").dt.year()


def avg_birthday(gender: str) -> pl.Expr:
    return (
        compute_age()
        .filter(pl.col("gender") == gender)
        .mean()
        .alias(f"avg {gender} birthday")
    )


q = (
    dataset.lazy()
    .groupby("state")
    .agg(
        avg_birthday("M"),
        avg_birthday("F"),
        (pl.col("gender") == "M").sum().alias("# male"),
        (pl.col("gender") == "F").sum().alias("# female"),
    )
    .limit(5)
)

df = q.collect()
print(df)
# --8<-- [end:filter]


# --8<-- [start:sort]
def get_person() -> pl.Expr:
    return pl.col("first_name") + pl.lit(" ") + pl.col("last_name")


q = (
    dataset.lazy()
    .sort("birthday", descending=True)
    .groupby("state")
    .agg(
        get_person().first().alias("youngest"),
        get_person().last().alias("oldest"),
    )
    .limit(5)
)

df = q.collect()
print(df)
# --8<-- [end:sort]


# --8<-- [start:sort2]
def get_person() -> pl.Expr:
    return pl.col("first_name") + pl.lit(" ") + pl.col("last_name")


q = (
    dataset.lazy()
    .sort("birthday", descending=True)
    .groupby("state")
    .agg(
        get_person().first().alias("youngest"),
        get_person().last().alias("oldest"),
        get_person().sort().first().alias("alphabetical_first"),
    )
    .limit(5)
)

df = q.collect()
print(df)
# --8<-- [end:sort2]


# --8<-- [start:sort3]
def get_person() -> pl.Expr:
    return pl.col("first_name") + pl.lit(" ") + pl.col("last_name")


q = (
    dataset.lazy()
    .sort("birthday", descending=True)
    .groupby("state")
    .agg(
        get_person().first().alias("youngest"),
        get_person().last().alias("oldest"),
        get_person().sort().first().alias("alphabetical_first"),
        pl.col("gender").sort_by("first_name").first().alias("gender"),
    )
    .sort("state")
    .limit(5)
)

df = q.collect()
print(df)
# --8<-- [end:sort3]
