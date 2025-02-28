# --8<-- [start:pokemon]
import polars as pl

types = (
    "Grass Water Fire Normal Ground Electric Psychic Fighting Bug Steel "
    "Flying Dragon Dark Ghost Poison Rock Ice Fairy".split()
)
type_enum = pl.Enum(types)
# then let's load some csv data with information about pokemon
pokemon = pl.read_csv(
    "https://gist.githubusercontent.com/ritchie46/cac6b337ea52281aa23c049250a4ff03/raw/89a957ff3919d90e6ef2d34235e6bf22304f3366/pokemon.csv",
).cast({"Type 1": type_enum, "Type 2": type_enum})
print(pokemon.head())
# --8<-- [end:pokemon]

# --8<-- [start:rank]
result = pokemon.select(
    pl.col("Name", "Type 1"),
    pl.col("Speed").rank("dense", descending=True).over("Type 1").alias("Speed rank"),
)

print(result)
# --8<-- [end:rank]

# --8<-- [start:rank-multiple]
result = pokemon.select(
    pl.col("Name", "Type 1", "Type 2"),
    pl.col("Speed")
    .rank("dense", descending=True)
    .over("Type 1", "Type 2")
    .alias("Speed rank"),
)

print(result)
# --8<-- [end:rank-multiple]

# --8<-- [start:rank-explode]
result = (
    pokemon.group_by("Type 1")
    .agg(
        pl.col("Name"),
        pl.col("Speed").rank("dense", descending=True).alias("Speed rank"),
    )
    .select(pl.col("Name"), pl.col("Type 1"), pl.col("Speed rank"))
    .explode("Name", "Speed rank")
)

print(result)
# --8<-- [end:rank-explode]

# --8<-- [start:athletes]
athletes = pl.DataFrame(
    {
        "athlete": list("ABCDEF"),
        "country": ["PT", "NL", "NL", "PT", "PT", "NL"],
        "rank": [6, 1, 5, 4, 2, 3],
    }
)
print(athletes)
# --8<-- [end:athletes]

# --8<-- [start:athletes-sort-over-country]
result = athletes.select(
    pl.col("athlete", "rank").sort_by(pl.col("rank")).over(pl.col("country")),
    pl.col("country"),
)

print(result)
# --8<-- [end:athletes-sort-over-country]

# --8<-- [start:athletes-explode]
result = athletes.select(
    pl.all()
    .sort_by(pl.col("rank"))
    .over(pl.col("country"), mapping_strategy="explode"),
)

print(result)
# --8<-- [end:athletes-explode]

# --8<-- [start:athletes-join]
result = athletes.with_columns(
    pl.col("rank").sort().over(pl.col("country"), mapping_strategy="join"),
)

print(result)
# --8<-- [end:athletes-join]

# --8<-- [start:pokemon-mean]
result = pokemon.select(
    pl.col("Name", "Type 1", "Speed"),
    pl.col("Speed").mean().over(pl.col("Type 1")).alias("Mean speed in group"),
)

print(result)
# --8<-- [end:pokemon-mean]


# --8<-- [start:group_by]
result = pokemon.select(
    "Type 1",
    "Type 2",
    pl.col("Attack").mean().over("Type 1").alias("avg_attack_by_type"),
    pl.col("Defense")
    .mean()
    .over(["Type 1", "Type 2"])
    .alias("avg_defense_by_type_combination"),
    pl.col("Attack").mean().alias("avg_attack"),
)
print(result)
# --8<-- [end:group_by]

# --8<-- [start:operations]
filtered = pokemon.filter(pl.col("Type 2") == "Psychic").select(
    "Name",
    "Type 1",
    "Speed",
)
print(filtered)
# --8<-- [end:operations]

# --8<-- [start:sort]
result = filtered.with_columns(
    pl.col("Name", "Speed").sort_by("Speed", descending=True).over("Type 1"),
)
print(result)
# --8<-- [end:sort]

# --8<-- [start:examples]
result = pokemon.sort("Type 1").select(
    pl.col("Type 1").head(3).over("Type 1", mapping_strategy="explode"),
    pl.col("Name")
    .sort_by(pl.col("Speed"), descending=True)
    .head(3)
    .over("Type 1", mapping_strategy="explode")
    .alias("fastest/group"),
    pl.col("Name")
    .sort_by(pl.col("Attack"), descending=True)
    .head(3)
    .over("Type 1", mapping_strategy="explode")
    .alias("strongest/group"),
    pl.col("Name")
    .sort()
    .head(3)
    .over("Type 1", mapping_strategy="explode")
    .alias("sorted_by_alphabet"),
)
print(result)
# --8<-- [end:examples]
