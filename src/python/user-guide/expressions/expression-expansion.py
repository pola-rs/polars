# --8<-- [start:df]
import polars as pl

df = pl.DataFrame(
    {  # As of 14th October 2024, ~3pm UTC
        "ticker": ["AAPL", "NVDA", "MSFT", "GOOG", "AMZN"],
        "company_name": ["Apple", "NVIDIA", "Microsoft", "Alphabet (Google)", "Amazon"],
        "price": [229.9, 138.93, 420.56, 166.41, 188.4],
        "day_high": [231.31, 139.6, 424.04, 167.62, 189.83],
        "day_low": [228.6, 136.3, 417.52, 164.78, 188.44],
        "year_high": [237.23, 140.76, 468.35, 193.31, 201.2],
        "year_low": [164.08, 39.23, 324.39, 121.46, 118.35],
    }
)

print(df)
# --8<-- [end:df]

# --8<-- [start:col-with-names]
eur_usd_rate = 1.09  # As of 14th October 2024

result = df.with_columns(
    (
        pl.col(
            "price",
            "day_high",
            "day_low",
            "year_high",
            "year_low",
        )
        / eur_usd_rate
    ).round(2)
)
print(result)
# --8<-- [end:col-with-names]

# --8<-- [start:expression-list]
exprs = [
    (pl.col("price") / eur_usd_rate).round(2),
    (pl.col("day_high") / eur_usd_rate).round(2),
    (pl.col("day_low") / eur_usd_rate).round(2),
    (pl.col("year_high") / eur_usd_rate).round(2),
    (pl.col("year_low") / eur_usd_rate).round(2),
]

result2 = df.with_columns(exprs)
print(result.equals(result2))
# --8<-- [end:expression-list]

# --8<-- [start:col-with-dtype]
result = df.with_columns((pl.col(pl.Float64) / eur_usd_rate).round(2))
print(result)
# --8<-- [end:col-with-dtype]

# --8<-- [start:col-with-dtypes]
result2 = df.with_columns(
    (
        pl.col(
            pl.Float32,
            pl.Float64,
        )
        / eur_usd_rate
    ).round(2)
)
print(result.equals(result2))
# --8<-- [end:col-with-dtypes]

# --8<-- [start:col-with-regex]
result = df.select(pl.col("ticker", "^.*_high$", "^.*_low$"))
print(result)
# --8<-- [end:col-with-regex]

# --8<-- [start:col-error]
try:
    df.select(pl.col("ticker", pl.Float64))
except TypeError as err:
    print("TypeError:", err)
# --8<-- [end:col-error]

# --8<-- [start:all]
result = df.select(pl.all())
print(result.equals(df))
# --8<-- [end:all]

# --8<-- [start:all-exclude]
result = df.select(pl.all().exclude("^day_.*$"))
print(result)
# --8<-- [end:all-exclude]

# --8<-- [start:col-exclude]
result = df.select(pl.col(pl.Float64).exclude("^day_.*$"))
print(result)
# --8<-- [end:col-exclude]

# --8<-- [start:duplicate-error]
from polars.exceptions import DuplicateError

gbp_usd_rate = 1.31  # As of 14th October 2024

try:
    df.select(
        pl.col("price") / gbp_usd_rate,  # This would be named "price"...
        pl.col("price") / eur_usd_rate,  # And so would this.
    )
except DuplicateError as err:
    print("DuplicateError:", err)
# --8<-- [end:duplicate-error]

# --8<-- [start:alias]
result = df.select(
    (pl.col("price") / gbp_usd_rate).alias("price (GBP)"),
    (pl.col("price") / eur_usd_rate).alias("price (EUR)"),
)
# --8<-- [end:alias]

# --8<-- [start:prefix-suffix]
result = df.select(
    (pl.col("^year_.*$") / eur_usd_rate).name.prefix("in_eur_"),
    (pl.col("day_high", "day_low") / gbp_usd_rate).name.suffix("_gbp"),
)
print(result)
# --8<-- [end:prefix-suffix]

# --8<-- [start:name-map]
# There is also `.name.to_uppercase`, so this usage of `.map` is moot.
result = df.select(pl.all().name.map(str.upper))
print(result)
# --8<-- [end:name-map]

# --8<-- [start:for-with_columns]
result = df
for tp in ["day", "year"]:
    result = result.with_columns(
        (pl.col(f"{tp}_high") - pl.col(f"{tp}_low")).alias(f"{tp}_amplitude")
    )
print(result)
# --8<-- [end:for-with_columns]


# --8<-- [start:yield-expressions]
def amplitude_expressions(time_periods):
    for tp in time_periods:
        yield (pl.col(f"{tp}_high") - pl.col(f"{tp}_low")).alias(f"{tp}_amplitude")


result = df.with_columns(amplitude_expressions(["day", "year"]))
print(result)
# --8<-- [end:yield-expressions]

# --8<-- [start:selectors]
import polars.selectors as cs

result = df.select(cs.string() | cs.ends_with("_high"))
print(result)
# --8<-- [end:selectors]

# --8<-- [start:selectors-set-operations]
result = df.select(cs.contains("_") - cs.string())
print(result)
# --8<-- [end:selectors-set-operations]

# --8<-- [start:selectors-expressions]
result = df.select((cs.contains("_") - cs.string()) / eur_usd_rate)
print(result)
# --8<-- [end:selectors-expressions]

# --8<-- [start:selector-ambiguity]
people = pl.DataFrame(
    {
        "name": ["Anna", "Bob"],
        "has_partner": [True, False],
        "has_kids": [False, False],
        "has_tattoos": [True, False],
        "is_alive": [True, True],
    }
)

wrong_result = people.select((~cs.starts_with("has_")).name.prefix("not_"))
print(wrong_result)
# --8<-- [end:selector-ambiguity]

# --8<-- [start:as_expr]
result = people.select((~cs.starts_with("has_").as_expr()).name.prefix("not_"))
print(result)
# --8<-- [end:as_expr]

# --8<-- [start:is_selector]
print(cs.is_selector(~cs.starts_with("has_").as_expr()))
# --8<-- [end:is_selector]

# --8<-- [start:expand_selector]
print(
    cs.expand_selector(
        people,
        cs.starts_with("has_"),
    )
)
# --8<-- [end:expand_selector]
