# --8<-- [start:df]
import polars as pl

df = pl.DataFrame(
    {
        "language": ["English", "Dutch", "Portuguese", "Finish"],
        "fruit": ["pear", "peer", "pêra", "päärynä"],
    }
)

result = df.with_columns(
    pl.col("fruit").str.len_bytes().alias("byte_count"),
    pl.col("fruit").str.len_chars().alias("letter_count"),
)
print(result)
# --8<-- [end:df]

# --8<-- [start:existence]
result = df.select(
    pl.col("fruit"),
    pl.col("fruit").str.starts_with("p").alias("starts_with_p"),
    pl.col("fruit").str.contains("p..r").alias("p..r"),
    pl.col("fruit").str.contains("e+").alias("e+"),
    pl.col("fruit").str.ends_with("r").alias("ends_with_r"),
)
print(result)
# --8<-- [end:existence]

# --8<-- [start:extract]
df = pl.DataFrame(
    {
        "urls": [
            "http://vote.com/ballon_dor?candidate=messi&ref=polars",
            "http://vote.com/ballon_dor?candidat=jorginho&ref=polars",
            "http://vote.com/ballon_dor?candidate=ronaldo&ref=polars",
        ]
    }
)
result = df.select(
    pl.col("urls").str.extract(r"candidate=(\w+)", group_index=1),
)
print(result)
# --8<-- [end:extract]


# --8<-- [start:extract_all]
df = pl.DataFrame({"text": ["123 bla 45 asd", "xyz 678 910t"]})
result = df.select(
    pl.col("text").str.extract_all(r"(\d+)").alias("extracted_nrs"),
)
print(result)
# --8<-- [end:extract_all]


# --8<-- [start:replace]
df = pl.DataFrame({"text": ["123abc", "abc456"]})
result = df.with_columns(
    pl.col("text").str.replace(r"\d", "-"),
    pl.col("text").str.replace_all(r"\d", "-").alias("text_replace_all"),
)
print(result)
# --8<-- [end:replace]

# --8<-- [start:casing]
addresses = pl.DataFrame(
    {
        "addresses": [
            "128 PERF st",
            "Rust blVD, 158",
            "PoLaRs Av, 12",
            "1042 Query sq",
        ]
    }
)

addresses = addresses.select(
    pl.col("addresses").alias("originals"),
    pl.col("addresses").str.to_titlecase(),
    pl.col("addresses").str.to_lowercase().alias("lower"),
    pl.col("addresses").str.to_uppercase().alias("upper"),
)
print(addresses)
# --8<-- [end:casing]

# --8<-- [start:strip]
addr = pl.col("addresses")
chars = ", 0123456789"
result = addresses.select(
    addr.str.strip_chars(chars).alias("strip"),
    addr.str.strip_chars_end(chars).alias("end"),
    addr.str.strip_chars_start(chars).alias("start"),
    addr.str.strip_prefix("128 ").alias("prefix"),
    addr.str.strip_suffix(", 158").alias("suffix"),
)
print(result)
# --8<-- [end:strip]

# --8<-- [start:slice]
df = pl.DataFrame(
    {
        "fruits": ["pear", "mango", "dragonfruit", "passionfruit"],
        "n": [1, -1, 4, -4],
    }
)

result = df.with_columns(
    pl.col("fruits").str.slice(pl.col("n")).alias("slice"),
    pl.col("fruits").str.head(pl.col("n")).alias("head"),
    pl.col("fruits").str.tail(pl.col("n")).alias("tail"),
)
print(result)
# --8<-- [end:slice]
