# --8<-- [start:setup]
import polars as pl

# --8<-- [end:setup]


# --8<-- [start:df]
df = pl.DataFrame({"animal": ["Crab", "cat and dog", "rab$bit", None]})

out = df.select(
    pl.col("animal").str.len_bytes().alias("byte_count"),
    pl.col("animal").str.len_chars().alias("letter_count"),
)
print(out)
# --8<-- [end:df]

# --8<-- [start:existence]
out = df.select(
    pl.col("animal"),
    pl.col("animal").str.contains("cat|bit").alias("regex"),
    pl.col("animal").str.contains("rab$", literal=True).alias("literal"),
    pl.col("animal").str.starts_with("rab").alias("starts_with"),
    pl.col("animal").str.ends_with("dog").alias("ends_with"),
)
print(out)
# --8<-- [end:existence]

# --8<-- [start:extract]
df = pl.DataFrame(
    {
        "a": [
            "http://vote.com/ballon_dor?candidate=messi&ref=polars",
            "http://vote.com/ballon_dor?candidat=jorginho&ref=polars",
            "http://vote.com/ballon_dor?candidate=ronaldo&ref=polars",
        ]
    }
)
out = df.select(
    pl.col("a").str.extract(r"candidate=(\w+)", group_index=1),
)
print(out)
# --8<-- [end:extract]


# --8<-- [start:extract_all]
df = pl.DataFrame({"foo": ["123 bla 45 asd", "xyz 678 910t"]})
out = df.select(
    pl.col("foo").str.extract_all(r"(\d+)").alias("extracted_nrs"),
)
print(out)
# --8<-- [end:extract_all]


# --8<-- [start:replace]
df = pl.DataFrame({"id": [1, 2], "text": ["123abc", "abc456"]})
out = df.with_columns(
    pl.col("text").str.replace(r"abc\b", "ABC"),
    pl.col("text").str.replace_all("a", "-", literal=True).alias("text_replace_all"),
)
print(out)
# --8<-- [end:replace]
