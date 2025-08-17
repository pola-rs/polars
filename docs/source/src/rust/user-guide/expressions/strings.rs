fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --8<-- [start:df]
    use polars::prelude::*;

    let df = df! (
        "language" => ["English", "Dutch", "Portuguese", "Finish"],
        "fruit" => ["pear", "peer", "pêra", "päärynä"],
    )?;

    let result = df
        .clone()
        .lazy()
        .with_columns([
            col("fruit").str().len_bytes().alias("byte_count"),
            col("fruit").str().len_chars().alias("letter_count"),
        ])
        .collect()?;

    println!("{result}");
    // --8<-- [end:df]

    // --8<-- [start:existence]
    let result = df
        .lazy()
        .select([
            col("fruit"),
            col("fruit")
                .str()
                .starts_with(lit("p"))
                .alias("starts_with_p"),
            col("fruit").str().contains(lit("p..r"), true).alias("p..r"),
            col("fruit").str().contains(lit("e+"), true).alias("e+"),
            col("fruit").str().ends_with(lit("r")).alias("ends_with_r"),
        ])
        .collect()?;

    println!("{result}");
    // --8<-- [end:existence]

    // --8<-- [start:extract]
    let df = df! (
        "urls" => [
            "http://vote.com/ballon_dor?candidate=messi&ref=polars",
            "http://vote.com/ballon_dor?candidat=jorginho&ref=polars",
            "http://vote.com/ballon_dor?candidate=ronaldo&ref=polars",
        ]
    )?;

    let result = df
        .lazy()
        .select([col("urls").str().extract(lit(r"candidate=(\w+)"), 1)])
        .collect()?;

    println!("{result}");
    // --8<-- [end:extract]

    // --8<-- [start:extract_all]
    let df = df! (
        "text" => ["123 bla 45 asd", "xyz 678 910t"]
    )?;

    let result = df
        .lazy()
        .select([col("text")
            .str()
            .extract_all(lit(r"(\d+)"))
            .alias("extracted_nrs")])
        .collect()?;

    println!("{result}");
    // --8<-- [end:extract_all]

    // --8<-- [start:replace]
    let df = df! (
        "text" => ["123abc", "abc456"]
    )?;

    let result = df
        .lazy()
        .with_columns([
            col("text").str().replace(lit(r"\d"), lit("-"), false),
            col("text")
                .str()
                .replace_all(lit(r"\d"), lit("-"), false)
                .alias("text_replace_all"),
        ])
        .collect()?;

    println!("{result}");
    // --8<-- [end:replace]

    // --8<-- [start:casing]
    let addresses = df! (
        "addresses" => [
            "128 PERF st",
            "Rust blVD, 158",
            "PoLaRs Av, 12",
            "1042 Query sq",
        ]
    )?;

    let addresses = addresses
        .lazy()
        .select([
            col("addresses").alias("originals"),
            col("addresses").str().to_titlecase(),
            col("addresses").str().to_lowercase().alias("lower"),
            col("addresses").str().to_uppercase().alias("upper"),
        ])
        .collect()?;

    println!("{addresses}");
    // --8<-- [end:casing]

    // --8<-- [start:strip]
    let addr = col("addresses");
    let chars = lit(", 0123456789");
    let result = addresses
        .lazy()
        .select([
            addr.clone().str().strip_chars(chars.clone()).alias("strip"),
            addr.clone()
                .str()
                .strip_chars_end(chars.clone())
                .alias("end"),
            addr.clone().str().strip_chars_start(chars).alias("start"),
            addr.clone().str().strip_prefix(lit("128 ")).alias("prefix"),
            addr.str().strip_suffix(lit(", 158")).alias("suffix"),
        ])
        .collect()?;

    println!("{result}");
    // --8<-- [end:strip]

    // --8<-- [start:slice]
    let df = df! (
        "fruits" => ["pear", "mango", "dragonfruit", "passionfruit"],
        "n" => [1, -1, 4, -4],
    )?;

    let result = df
        .lazy()
        .with_columns([
            col("fruits")
                .str()
                .slice(col("n"), lit(NULL))
                .alias("slice"),
            col("fruits").str().head(col("n")).alias("head"),
            col("fruits").str().tail(col("n")).alias("tail"),
        ])
        .collect()?;

    println!("{result}");
    // --8<-- [end:slice]

    Ok(())
}
