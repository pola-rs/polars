// --8<-- [start:setup]
use polars::prelude::*;
// --8<-- [end:setup]

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --8<-- [start:df]
    let df = df! (
            "animal" => &[Some("Crab"), Some("cat and dog"), Some("rab$bit"), None],
    )?;

    let out = df
        .clone()
        .lazy()
        .select([
            col("animal").str().len_bytes().alias("byte_count"),
            col("animal").str().len_chars().alias("letter_count"),
        ])
        .collect()?;

    println!("{}", &out);
    // --8<-- [end:df]

    // --8<-- [start:existence]
    let out = df
        .clone()
        .lazy()
        .select([
            col("animal"),
            col("animal")
                .str()
                .contains(lit("cat|bit"), false)
                .alias("regex"),
            col("animal")
                .str()
                .contains_literal(lit("rab$"))
                .alias("literal"),
            col("animal")
                .str()
                .starts_with(lit("rab"))
                .alias("starts_with"),
            col("animal").str().ends_with(lit("dog")).alias("ends_with"),
        ])
        .collect()?;
    println!("{}", &out);
    // --8<-- [end:existence]

    // --8<-- [start:extract]
    let df = df!(
            "a" =>  &[
                "http://vote.com/ballon_dor?candidate=messi&ref=polars",
                "http://vote.com/ballon_dor?candidat=jorginho&ref=polars",
                "http://vote.com/ballon_dor?candidate=ronaldo&ref=polars",
            ]
    )?;
    let out = df
        .clone()
        .lazy()
        .select([col("a").str().extract(lit(r"candidate=(\w+)"), 1)])
        .collect()?;
    println!("{}", &out);
    // --8<-- [end:extract]

    // --8<-- [start:extract_all]
    let df = df!("foo"=> &["123 bla 45 asd", "xyz 678 910t"])?;
    let out = df
        .clone()
        .lazy()
        .select([col("foo")
            .str()
            .extract_all(lit(r"(\d+)"))
            .alias("extracted_nrs")])
        .collect()?;
    println!("{}", &out);
    // --8<-- [end:extract_all]

    // --8<-- [start:replace]
    let df = df!("id"=> &[1, 2], "text"=> &["123abc", "abc456"])?;
    let out = df
        .clone()
        .lazy()
        .with_columns([
            col("text").str().replace(lit(r"abc\b"), lit("ABC"), false),
            col("text")
                .str()
                .replace_all(lit("a"), lit("-"), false)
                .alias("text_replace_all"),
        ])
        .collect()?;
    println!("{}", &out);
    // --8<-- [end:replace]

    Ok(())
}
