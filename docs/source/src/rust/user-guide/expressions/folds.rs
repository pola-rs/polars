fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --8<-- [start:mansum]
    use polars::prelude::*;
    // Contribute the Rust translation of the Python example by opening a PR.
    // --8<-- [end:mansum]

    // --8<-- [start:mansum-explicit]
    // Contribute the Rust translation of the Python example by opening a PR.
    // --8<-- [end:mansum-explicit]

    // --8<-- [start:manprod]
    // Contribute the Rust translation of the Python example by opening a PR.
    // --8<-- [end:manprod]

    // --8<-- [start:manprod-fixed]
    // Contribute the Rust translation of the Python example by opening a PR.
    // --8<-- [end:manprod-fixed]

    // --8<-- [start:conditional]
    // Contribute the Rust translation of the Python example by opening a PR.
    // --8<-- [end:conditional]

    // --8<-- [start:string]
    let df = df!(
        "a" => ["a", "b", "c"],
        "b" => [1, 2, 3],
    )?;

    let result = df
        .lazy()
        .select([concat_str([col("a"), col("b")], "", false)])
        .collect()?;
    println!("{:?}", result);
    // --8<-- [end:string]

    Ok(())
}
