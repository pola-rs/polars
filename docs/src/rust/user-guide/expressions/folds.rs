use polars::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    
    // --8<-- [start:mansum]
    let df = df!(
        "a" => &[1, 2, 3],
        "b" => &[10, 20, 30],
    )?;

    let out = df
        .lazy()
        .select([fold_exprs(lit(0), |acc, x| Ok(Some(acc + x)), [col("*")]).alias("sum")])
        .collect()?;
    println!("{}", out);
    // --8<-- [end:mansum]

    // --8<-- [start:conditional]
    let df = df!(
        "a" => &[1, 2, 3],
        "b" => &[0, 1, 2],
    )?;

    let out = df
        .lazy()
        .filter(fold_exprs(
            lit(true),
            |acc, x| Some(acc.bitand(&x)),
            [col("*").gt(1)],
        ))
        .collect()?;
    println!("{}", out);
    // --8<-- [end:conditional]

    // --8<-- [start:string]
    let df = df!(
        "a" => &["a", "b", "c"],
        "b" => &[1, 2, 3],
    )?;

    let out = df
        .lazy()
        .select([concat_str([col("a"), col("b")], "")])
        .collect()?;
    println!("{:?}", out);
    // --8<-- [end:string]

    Ok(())
}