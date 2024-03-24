// --8<-- [start:setup]
use polars::prelude::pivot::pivot;
use polars::prelude::*;
// --8<-- [end:setup]

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --8<-- [start:df]
    let df = df!(
            "foo"=> ["A", "A", "B", "B", "C"],
            "bar"=> ["k", "l", "m", "n", "o"],
            "N"=> [1, 2, 2, 4, 2],
    )?;
    println!("{}", &df);
    // --8<-- [end:df]

    // --8<-- [start:eager]
    let out = pivot(&df, ["foo"], ["bar"], Some(["N"]), false, None, None)?;
    println!("{}", &out);
    // --8<-- [end:eager]

    // --8<-- [start:lazy]
    let q = df.lazy();
    let q2 = pivot(
        &q.collect()?,
        ["foo"],
        ["bar"],
        Some(["N"]),
        false,
        None,
        None,
    )?
    .lazy();
    let out = q2.collect()?;
    println!("{}", &out);
    // --8<-- [end:lazy]

    Ok(())
}
