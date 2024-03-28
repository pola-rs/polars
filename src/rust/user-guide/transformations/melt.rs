// --8<-- [start:setup]
use polars::prelude::*;
// --8<-- [end:setup]

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --8<-- [start:df]
    let df = df!(
            "A"=> &["a", "b", "a"],
            "B"=> &[1, 3, 5],
            "C"=> &[10, 11, 12],
            "D"=> &[2, 4, 6],
    )?;
    println!("{}", &df);
    // --8<-- [end:df]

    // --8<-- [start:melt]
    let out = df.melt(["A", "B"], ["C", "D"])?;
    println!("{}", &out);
    // --8<-- [end:melt]
    Ok(())
}
