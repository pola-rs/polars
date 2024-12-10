use polars::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --8<-- [start:dataframe]
    let df = df!(
        "keys" => &["a", "a", "b", "b"],
        "values" => &[10, 7, 1, 23],
    )?;
    println!("{}", df);
    // --8<-- [end:dataframe]

    // --8<-- [start:individual_log]
    // --8<-- [end:individual_log]

    // --8<-- [start:diff_from_mean]
    // --8<-- [end:diff_from_mean]

    // --8<-- [start:np_log]
    // --8<-- [end:np_log]

    // --8<-- [start:diff_from_mean_numba]
    // --8<-- [end:diff_from_mean_numba]

    // --8<-- [start:combine]
    // --8<-- [end:combine]
    Ok(())
}
