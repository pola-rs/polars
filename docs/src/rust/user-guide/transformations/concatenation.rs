// --8<-- [start:setup]
use polars::prelude::*;
// --8<-- [end:setup]

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --8<-- [start:vertical]
    let df_v1 = df!(
            "a"=> &[1],
            "b"=> &[3],
    )?;
    let df_v2 = df!(
            "a"=> &[2],
            "b"=> &[4],
    )?;
    let df_vertical_concat = concat(
        [df_v1.clone().lazy(), df_v2.clone().lazy()],
        UnionArgs::default(),
    )?
    .collect()?;
    println!("{}", &df_vertical_concat);
    // --8<-- [end:vertical]

    // --8<-- [start:horizontal]
    let df_h1 = df!(
            "l1"=> &[1, 2],
            "l2"=> &[3, 4],
    )?;
    let df_h2 = df!(
            "r1"=> &[5, 6],
            "r2"=> &[7, 8],
            "r3"=> &[9, 10],
    )?;
    let df_horizontal_concat = polars::functions::concat_df_horizontal(&[df_h1, df_h2])?;
    println!("{}", &df_horizontal_concat);
    // --8<-- [end:horizontal]
    //
    // --8<-- [start:horizontal_different_lengths]
    let df_h1 = df!(
            "l1"=> &[1, 2],
            "l2"=> &[3, 4],
    )?;
    let df_h2 = df!(
            "r1"=> &[5, 6, 7],
            "r2"=> &[8, 9, 10],
    )?;
    let df_horizontal_concat = polars::functions::concat_df_horizontal(&[df_h1, df_h2])?;
    println!("{}", &df_horizontal_concat);
    // --8<-- [end:horizontal_different_lengths]

    // --8<-- [start:cross]
    let df_d1 = df!(
        "a"=> &[1],
        "b"=> &[3],
    )?;
    let df_d2 = df!(
            "a"=> &[2],
            "d"=> &[4],)?;
    let df_diagonal_concat = polars::functions::concat_df_diagonal(&[df_d1, df_d2])?;
    println!("{}", &df_diagonal_concat);
    // --8<-- [end:cross]
    Ok(())
}
