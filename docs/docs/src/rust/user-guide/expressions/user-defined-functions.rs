use polars::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {

    // --8<-- [start:dataframe]
    let df = df!(
        "keys" => &["a", "a", "b"],
        "values" => &[10, 7, 1],
    )?;

    let out = df
        .lazy()
        .groupby(["keys"])
        .agg([
            col("values")
                .map(|s| Ok(s.shift(1)), GetOutput::default())
                .alias("shift_map"),
            col("values").shift(1).alias("shift_expression"),
        ])
        .collect()?;

    println!("{}", out);
    // --8<-- [end:dataframe]

    // --8<-- [start:apply]
    let out = df
        .clone()
        .lazy()
        .groupby([col("keys")])
        .agg([
            col("values")
                .apply(|s| Ok(s.shift(1)), GetOutput::default())
                .alias("shift_map"),
            col("values").shift(1).alias("shift_expression"),
        ])
        .collect()?;
    println!("{}", out);
    // --8<-- [end:apply]

    // --8<-- [start:counter]
    
    // --8<-- [end:counter]

    // --8<-- [start:combine]
    let out = df
        .lazy()
        .select([
            // pack to struct to get access to multiple fields in a custom `apply/map`
            as_struct(&[col("keys"), col("values")])
                // we will compute the len(a) + b
                .apply(
                    |s| {
                        // downcast to struct
                        let ca = s.struct_()?;

                        // get the fields as Series
                        let s_a = &ca.fields()[0];
                        let s_b = &ca.fields()[1];

                        // downcast the `Series` to their known type
                        let ca_a = s_a.utf8()?;
                        let ca_b = s_b.i32()?;

                        // iterate both `ChunkedArrays`
                        let out: Int32Chunked = ca_a
                            .into_iter()
                            .zip(ca_b)
                            .map(|(opt_a, opt_b)| match (opt_a, opt_b) {
                                (Some(a), Some(b)) => Some(a.len() as i32 + b),
                                _ => None,
                            })
                            .collect();

                        Ok(out.into_series())
                    },
                    GetOutput::from_type(DataType::Int32),
                )
                .alias("solution_apply"),
            (col("keys").str().count_match(".") + col("values")).alias("solution_expr"),
        ])
        .collect()?;
    println!("{}", out);
    // --8<-- [end:combine]
    Ok(())
}