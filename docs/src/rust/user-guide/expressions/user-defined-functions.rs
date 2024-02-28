use polars::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --8<-- [start:dataframe]
    let df = df!(
        "keys" => &["a", "a", "b"],
        "values" => &[10, 7, 1],
    )?;
    println!("{}", df);
    // --8<-- [end:dataframe]

    // --8<-- [start:custom_sum]
    // --8<-- [end:custom_sum]

    // --8<-- [start:custom_sum_numba]
    // --8<-- [end:custom_sum_numba]

    // --8<-- [start:dataframe2]
    // --8<-- [end:dataframe2]

    // --8<-- [start:custom_mean_numba]
    // --8<-- [end:custom_mean_numba]

    // --8<-- [start:combine]
    let out = df
        .lazy()
        .select([
            // pack to struct to get access to multiple fields in a custom `apply/map`
            as_struct(vec![col("keys"), col("values")])
                // we will compute the len(a) + b
                .apply(
                    |s| {
                        // downcast to struct
                        let ca = s.struct_()?;

                        // get the fields as Series
                        let s_a = &ca.fields()[0];
                        let s_b = &ca.fields()[1];

                        // downcast the `Series` to their known type
                        let ca_a = s_a.str()?;
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

                        Ok(Some(out.into_series()))
                    },
                    GetOutput::from_type(DataType::Int32),
                )
                // note: the `'solution_map_elements'` alias is just there to show how you
                // get the same output as in the Python API example.
                .alias("solution_map_elements"),
            (col("keys").str().count_matches(lit("."), true) + col("values"))
                .alias("solution_expr"),
        ])
        .collect()?;
    println!("{}", out);
    // --8<-- [end:combine]
    Ok(())
}
