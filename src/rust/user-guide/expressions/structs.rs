fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --8<-- [start:ratings_df]
    use polars::prelude::*;
    let ratings = df!(
            "Movie"=> ["Cars", "IT", "ET", "Cars", "Up", "IT", "Cars", "ET", "Up", "Cars"],
            "Theatre"=> ["NE", "ME", "IL", "ND", "NE", "SD", "NE", "IL", "IL", "NE"],
            "Avg_Rating"=> [4.5, 4.4, 4.6, 4.3, 4.8, 4.7, 4.5, 4.9, 4.7, 4.6],
            "Count"=> [30, 27, 26, 29, 31, 28, 28, 26, 33, 28],

    )?;
    println!("{}", &ratings);
    // --8<-- [end:ratings_df]

    // --8<-- [start:state_value_counts]
    let result = ratings
        .clone()
        .lazy()
        .select([col("Theatre").value_counts(true, true, "count", false)])
        .collect()?;
    println!("{}", result);
    // --8<-- [end:state_value_counts]

    // --8<-- [start:struct_unnest]
    let result = ratings
        .clone()
        .lazy()
        .select([col("Theatre").value_counts(true, true, "count", false)])
        .unnest(["Theatre"])
        .collect()?;
    println!("{}", result);
    // --8<-- [end:struct_unnest]

    // --8<-- [start:series_struct]
    // Don't think we can make it the same way in rust, but this works
    let rating_series = df!(
        "Movie" => &["Cars", "Toy Story"],
        "Theatre" => &["NE", "ME"],
        "Avg_Rating" => &[4.5, 4.9],
    )?
    .into_struct("ratings".into())
    .into_series();
    println!("{}", &rating_series);
    // // --8<-- [end:series_struct]

    // --8<-- [start:series_struct_error]
    // Contribute the Rust translation of the Python example by opening a PR.
    // --8<-- [end:series_struct_error]

    // --8<-- [start:series_struct_extract]
    let result = rating_series.struct_()?.field_by_name("Movie")?;
    println!("{}", result);
    // --8<-- [end:series_struct_extract]

    // --8<-- [start:series_struct_rename]
    // Contribute the Rust translation of the Python example by opening a PR.
    // --8<-- [end:series_struct_rename]

    // --8<-- [start:struct-rename-check]
    // Contribute the Rust translation of the Python example by opening a PR.
    // --8<-- [end:struct-rename-check]

    // --8<-- [start:struct_duplicates]
    // Contribute the Rust translation of the Python example by opening a PR.
    // --8<-- [end:struct_duplicates]

    // --8<-- [start:struct_ranking]
    let result = ratings
        .clone()
        .lazy()
        .with_columns([as_struct(vec![col("Count"), col("Avg_Rating")])
            .rank(
                RankOptions {
                    method: RankMethod::Dense,
                    descending: true,
                },
                None,
            )
            .over([col("Movie"), col("Theatre")])
            .alias("Rank")])
        // .filter(as_struct(&[col("Movie"), col("Theatre")]).is_duplicated())
        // Error: .is_duplicated() not available if you try that
        // https://github.com/pola-rs/polars/issues/3803
        .filter(len().over([col("Movie"), col("Theatre")]).gt(lit(1)))
        .collect()?;
    println!("{}", result);
    // --8<-- [end:struct_ranking]

    // --8<-- [start:multi_column_apply]
    let df = df!(
        "keys" => ["a", "a", "b"],
        "values" => [10, 7, 1],
    )?;

    let result = df
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
                        let s_a = &ca.fields_as_series()[0];
                        let s_b = &ca.fields_as_series()[1];

                        // downcast the `Series` to their known type
                        let ca_a = s_a.str()?;
                        let ca_b = s_b.i32()?;

                        // iterate both `ChunkedArrays`
                        let result: Int32Chunked = ca_a
                            .into_iter()
                            .zip(ca_b)
                            .map(|(opt_a, opt_b)| match (opt_a, opt_b) {
                                (Some(a), Some(b)) => Some(a.len() as i32 + b),
                                _ => None,
                            })
                            .collect();

                        Ok(Some(result.into_column()))
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
    println!("{}", result);
    // --8<-- [end:multi_column_apply]

    // --8<-- [start:ack]
    // Contribute the Rust translation of the Python example by opening a PR.
    // --8<-- [end:ack]

    // --8<-- [start:struct-ack]
    // Contribute the Rust translation of the Python example by opening a PR.
    // --8<-- [end:struct-ack]

    Ok(())
}
