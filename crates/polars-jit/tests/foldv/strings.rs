use polars::prelude::*;
use polars_jit::foldv;

fn strings(values: &[Option<&str>]) -> PolarsResult<DataFrame> {
    df!("s" => values)
}

fn string_column(result: &DataFrame) -> PolarsResult<Vec<Option<String>>> {
    Ok(result
        .column("result")?
        .str()?
        .iter()
        .map(|value| value.map(str::to_owned))
        .collect())
}

#[test]
fn foldv_concatenates_rows_into_a_string_state() -> PolarsResult<()> {
    // Each row's state is a new owned string built from the previous state.
    let result = strings(&[Some("a"), Some("é"), Some(""), Some("日本")])?
        .lazy()
        .select([foldv(lit(""), Expr::Element + col("s"))?.alias("result")])
        .collect()?;

    assert_eq!(
        string_column(&result)?,
        ["a", "aé", "aé", "aé日本"].map(|value| Some(value.to_owned()))
    );
    Ok(())
}

#[test]
fn foldv_replaces_a_string_state_with_an_input_value() -> PolarsResult<()> {
    // The longest string so far: the state is replaced by a borrowed input value, which the
    // runtime copies before the input row is left behind.
    let function = when(
        col("s")
            .str()
            .len_chars()
            .gt(Expr::Element.str().len_chars()),
    )
    .then(col("s"))
    .otherwise(Expr::Element);
    let result = strings(&[Some("ab"), Some("x"), Some("日本語"), Some("abcd")])?
        .lazy()
        .select([foldv(lit(""), function)?.alias("result")])
        .collect()?;

    assert_eq!(
        string_column(&result)?,
        ["ab", "ab", "日本語", "abcd"].map(|value| Some(value.to_owned()))
    );
    Ok(())
}

#[test]
fn foldv_compares_strings_by_their_bytes() -> PolarsResult<()> {
    // Polars orders strings by their UTF-8 bytes, so "é" sorts after "z".
    let function = when(col("s").gt(lit("b")))
        .then(Expr::Element + lit(1_i64))
        .otherwise(Expr::Element);
    let result = strings(&[Some("a"), Some("b"), Some("c"), Some("é"), Some("z")])?
        .lazy()
        .select([foldv(lit(0_i64).cast(DataType::Int64), function)?.alias("result")])
        .collect()?;

    assert_eq!(
        result.column("result")?.i64()?.to_vec(),
        [Some(0), Some(0), Some(1), Some(2), Some(3)]
    );
    Ok(())
}

#[test]
fn foldv_counts_bytes_and_characters() -> PolarsResult<()> {
    // "é" is one character in two bytes; "日本" is two characters in six bytes.
    let df = strings(&[Some("é"), Some("日本"), None])?;
    let result = df
        .lazy()
        .select([
            foldv(
                lit(0_i64).cast(DataType::Int64),
                Expr::Element + col("s").str().len_bytes(),
            )?
            .alias("bytes"),
            foldv(
                lit(0_i64).cast(DataType::Int64),
                Expr::Element + col("s").str().len_chars(),
            )?
            .alias("chars"),
        ])
        .collect()?;

    assert_eq!(
        result.column("bytes")?.i64()?.to_vec(),
        [Some(2), Some(8), None]
    );
    assert_eq!(
        result.column("chars")?.i64()?.to_vec(),
        [Some(1), Some(3), None]
    );
    Ok(())
}

#[test]
fn foldv_joins_columns_with_concat_str() -> PolarsResult<()> {
    // concat_str inserts the separator between inputs and is null when any input is null.
    let df = df!(
        "s" => [Some("a"), Some("b"), None],
        "t" => [Some("x"), Some("y"), Some("z")],
    )?;
    let result = df
        .lazy()
        .select([foldv(
            lit(""),
            when(Expr::Element.is_null())
                .then(lit("reset"))
                .otherwise(Expr::Element + concat_str([col("s"), col("t")], "-", false)),
        )?
        .alias("result")])
        .collect()?;

    assert_eq!(
        string_column(&result)?,
        [Some("a-x".to_owned()), Some("a-xb-y".to_owned()), None]
    );
    Ok(())
}

#[test]
fn foldv_recovers_a_string_state_from_null() -> PolarsResult<()> {
    // A null input makes the concatenation null; fill_null restarts from the current value.
    let function = Expr::Element.fill_null(lit("")) + col("s").fill_null(lit("?"));
    let result = strings(&[Some("a"), None, Some("b")])?
        .lazy()
        .select([foldv(lit(NULL).cast(DataType::String), function)?.alias("result")])
        .collect()?;

    assert_eq!(
        string_column(&result)?,
        ["a", "a?", "a?b"].map(|value| Some(value.to_owned()))
    );
    Ok(())
}

#[test]
fn foldv_returns_an_empty_string_column_for_empty_input() -> PolarsResult<()> {
    let result = df!("s" => Vec::<&str>::new())?
        .lazy()
        .select([foldv(lit("start"), Expr::Element + col("s"))?.alias("result")])
        .collect()?;

    assert_eq!(result.column("result")?.dtype(), &DataType::String);
    assert!(result.column("result")?.is_empty());
    Ok(())
}

#[test]
fn foldv_builds_a_long_string_over_many_rows() -> PolarsResult<()> {
    // Repeated concatenation replaces the state on every row; each row's output is its own copy.
    let rows = 2_000;
    let values = (0..rows)
        .map(|row| Some(if row % 2 == 0 { "é" } else { "x" }))
        .collect::<Vec<_>>();
    let result = strings(&values)?
        .lazy()
        .select([foldv(lit(""), Expr::Element + col("s"))?.alias("result")])
        .collect()?;

    let column = result.column("result")?.str()?;
    assert_eq!(column.get(0), Some("é"));
    assert_eq!(
        column.get(rows - 1).map(|value| value.chars().count()),
        Some(rows)
    );
    assert_eq!(column.get(rows - 1).map(str::len), Some(rows / 2 * 3));
    Ok(())
}

#[test]
fn foldv_releases_strings_when_a_strict_cast_stops_the_fold() -> PolarsResult<()> {
    // The runtime owns every string, so an error in the middle of the loop leaks nothing.
    let function = when(col("n").strict_cast(DataType::Int32).gt(lit(0_i32)))
        .then(Expr::Element + col("s"))
        .otherwise(Expr::Element);
    let error = df!("s" => ["a", "b", "c"], "n" => [1_i64, 1, 3_000_000_000])?
        .lazy()
        .select([foldv(lit(""), function)?])
        .collect()
        .unwrap_err();

    assert!(
        error.to_string().contains("failed in foldv at row 2"),
        "{error}"
    );
    Ok(())
}

#[test]
fn foldv_string_output_outlives_its_inputs() -> PolarsResult<()> {
    // The output owns its bytes; it does not borrow the input or the call's scratch storage.
    let result = {
        let df = strings(&[Some("ab"), Some("cd")])?;
        df.lazy()
            .select([foldv(lit(""), col("s"))?.alias("result")])
            .collect()?
    };

    assert_eq!(
        string_column(&result)?,
        ["ab", "cd"].map(|value| Some(value.to_owned()))
    );
    Ok(())
}
