use super::*;

#[test]
#[cfg(feature = "arange")]
fn test_arange_agg() -> Result<()> {
    let df = df![
        "x" => [5, 5, 4, 4, 2, 2]
    ]?;

    let out = df
        .lazy()
        .with_columns([arange(lit(0i32), count(), 1).over([col("x")])])
        .collect()?;
    assert_eq!(
        Vec::from_iter(out.column("literal")?.i64()?.into_no_null_iter()),
        &[0, 1, 0, 1, 0, 1]
    );

    Ok(())
}

#[test]
#[cfg(all(feature = "unique_counts", feature = "log"))]
fn test_groups_update() -> Result<()> {
    let df = df!["group" => ["A" ,"A", "A", "B", "B", "B", "B"],
    "id"=> [1, 1, 2, 3, 4, 3, 5]
    ]?;

    let out = df
        .lazy()
        .groupby_stable([col("group")])
        .agg([col("id").unique_counts().log(2.0)])
        .explode([col("id")])
        .collect()?;
    assert_eq!(
        out.column("id")?
            .f64()?
            .into_no_null_iter()
            .collect::<Vec<_>>(),
        &[1.0, 0.0, 1.0, 0.0, 0.0]
    );
    Ok(())
}

#[test]
#[cfg(feature = "log")]
fn test_groups_update_binary_shift_log() -> Result<()> {
    let out = df![
        "a" => [1, 2, 3, 5],
        "b" => [1, 2, 1, 2],
    ]?
    .lazy()
    .groupby([col("b")])
    .agg([col("a") - col("a").shift(1).log(2.0)])
    .sort("b", Default::default())
    .explode([col("a")])
    .collect()?;
    assert_eq!(
        Vec::from(out.column("a")?.f64()?),
        &[None, Some(3.0), None, Some(4.0)]
    );

    Ok(())
}

#[test]
fn test_expand_list() -> Result<()> {
    let out = df![
        "a" => [1, 2],
        "b" => [2, 3],
    ]?
    .lazy()
    .select([cols(["a", "b"]).cumsum(false)])
    .collect()?;

    let expected = df![
        "a" => [1, 3],
        "b" => [2, 5]
    ]?;

    assert!(out.frame_equal(&expected));

    Ok(())
}
