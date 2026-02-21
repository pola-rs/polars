use polars::prelude::*;

#[test]
fn test_fused_select_slice_last() -> PolarsResult<()> {
    let df = df![
        "a" => [1, 2, 3],
        "b" => [4, 5, 6],
    ]?;

    let q = df.lazy().select([col("a").last(), col("b").last()]);

    let plan = q.explain(true)?;
    println!("{}", plan);

    // Verify that SLICE is present in the plan
    assert!(plan.contains("SLICE[offset: -1, len: 1]"));

    let out = q.collect()?;
    let expected = df![
        "a" => [3],
        "b" => [6],
    ]?;
    assert!(out.equals(&expected));

    Ok(())
}

#[test]
fn test_fused_select_slice_first() -> PolarsResult<()> {
    let df = df![
        "a" => [1, 2, 3],
        "b" => [4, 5, 6],
    ]?;

    let q = df.lazy().select([col("a").first(), col("b").first()]);

    let plan = q.explain(true)?;
    println!("{}", plan);

    assert!(plan.contains("SLICE[offset: 0, len: 1]"));

    let out = q.collect()?;
    let expected = df![
        "a" => [1],
        "b" => [4],
    ]?;
    assert!(out.equals(&expected));

    Ok(())
}

#[test]
fn test_fused_select_slice_head() -> PolarsResult<()> {
    let df = df![
        "a" => [1, 2, 3, 4, 5],
        "b" => [10, 20, 30, 40, 50],
    ]?;

    let q = df.lazy().select([col("a").head(Some(2)), col("b").head(Some(2))]);

    let plan = q.explain(true)?;
    println!("{}", plan);

    assert!(plan.contains("SLICE[offset: 0, len: 2]"));

    let out = q.collect()?;
    let expected = df![
        "a" => [1, 2],
        "b" => [10, 20],
    ]?;
    assert!(out.equals(&expected));

    Ok(())
}
