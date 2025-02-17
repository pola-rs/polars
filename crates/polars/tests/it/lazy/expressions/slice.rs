use polars_core::prelude::*;

use super::*;

#[test]
fn test_slice_args() -> PolarsResult<()> {
    let groups: StringChunked = std::iter::repeat("a")
        .take(10)
        .chain(std::iter::repeat("b").take(20))
        .collect();

    let df = df![
        "groups" => groups.into_series(),
        "vals" => 0i32..30
    ]?
    .lazy()
    .group_by_stable([col("groups")])
    .agg([col("vals").slice(lit(0i64), (len() * lit(0.2)).cast(DataType::Int32))])
    .collect()?;

    let out = df.column("vals")?.explode()?;
    let out = out.i32().unwrap();
    assert_eq!(
        out.into_no_null_iter().collect::<Vec<_>>(),
        &[0, 1, 10, 11, 12, 13]
    );

    Ok(())
}
