#[cfg(feature = "csv")]
use polars_core::prelude::*;
#[cfg(feature = "csv")]
use polars_sql::*;

#[test]
#[cfg(feature = "csv")]
fn iss_8395() -> PolarsResult<()> {
    use polars_core::series::Series;

    let mut context = SQLContext::new();
    let sql = r#"
    with foods as (
        SELECT *
        FROM read_csv('../../examples/datasets/foods1.csv')
    )
    select * from foods where category IN ('vegetables', 'seafood')"#;
    let res = context.execute(sql)?;
    let df = res.collect()?;

    // assert that the df only contains [vegetables, seafood]
    let s = df.column("category")?.unique()?.sort(false, false);
    let expected = Series::new("category", &["seafood", "vegetables"]);
    assert!(s.equals(&expected));
    Ok(())
}
