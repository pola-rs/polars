use polars_core::prelude::*;
use polars_lazy::prelude::*;
use polars_sql::*;

#[test]
fn iss_14217() -> PolarsResult<()> {
    let mut ctx = SQLContext::new();
    
    let df1 = df! {
        "a" => [1, 2, 3],
        "b" => [4, 5, 6],
    }
    .unwrap();

    let df2 = df! {
        "b" => [4, 5, 6],
        "c" => [7, 8, 9]
    }
    .unwrap();

    let df3 = df! {
        "c" => [7, 8, 9],
        "d" => [10, 11, 12],
    }
    .unwrap();

    ctx.register("df1", df1.lazy());
    ctx.register("df2", df2.lazy());
    ctx.register("df3", df3.lazy());
    
    let sql = r#"
        SELECT * FROM df1
        INNER JOIN df2 ON df1.b = df2.b
        INNER JOIN df3 ON df2.c = df3.c
    "#;

    let result = ctx.execute(sql).unwrap();
    
    let expected_df = df! {
        "a" => [1, 2, 3],
        "b" => [4, 5, 6],
        "c" => [7, 8, 9],
        "d" => [10, 11, 12],
    }
    .unwrap();
    
    assert!(result.collect()?.equals(&expected_df));

    Ok(())
}
