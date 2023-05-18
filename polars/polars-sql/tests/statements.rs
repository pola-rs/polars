use polars_core::prelude::*;
use polars_lazy::prelude::*;
use polars_sql::*;

fn create_ctx() -> SQLContext {
    let a = Series::new("a", (1..10i64).map(|i| i / 100).collect::<Vec<_>>());
    let b = Series::new("b", 1..10i64);
    let df = DataFrame::new(vec![a, b]).unwrap().lazy();
    let mut ctx = SQLContext::new();
    ctx.register("df", df);
    ctx
}

#[test]
fn trailing_commas_allowed() {
    let mut ctx = create_ctx();
    let sql = r#"
    SELECT
        a,
        b,
    FROM df
    "#;
    let actual = ctx.execute(sql);
    assert!(actual.is_ok());
}
