use polars_core::prelude::*;
use polars_lazy::prelude::IntoLazy;
use polars_plan::dsl::BaseColumnUdf;
use polars_plan::prelude::{FunctionOptions, UserDefinedFunction};
use polars_sql::SQLContext;
use polars_sql::function_registry::FunctionRegistry;

struct MyFunctionRegistry {
    functions: PlHashMap<String, UserDefinedFunction>,
}

impl MyFunctionRegistry {
    fn new(funcs: Vec<UserDefinedFunction>) -> Self {
        let functions = funcs.into_iter().map(|f| (f.name.to_string(), f)).collect();
        MyFunctionRegistry { functions }
    }
}

impl FunctionRegistry for MyFunctionRegistry {
    fn register(&mut self, name: &str, fun: UserDefinedFunction) -> PolarsResult<()> {
        self.functions.insert(name.to_string(), fun);
        Ok(())
    }

    fn get_udf(&self, name: &str) -> PolarsResult<Option<UserDefinedFunction>> {
        Ok(self.functions.get(name).cloned())
    }

    fn contains(&self, name: &str) -> bool {
        self.functions.contains_key(name)
    }
}

#[test]
fn test_udfs() -> PolarsResult<()> {
    let add = UserDefinedFunction::new(
        "add".into(),
        BaseColumnUdf::new(
            move |c: &mut [Column]| {
                let first = c[0].as_materialized_series().clone();
                let second = c[1].as_materialized_series().clone();
                (first + second).map(Column::from)
            },
            |_: &Schema, fs: &[Field]| Ok(fs[0].clone()),
        ),
    );

    let mut ctx =
        SQLContext::new().with_function_registry(Arc::new(MyFunctionRegistry::new(vec![add])));

    let df = df! {
        "a" => &[1, 2, 3],
        "b" => &[1, 2, 3],
    }?
    .lazy();
    ctx.register("foo", df);

    let expected = df! { "ab" => &[2, 4, 6] }?;

    // derived table (subquery in FROM)
    let res = ctx
        .execute("SELECT * FROM (SELECT ADD(a, b) AS ab FROM foo)")?
        .collect()?;
    assert!(expected.equals_missing(&res));

    // common table expression
    let res = ctx
        .execute("WITH cte AS (SELECT ADD(a, b) AS ab FROM foo) SELECT * FROM cte")?
        .collect()?;
    assert!(expected.equals_missing(&res));

    // set operation
    let res = ctx
        .execute("SELECT ADD(a, b) AS ab FROM foo UNION ALL SELECT a AS ab FROM foo")?
        .collect()?
        .sort(["ab"], Default::default())?;
    assert!(df! { "ab" => &[1, 2, 2, 3, 4, 6] }?.equals_missing(&res));

    // scalar subquery in the SELECT list
    let res = ctx
        .execute("SELECT (SELECT MAX(ADD(a, b)) FROM foo) AS ab FROM foo LIMIT 1")?
        .collect()?;
    assert!(df! { "ab" => &[6] }?.equals_missing(&res));

    // subquery in the WHERE clause
    let res = ctx
        .execute("SELECT a FROM foo WHERE a IN (SELECT ADD(a, b) AS ab FROM foo)")?
        .collect()?;
    assert!(df! { "a" => &[2] }?.equals_missing(&res));

    Ok(())
}

#[test]
fn test_group_by_aggregate_udfs() -> PolarsResult<()> {
    let mut agg_plugin = UserDefinedFunction::new(
        "agg_plugin".into(),
        BaseColumnUdf::new(
            move |c: &mut [Column]| {
                let series = c[0].as_materialized_series();
                let scalar = series.sum_reduce()?;
                Ok(Column::new_scalar(series.name().clone(), scalar, 1))
            },
            |_: &Schema, fs: &[Field]| {
                polars_ensure!(fs.len() == 1, SchemaMismatch: "expected one argument");
                Ok(fs[0].clone())
            },
        ),
    );
    agg_plugin.options = FunctionOptions::aggregation();

    let mut ctx = SQLContext::new()
        .with_function_registry(Arc::new(MyFunctionRegistry::new(vec![agg_plugin])));

    let df = df! {
        "g" => &["x", "x", "y"],
        "v" => &[1i64, 2, 3],
    }?
    .lazy();
    ctx.register("foo", df);

    let res = ctx
        .execute("SELECT g, agg_plugin(v) AS total FROM foo GROUP BY g")?
        .collect()?
        .sort(["g"], Default::default())?;
    let expected = df! {
        "g" => &["x", "y"],
        "total" => &[3i64, 3],
    }?;
    assert!(expected.equals_missing(&res));

    Ok(())
}
