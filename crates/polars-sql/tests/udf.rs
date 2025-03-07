use polars_core::prelude::*;
use polars_lazy::prelude::IntoLazy;
use polars_plan::prelude::{GetOutput, UserDefinedFunction};
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
    let my_custom_sum = UserDefinedFunction::new(
        "my_custom_sum".into(),
        GetOutput::map_dtypes(|dtypes| {
            // UDF is responsible for schema validation
            let Ok([first, second]) = <[&DataType; 2]>::try_from(dtypes) else {
                polars_bail!(SchemaMismatch: "expected two arguments")
            };
            if first != second {
                polars_bail!(SchemaMismatch: "mismatched types")
            }
            Ok(first.clone())
        }),
        move |c: &mut [Column]| {
            let first = c[0].as_materialized_series().clone();
            let second = c[1].as_materialized_series().clone();
            (first + second).map(Column::from).map(Some)
        },
    );

    let mut ctx = SQLContext::new()
        .with_function_registry(Arc::new(MyFunctionRegistry::new(vec![my_custom_sum])));

    let df = df! {
        "a" => &[1, 2, 3],
        "b" => &[1, 2, 3],
        "c" => &["a", "b", "c"]
    }
    .unwrap()
    .lazy();

    ctx.register("foo", df);
    let res = ctx.execute("SELECT a, b, my_custom_sum(a, b) FROM foo");
    assert!(res.is_ok());

    // schema is invalid so it will fail
    assert!(matches!(
        ctx.execute("SELECT a, b, my_custom_sum(c) as invalid FROM foo"),
        Err(PolarsError::SchemaMismatch(_))
    ));

    // create a new UDF to be registered on the context
    let my_custom_divide = UserDefinedFunction::new(
        "my_custom_divide".into(),
        GetOutput::map_dtypes(|dtypes| {
            // UDF is responsible for schema validation
            let Ok([first, second]) = <[&DataType; 2]>::try_from(dtypes) else {
                polars_bail!(SchemaMismatch: "expected two arguments")
            };
            if first != second {
                polars_bail!(SchemaMismatch: "mismatched types")
            }
            Ok(first.clone())
        }),
        move |c: &mut [Column]| {
            let first = c[0].as_materialized_series().clone();
            let second = c[1].as_materialized_series().clone();
            (first / second).map(Column::from).map(Some)
        },
    );

    // register a new UDF on an existing context
    ctx.registry_mut().register("my_div", my_custom_divide)?;

    // execute the query
    let res = ctx
        .execute("SELECT a, b, my_div(a, b) as my_div FROM foo")?
        .collect()?;
    let expected = df! {
        "a" => &[1, 2, 3],
        "b" => &[1, 2, 3],
        "my_div" => &[1, 1, 1]
    }?;
    assert!(expected.equals_missing(&res));

    Ok(())
}
