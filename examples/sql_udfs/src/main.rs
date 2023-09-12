use std::collections::HashMap;

use polars::lazy::dsl::udf::UserDefinedFunction;
use polars::lazy::dsl::GetOutput;
use polars::prelude::*;
use polars::sql::FunctionRegistry;

struct MyFunctionRegistry {
    functions: HashMap<String, UserDefinedFunction>,
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

fn main() -> PolarsResult<()> {
    let my_custom_sum = UserDefinedFunction::new(
        "my_custom_sum",
        vec![
            Field::new("a", DataType::Int32),
            Field::new("b", DataType::Int32),
        ],
        GetOutput::same_type(),
        move |s: &mut [Series]| {
            let first = s[0].clone();
            let second = s[1].clone();
            Ok(Some(first + second))
        },
    );

    let mut ctx = polars::sql::SQLContext::new()
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
    println!("{:?}", res.unwrap().collect()?);

    // schema is invalid so it will fail
    assert!(ctx
        .execute("SELECT a, b, my_custom_sum(c) as invalid FROM foo")
        .is_err());

    // create a new UDF to be registered on the context
    let my_custom_divide = UserDefinedFunction::new(
        "my_custom_divide",
        vec![
            Field::new("a", DataType::Int32),
            Field::new("b", DataType::Int32),
        ],
        GetOutput::same_type(),
        move |s: &mut [Series]| {
            let first = s[0].clone();
            let second = s[1].clone();
            Ok(Some(first / second))
        },
    );

    // register a new UDF on an existing context
    ctx.registry_mut().register("my_div", my_custom_divide)?;

    // execute the query
    let res = ctx.execute("SELECT a, b, my_div(a, b) as my_div FROM foo")?;

    println!("{:?}", res.collect()?);

    Ok(())
}
