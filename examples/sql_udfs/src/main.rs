use std::collections::HashMap;

use polars::lazy::dsl::{Expr, GetOutput, SpecialEq};
use polars::prelude::*;
use polars::sql::{FunctionOptions, FunctionRegistry, UserDefinedFunction};

struct MyFunctionRegistry {
    functions: HashMap<String, UserDefinedFunction>,
}

impl MyFunctionRegistry {
    fn new(funcs: Vec<UserDefinedFunction>) -> Self {
        let functions = funcs
            .into_iter()
            .map(|f| (f.options.fmt_str.to_string(), f))
            .collect::<HashMap<String, _>>();

        MyFunctionRegistry { functions }
    }
}

impl FunctionRegistry for MyFunctionRegistry {
    fn register(&mut self, name: &str, fun: UserDefinedFunction) -> PolarsResult<()> {
        self.functions.insert(name.to_string(), fun);
        Ok(())
    }

    fn call_udf(&self, name: &str, args: Vec<Expr>) -> PolarsResult<Expr> {
        let fun = self.functions.get(name).unwrap();

        let expr = Expr::AnonymousFunction {
            input: args,
            function: SpecialEq::new(fun.function.clone()),
            output_type: fun.output_type.clone(),
            options: fun.options,
        };

        Ok(expr)
    }

    fn contains(&self, name: &str) -> bool {
        self.functions.contains_key(name)
    }
}

fn main() {
    let my_custom_sum = UserDefinedFunction {
        function: Arc::new(move |series: &mut [Series]| {
            let first = series[0].clone();
            let second = series[1].clone();
            Ok(Some(first + second))
        }),
        output_type: GetOutput::same_type(),
        options: FunctionOptions {
            fmt_str: "my_custom_sum",
            ..Default::default()
        },
    };

    let mut ctx = polars::sql::SQLContext::new()
        .with_function_registry(Arc::new(MyFunctionRegistry::new(vec![my_custom_sum])));
    let df = df! {
        "a" => &[1, 2, 3],
        "b" => &[1, 2, 3],
    }
    .unwrap()
    .lazy();

    ctx.register("foo", df);
    let res = ctx
        .execute("SELECT a, b, my_custom_sum(a, b) as a_plus_b FROM foo")
        .unwrap()
        .collect()
        .unwrap();

    println!("{res}")
}
