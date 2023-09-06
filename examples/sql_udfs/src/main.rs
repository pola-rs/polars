use std::collections::HashMap;

use polars::lazy::dsl::{Expr, GetOutput, SpecialEq};
use polars::prelude::*;
use polars::sql::{FunctionOptions, FunctionRegistry, UserDefinedFunction};

struct MyFunctionRegistry {
    functions: HashMap<String, MyUdf>,
}

#[derive(Clone)]
struct MyUdf {
    name: &'static str,
    input_fields: Vec<Field>,
    output_field: Field,
    function_impl: Arc<dyn SeriesUdf>,
}

impl MyUdf {
    pub fn new(
        name: &'static str,
        fields: Vec<Field>,
        output_field: Field,
        fun: impl SeriesUdf + 'static,
    ) -> Self {
        MyUdf {
            name,
            input_fields: fields,
            output_field,
            function_impl: Arc::new(fun),
        }
    }
}

impl UserDefinedFunction for MyUdf {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn call_udf(&self, s: &mut [Series]) -> PolarsResult<Option<Series>> {
        if s.len() != self.input_fields.len() {
            polars_bail!(
                ComputeError:
                "expected {} arguments, got {}",
                self.input_fields.len(),
                s.len()
            )
        };

        self.function_impl.as_ref().call_udf(s)
    }

    fn output_type(&self) -> GetOutput {
        GetOutput::from_type(self.output_field.dtype.clone())
    }

    fn options(&self) -> FunctionOptions {
        FunctionOptions {
            fmt_str: self.name,
            ..FunctionOptions::default()
        }
    }
}

impl MyFunctionRegistry {
    fn new(funcs: Vec<MyUdf>) -> Self {
        let functions = funcs.into_iter().map(|f| (f.name.to_string(), f)).collect();
        MyFunctionRegistry { functions }
    }
}

impl FunctionRegistry for MyFunctionRegistry {
    fn register(&mut self, name: &str, fun: &dyn UserDefinedFunction) -> PolarsResult<()> {
        if let Some(f) = fun.as_any().downcast_ref::<MyUdf>() {
            self.functions.insert(name.to_string(), f.clone());
            Ok(())
        } else {
            polars_bail!(ComputeError: "unexpected Udf")
        }
    }

    fn call(&self, name: &str, args: Vec<Expr>) -> PolarsResult<Expr> {
        if let Some(f) = self.functions.get(name) {
            let schema = Schema::from_iter(f.input_fields.clone());

            if args
                .iter()
                .map(|e| e.to_field(&schema, polars::sql::Context::Default))
                .collect::<PolarsResult<Vec<_>>>()
                .is_err()
            {
                polars_bail!(InvalidOperation: "unexpected field in UDF \nexpected: {:?}\n received {:?}", f.input_fields, args)
            };

            let func: SpecialEq<Arc<dyn SeriesUdf>> = SpecialEq::new(f.function_impl.clone());
            let expr = Expr::AnonymousFunction {
                input: args,
                function: func,
                output_type: f.output_type(),
                options: f.options(),
            };
            Ok(expr)
        } else {
            todo!()
        }
    }

    fn contains(&self, name: &str) -> bool {
        self.functions.contains_key(name)
    }
}

fn main() -> PolarsResult<()> {
    let my_custom_sum = MyUdf::new(
        "my_custom_sum",
        vec![
            Field::new("a", DataType::Int32),
            Field::new("b", DataType::Int32),
        ],
        Field::new("a_plus_b", DataType::Int32),
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
    let ok_res = ctx.execute("SELECT a, b, my_custom_sum(a, b) as a_plus_b FROM foo");
    assert!(ok_res.is_ok());
    println!("{:?}", ok_res.unwrap().collect()?);
    assert!(ctx
        .execute("SELECT a, b, my_custom_sum(c) as invalid FROM foo")
        .is_err());

    Ok(())
}
