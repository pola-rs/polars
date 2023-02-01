use polars_lazy::dsl::{custom_series_flat_udf_fn, Expr};
use polars_lazy::prelude::*;
use polars_lazy::udf::*;
use serde::{Deserialize, Serialize};

use super::*;

#[derive(Serialize, Deserialize)]
pub enum MyUdf {
    PlusOne,
    MulTwo,
}

impl polars_lazy::dsl::SeriesUdf for MyUdf {
    fn call_udf(&self, s: &mut [Series]) -> PolarsResult<Series> {
        match self {
            Self::PlusOne => {
                println!("PlusOne!");
                let series = s.first_mut().ok_or_else(|| {
                    PolarsError::ComputeError("PlusOne must have one series input".into())
                })?;
                Ok(&*series + 1)
            }
            Self::MulTwo => {
                println!("MulTwo!");
                let series = s.first_mut().ok_or_else(|| {
                    PolarsError::ComputeError("MulTwo must have one series input".into())
                })?;
                Ok(&*series * 2)
            }
        }
    }

    fn as_serialize(&self) -> Option<(&str, &dyn ErasedSerialize)> {
        Some(("my_udf", self))
    }
}

impl polars_lazy::dsl::FunctionOutputField for MyUdf {
    fn get_field(
        &self,
        _input_schema: &Schema,
        _cntxt: polars_lazy::dsl::Context,
        fields: &[Field],
    ) -> Field {
        Field::new(fields[0].name(), fields[0].data_type().clone())
    }

    fn as_serialize(&self) -> Option<(&str, &dyn ErasedSerialize)> {
        Some(("my_udf", self))
    }
}

#[test]
fn ser_de_with_registry() -> PolarsResult<()> {
    // Registry

    let registry = UdfSerializeRegistry {
        expr_series_udf: Registry::default().with::<MyUdf>("my_udf", |e| Arc::new(e) as _),
        expr_fn_output_field: Registry::default().with::<MyUdf>("my_udf", |e| Arc::new(e) as _),
        ..Default::default()
    };

    let _ = UDF_DESERIALIZE_REGISTRY.set(registry);

    // Plan

    let df = df![
        "hello" => 0..400,
    ]?
    .lazy()
    .with_column(Expr::Alias(
        Box::new(custom_series_flat_udf_fn(
            Arc::new(MyUdf::MulTwo),
            &[Expr::Column("hello".into())],
        )),
        "mul_two".into(),
    ))
    .with_column(Expr::Alias(
        Box::new(custom_series_flat_udf_fn(
            Arc::new(MyUdf::PlusOne),
            &[Expr::Column("hello".into())],
        )),
        "plus_one".into(),
    ));

    let s = serde_json::to_string(&df.logical_plan).unwrap();
    println!("Logical plan: {s}");

    let obj: LogicalPlan = serde_json::from_str(&s).unwrap();
    println!("Deser: {obj:?}");

    let a = LazyFrame::from(obj);
    let frame = a.collect()?;

    println!("Executed: {frame}");

    todo!()
}
