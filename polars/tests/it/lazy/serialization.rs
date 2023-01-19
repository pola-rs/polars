use polars_lazy::dsl::{custom_series_flat_udf_fn, Expr};
use polars_lazy::prelude::*;
use polars_lazy::udf_registry::*;
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
                let series = s.first_mut().ok_or_else(|| {
                    PolarsError::ComputeError("PlusOne must have one series input".into())
                })?;
                Ok(&*series + 1)
            }
            Self::MulTwo => {
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
        expr_series_udf: Registry::new(
            [(
                "my_udf".to_string(),
                (|deser| MyUdf::deserialize(deser).map(|e| Arc::new(e) as _)) as DeserializeFn<_>,
            )]
            .into(),
        ),
        expr_fn_output_field: Registry::new(
            [(
                "my_udf".to_string(),
                (|deser| MyUdf::deserialize(deser).map(|e| Arc::new(e) as _)) as DeserializeFn<_>,
            )]
            .into(),
        ),
        ..Default::default()
    };

    // Plan

    let f = Arc::new(MyUdf::PlusOne);

    let df = df![
        "hello" => 0..400,
    ]?
    .lazy()
    .with_column(custom_series_flat_udf_fn(f, &[Expr::Wildcard]));

    let s = serde_json::to_string(df.logical_plan);
    println!("Logical plan: {s}");

    s
}
