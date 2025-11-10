use polars::error::{polars_bail, PolarsResult};
use polars::prelude::{AnyValue, ArgAgg, Field, PlSmallStr, Scalar, Schema, SchemaExt};
use polars::series::Series;
use pyo3_polars::polars_plugin_expr_info;
use pyo3_polars::v1::{self, PolarsPluginExprInfo};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct MinBy;
#[derive(Serialize, Deserialize, Default, Clone)]
struct MinByState {
    name: PlSmallStr,
    value: Scalar,
    by: Scalar,
}

impl v1::map_reduce::PolarsMapReducePlugin for MinBy {
    type State = MinByState;

    fn to_field(&self, fields: &Schema) -> PolarsResult<Field> {
        assert_eq!(fields.len(), 2);
        Ok(fields.iter_fields().next().unwrap())
    }

    fn map(&self, inputs: &[Series]) -> PolarsResult<Self::State> {
        assert_eq!(inputs.len(), 2);

        let mut inputs = inputs.to_vec();

        // Broadcasting behavior.
        if inputs[0].len() != inputs[1].len() {
            if inputs[0].len() == 1 {
                inputs[0] = inputs[0].new_from_index(0, inputs[1].len());
            } else if inputs[1].len() == 1 {
                inputs[1] = inputs[1].new_from_index(0, inputs[0].len());
            } else {
                polars_bail!(length_mismatch = "min_by", inputs[0].len(), inputs[1].len());
            }
        }

        let (value, by) = if let Some(arg_min) = inputs[1].arg_min() {
            (
                inputs[0].get(arg_min)?.into_static(),
                inputs[1].get(arg_min)?.into_static(),
            )
        } else {
            (AnyValue::Null, AnyValue::Null)
        };

        let value = Scalar::new(inputs[0].dtype().clone(), value);
        let by = Scalar::new(inputs[1].dtype().clone(), by);

        Ok(MinByState {
            name: inputs[0].name().clone(),
            value,
            by,
        })
    }

    fn reduce(&self, left: &Self::State, right: &Self::State) -> PolarsResult<Self::State> {
        Ok(
            if left.by.is_null() || (!right.by.is_null() && right.by.value() < left.by.value()) {
                right.clone()
            } else {
                left.clone()
            },
        )
    }

    fn finalize(&self, state: Self::State) -> PolarsResult<Series> {
        Ok(state.value.into_series(state.name))
    }
}

#[pyo3::pyfunction]
pub fn min_by() -> PolarsPluginExprInfo {
    polars_plugin_expr_info!(
        "min_by",
        v1::map_reduce::Plugin(MinBy),
        v1::map_reduce::Plugin<MinBy>
    )
}
