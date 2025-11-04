use polars::error::{polars_bail, PolarsResult};
use polars::prelude::{AnyValue, ArgAgg, Field, PlSmallStr, Scalar, Schema, SchemaExt};
use polars::series::Series;
use pyo3_polars::export::polars_plan::polars_plugin_expr_info;
use pyo3_polars::export::polars_plan::prelude::v2::{PolarsPluginExprInfo, StatefulUdfTrait};

struct MinBy;
#[derive(Clone)]
struct MinByState {
    name: PlSmallStr,
    value: Scalar,
    by: Scalar,
}

impl StatefulUdfTrait for MinBy {
    type State = MinByState;

    fn to_field(&self, fields: &Schema) -> PolarsResult<Field> {
        assert_eq!(fields.len(), 2);
        Ok(fields.iter_fields().next().unwrap())
    }

    fn initialize(&self, fields: &Schema) -> PolarsResult<Self::State> {
        assert_eq!(fields.len(), 2);
        let (name, dtype) = fields.get_at_index(0).unwrap();
        let name = name.clone();
        let value = Scalar::null(dtype.clone());
        let by = Scalar::null(fields.get_at_index(1).unwrap().1.clone());
        Ok(MinByState { name, value, by })
    }

    fn insert(&self, state: &mut Self::State, inputs: &[Series]) -> PolarsResult<Option<Series>> {
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

        if let Some(arg_min) = inputs[1].arg_min() {
            let new_by = inputs[1].get(arg_min)?;

            if state.by.is_null() || (!new_by.is_null() && &new_by < state.by.value()) {
                *state.value.any_value_mut() = inputs[0].get(arg_min)?.into_static();
                *state.by.any_value_mut() = new_by.into_static();
            }
        }
        Ok(None)
    }

    fn finalize(&self, state: &mut Self::State) -> PolarsResult<Option<Series>> {
        Ok(Some(state.value.clone().into_series(state.name.clone())))
    }

    fn new_empty(&self, state: &Self::State) -> PolarsResult<Self::State> {
        let mut state = state.clone();
        self.reset(&mut state)?;
        Ok(state)
    }

    fn combine(&self, state: &mut Self::State, other: &Self::State) -> PolarsResult<()> {
        if state.by.is_null() || (!other.by.is_null() && other.by.value() < state.by.value()) {
            state.value = other.value.clone();
            state.by = other.by.clone();
        }
        Ok(())
    }

    fn reset(&self, state: &mut Self::State) -> PolarsResult<()> {
        *state.value.any_value_mut() = AnyValue::Null;
        *state.by.any_value_mut() = AnyValue::Null;
        Ok(())
    }
}

#[pyo3::pyfunction]
pub fn min_by() -> PolarsPluginExprInfo {
    polars_plugin_expr_info!("min_by", MinBy, MinBy)
}
