use std::collections::VecDeque;

use polars::error::{polars_ensure, PolarsResult};
use polars::prelude::{
    ChunkedBuilder, DataType, Field, Int64Type, PrimitiveChunkedBuilder, Schema, SchemaExt,
};
use polars::series::{IntoSeries, Series};
use pyo3_polars::export::polars_plan::polars_plugin_expr_info;
use pyo3_polars::export::polars_plan::prelude::v2::{PolarsPluginExprInfo, StatefulUdfTrait};

struct RollingProduct {
    n: usize,
}
#[derive(Clone)]
struct RollingProductState {
    product: i64,
    values: VecDeque<i64>,
}

impl StatefulUdfTrait for RollingProduct {
    type State = RollingProductState;

    fn to_field(&self, fields: &Schema) -> PolarsResult<Field> {
        assert_eq!(fields.len(), 1);
        let field = fields.iter_fields().next().unwrap();
        polars_ensure!(
            field.dtype() == &DataType::Int64,
            InvalidOperation: "rolling_product can only be performed on i64"
        );
        Ok(field)
    }

    fn initialize(&self, fields: &Schema) -> PolarsResult<Self::State> {
        assert_eq!(fields.len(), 1);
        Ok(RollingProductState {
            product: 1,
            values: VecDeque::with_capacity(self.n.clone()),
        })
    }

    fn insert(&self, state: &mut Self::State, inputs: &[Series]) -> PolarsResult<Option<Series>> {
        assert_eq!(inputs.len(), 1);
        let s = inputs[0].i64()?;
        let mut builder = PrimitiveChunkedBuilder::<Int64Type>::new(s.name().clone(), s.len());
        for v in s.iter() {
            let Some(v) = v else {
                builder.append_null();
                continue;
            };

            if state.values.len() >= self.n {
                state.product /= state.values.pop_front().unwrap();
            }
            state.values.push_back(v);
            state.product *= v;
            builder.append_value(state.product);
        }
        Ok(Some(builder.finish().into_series()))
    }

    fn finalize(&self, _state: &mut Self::State) -> PolarsResult<Option<Series>> {
        Ok(None)
    }

    fn new_empty(&self, state: &Self::State) -> PolarsResult<Self::State> {
        let mut state = state.clone();
        self.reset(&mut state)?;
        Ok(state)
    }

    fn reset(&self, state: &mut Self::State) -> PolarsResult<()> {
        state.product = 1;
        state.values.clear();
        Ok(())
    }
}

// fn insert_on_groups(&self, state: &mut [Self::State], data: Series, groups: )

// None
// Array[idx32, 2]
// Array[idx64, 2]
// List[idx32]
// List[idx64]
// fn evaluate_on_groups(&self, data: &[(Series, Array)]) -> PolarsResult<(Series, Array)> {
// fn insert_on_groups(&self, data: &[(Series, Array)]) -> PolarsResult<Vec<State>> {
// }

#[pyo3::pyfunction]
pub fn rolling_product(n: usize) -> PolarsPluginExprInfo {
    assert!(n > 0);
    polars_plugin_expr_info!("rolling_product", RollingProduct { n }, RollingProduct)
}
