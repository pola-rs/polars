use polars_core::chunked_array::cast::CastOptions;
use polars_core::error::ErrString;
use polars_core::frame::column::ScalarColumn;
use polars_core::prelude::*;
use polars_plan::constants::PL_STATE_NAME;

use super::*;
use crate::expressions::{AggState, AggregationContext, PartitionedAggregation, PhysicalExpr};

pub struct FoldvExpr {
    pub(crate) state_expr: Arc<dyn PhysicalExpr>,
    pub(crate) required_exprs: Vec<Arc<dyn PhysicalExpr>>,
}

impl PhysicalExpr for FoldvExpr {
    fn as_expression(&self) -> Option<&Expr> {
        None
    }

    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Column> {
        let columns = self
            .required_exprs
            .iter()
            .map(|e| e.evaluate(df, state))
            .collect::<PolarsResult<Vec<_>>>()?;
        let mut temp_cols = Vec::new();
        for col in &columns {
            temp_cols.push(Column::Scalar(ScalarColumn::new(
                col.name().clone(),
                Scalar::new(col.dtype().clone(), AnyValue::Null),
                1,
            )));
        }
        let mut state_val = AnyValue::Int64(0);
        temp_cols.push(Column::Scalar(ScalarColumn::new(
            PL_STATE_NAME.clone(),
            Scalar::new(state_val.dtype().clone(), AnyValue::Null),
            1,
        )));
        let mut temp_cols = Some(temp_cols);
        let mut states = Column::new_empty(PL_STATE_NAME.clone(), &state_val.dtype());
        for i in 0..df.height() {
            let mut use_cols = temp_cols.take().unwrap();
            for (col_idx, col) in columns.iter().enumerate() {
                let col_val = unsafe { col.get_unchecked(i).into_static() };
                let temp_col = &mut use_cols[col_idx];
                temp_col
                    .as_scalar_column_mut()
                    .unwrap()
                    .map_scalar(|x| x.with_value(col_val.clone()));
            }
            let state_col = use_cols.last_mut().unwrap();
            state_col
                .as_scalar_column_mut()
                .unwrap()
                .map_scalar(|x| x.with_value(state_val.clone()));

            let row_df = DataFrame::new(use_cols)?;
            let res = self.state_expr.evaluate(&row_df, state);
            let res = res?;
            state_val = res
                .as_scalar_column()
                .unwrap()
                .scalar()
                .clone()
                .into_value();
            states.append_owned(res);
            let mut use_cols = row_df.take_columns();
            temp_cols = Some(use_cols);
        }
        // let mut column = self.input.evaluate(df, state)?;
        // let s = column.into_materialized_series();
        // let a = s.i64()?;
        // let new_a = a
        //     .iter()
        //     .map(|opt_x| opt_x.map(|x| x + 1))
        //     .collect::<ChunkedArray<Int64Type>>();

        // Ok(new_a.into_column())
        Ok(states)
    }

    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupPositions,
        state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        return Err(PolarsError::InvalidOperation(ErrString::new_static(
            "sorry",
        )));
    }

    fn to_field(&self, input_schema: &Schema) -> PolarsResult<Field> {
        self.state_expr.to_field(input_schema)
    }

    fn is_scalar(&self) -> bool {
        self.state_expr.is_scalar()
    }

    fn as_partitioned_aggregator(&self) -> Option<&dyn PartitionedAggregation> {
        None
    }
}
