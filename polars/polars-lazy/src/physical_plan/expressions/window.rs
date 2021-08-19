use crate::logical_plan::Context;
use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;
use polars_core::frame::groupby::{GroupBy, GroupTuples};
use polars_core::frame::hash_join::private_left_join_multiple_keys;
use polars_core::prelude::*;
use std::sync::Arc;

pub struct WindowExpr {
    /// the root column that the Function will be applied on.
    /// This will be used to create a smaller DataFrame to prevent taking unneeded columns by index
    pub(crate) group_by: Vec<Arc<dyn PhysicalExpr>>,
    pub(crate) apply_columns: Vec<Arc<String>>,
    pub(crate) out_name: Option<Arc<String>>,
    /// A function Expr. i.e. Mean, Median, Max, etc.
    pub(crate) function: Expr,
    pub(crate) phys_function: Arc<dyn PhysicalExpr>,
    pub(crate) options: WindowOptions,
}

impl PhysicalExpr for WindowExpr {
    // Note: this was first implemented with expression evaluation but this performed really bad.
    // Therefore we choose the groupby -> apply -> self join approach

    // This first cached the groupby and the join tuples, but rayon under a mutex leads to deadlocks:
    // https://github.com/rayon-rs/rayon/issues/592
    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> Result<Series> {
        // This method does the following:
        // 1. determine groupby tuples based on the group_column
        // 2. apply an aggregation function
        // 3. join the results back to the original dataframe
        //    this stores all group values on the original df size
        // 4. select the final column and return
        let groupby_columns = self
            .group_by
            .iter()
            .map(|e| e.evaluate(df, state))
            .collect::<Result<Vec<_>>>()?;

        let mut gb = df.groupby_with_series(groupby_columns.clone(), true)?;
        let mut groups = std::mem::take(gb.get_groups_mut());

        // if we flatten this column we need to make sure the groups are sorted.
        if self.options.explode {
            groups.sort_unstable_by_key(|t| t.0);
        }

        // 2. create GroupBy object and apply aggregation
        let apply_columns = self.apply_columns.iter().map(|s| s.as_str()).collect();
        let gb = GroupBy::new(df, groupby_columns.clone(), groups, Some(apply_columns));

        let out = match self.phys_function.as_agg_expr() {
            // this branch catches all aggregation expressions
            // this is list, sum, etc. but also binary functions that are evaluated on groups
            Ok(agg_expr) => match agg_expr.aggregate(df, gb.get_groups(), state)? {
                Some(mut s) => {
                    s.rename(&self.apply_columns[0]);
                    let mut cols = gb.keys();
                    cols.push(s);
                    Ok(DataFrame::new_no_checks(cols))
                }
                None => Err(PolarsError::Other(
                    "aggregation did not return a column".into(),
                )),
            },
            // if we have a function that is not a final aggregation, we can always evaluate the
            // function in groupby context and aggregate the result to a list
            Err(_) => {
                let mut acc = self
                    .phys_function
                    .evaluate_on_groups(df, gb.get_groups(), state)?;
                let mut cols = gb.keys();
                let out = acc.aggregated().into_owned();
                cols.push(out);
                Ok(DataFrame::new_no_checks(cols))
            }
        }?;

        // 3. get the join tuples and use them to take the new Series
        let out_column = out.select_at_idx(out.width() - 1).unwrap();
        if self.options.explode {
            let mut out = out_column.clone();
            if let Some(name) = &self.out_name {
                out.rename(name.as_str());
            }
            return Ok(out);
        }

        let opt_join_tuples = if groupby_columns.len() == 1 {
            // group key from right column
            let right = out.select_at_idx(0).unwrap();
            groupby_columns[0].hash_join_left(right)
        } else {
            let df_right = DataFrame::new_no_checks(out.get_columns()[..out.width() - 1].to_vec());
            let df_left = DataFrame::new_no_checks(groupby_columns);
            private_left_join_multiple_keys(&df_left, &df_right)
        };

        let mut iter = opt_join_tuples
            .iter()
            .map(|(_left, right)| right.map(|i| i as usize));

        let mut out = unsafe { out_column.take_opt_iter_unchecked(&mut iter) };
        if let Some(name) = &self.out_name {
            out.rename(name.as_str());
        }
        Ok(out)
    }

    fn to_field(&self, input_schema: &Schema) -> Result<Field> {
        self.function.to_field(input_schema, Context::Default)
    }

    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups<'a>(
        &self,
        _df: &DataFrame,
        _groups: &'a GroupTuples,
        _state: &ExecutionState,
    ) -> Result<AggregationContext<'a>> {
        Err(PolarsError::InvalidOperation(
            "window expression not allowed in aggregation".into(),
        ))
    }

    fn as_expression(&self) -> &Expr {
        todo!()
    }
}
