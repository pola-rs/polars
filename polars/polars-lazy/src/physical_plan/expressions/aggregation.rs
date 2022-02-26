use crate::physical_plan::state::ExecutionState;
use crate::physical_plan::PhysicalAggregation;
use crate::prelude::*;
use polars_arrow::export::arrow::{array::*, compute::concatenate::concatenate};
use polars_arrow::prelude::QuantileInterpolOptions;
use polars_core::frame::groupby::{GroupByMethod, GroupsProxy};
use polars_core::{prelude::*, POOL};
use std::borrow::Cow;
use std::sync::Arc;

pub(crate) struct AggregationExpr {
    pub(crate) expr: Arc<dyn PhysicalExpr>,
    pub(crate) agg_type: GroupByMethod,
}

impl AggregationExpr {
    pub fn new(expr: Arc<dyn PhysicalExpr>, agg_type: GroupByMethod) -> Self {
        Self { expr, agg_type }
    }
}

impl PhysicalExpr for AggregationExpr {
    fn as_expression(&self) -> &Expr {
        unimplemented!()
    }

    fn evaluate(&self, _df: &DataFrame, _state: &ExecutionState) -> Result<Series> {
        unimplemented!()
    }
    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupsProxy,
        state: &ExecutionState,
    ) -> Result<AggregationContext<'a>> {
        let out = self.aggregate(df, groups, state)?.ok_or_else(|| {
            PolarsError::ComputeError("Aggregation did not return a Series".into())
        })?;
        Ok(AggregationContext::new(out, Cow::Borrowed(groups), true))
    }

    fn to_field(&self, input_schema: &Schema) -> Result<Field> {
        self.expr.to_field(input_schema)
    }

    fn as_agg_expr(&self) -> Result<&dyn PhysicalAggregation> {
        Ok(self)
    }
}

fn rename_option_series(opt: Option<Series>, name: &str) -> Option<Series> {
    opt.map(|mut s| {
        s.rename(name);
        s
    })
}

impl PhysicalAggregation for AggregationExpr {
    fn aggregate(
        &self,
        df: &DataFrame,
        groups: &GroupsProxy,
        state: &ExecutionState,
    ) -> Result<Option<Series>> {
        let mut ac = self.expr.evaluate_on_groups(df, groups, state)?;
        // don't change names by aggregations as is done in polars-core
        let keep_name = ac.series().name().to_string();

        match self.agg_type {
            GroupByMethod::Min => {
                let agg_s = ac.flat_naive().into_owned().agg_min(ac.groups());
                Ok(rename_option_series(agg_s, &keep_name))
            }
            GroupByMethod::Max => {
                let agg_s = ac.flat_naive().into_owned().agg_max(ac.groups());
                Ok(rename_option_series(agg_s, &keep_name))
            }
            GroupByMethod::Median => {
                let agg_s = ac.flat_naive().into_owned().agg_median(ac.groups());
                Ok(rename_option_series(agg_s, &keep_name))
            }
            GroupByMethod::Mean => {
                let agg_s = ac.flat_naive().into_owned().agg_mean(ac.groups());
                Ok(rename_option_series(agg_s, &keep_name))
            }
            GroupByMethod::Sum => {
                let agg_s = ac.flat_naive().into_owned().agg_sum(ac.groups());
                Ok(rename_option_series(agg_s, &keep_name))
            }
            GroupByMethod::Count => {
                let mut ca = ac.groups.group_count();
                ca.rename(&keep_name);
                Ok(Some(ca.into_series()))
            }
            GroupByMethod::First => {
                let mut agg_s = ac.flat_naive().into_owned().agg_first(ac.groups());
                agg_s.rename(&keep_name);
                Ok(Some(agg_s))
            }
            GroupByMethod::Last => {
                let mut agg_s = ac.flat_naive().into_owned().agg_last(ac.groups());
                agg_s.rename(&keep_name);
                Ok(Some(agg_s))
            }
            GroupByMethod::NUnique => {
                let opt_agg = ac.flat_naive().into_owned().agg_n_unique(ac.groups());
                let opt_agg = opt_agg.map(|mut agg| {
                    agg.rename(&keep_name);
                    agg.into_series()
                });
                Ok(opt_agg)
            }
            GroupByMethod::List => {
                let agg = ac.aggregated();
                Ok(rename_option_series(Some(agg), &keep_name))
            }
            GroupByMethod::Groups => {
                let mut column: ListChunked = ac.groups().as_list_chunked();
                column.rename(&keep_name);
                Ok(Some(column.into_series()))
            }
            GroupByMethod::Std => {
                let agg_s = ac.flat_naive().into_owned().agg_std(ac.groups());
                Ok(rename_option_series(agg_s, &keep_name))
            }
            GroupByMethod::Var => {
                let agg_s = ac.flat_naive().into_owned().agg_var(ac.groups());
                Ok(rename_option_series(agg_s, &keep_name))
            }
            GroupByMethod::Quantile(_, _) => {
                // implemented explicitly in AggQuantile struct
                unimplemented!()
            }
        }
    }

    fn evaluate_partitioned(
        &self,
        df: &DataFrame,
        groups: &GroupsProxy,
        state: &ExecutionState,
    ) -> Result<Option<Vec<Series>>> {
        match self.agg_type {
            GroupByMethod::Mean => {
                let series = self.expr.evaluate(df, state)?;
                let mut new_name = series.name().to_string();
                let agg_s = series.agg_sum(groups);

                // If the aggregation is successful,
                // we also count the valid values (len - null count)
                // this is needed to compute the final mean.
                if let Some(agg_s) = agg_s {
                    // we expect f64 from mean, so we already cast
                    let mut agg_s = agg_s.cast(&DataType::Float64)?;
                    agg_s.rename(&new_name);
                    new_name.push_str("__POLARS_MEAN_COUNT");
                    let mut count_s = series.agg_valid_count(groups).unwrap();
                    count_s.rename(&new_name);
                    Ok(Some(vec![agg_s, count_s]))
                } else {
                    Ok(None)
                }
            }
            GroupByMethod::List => {
                let series = self.expr.evaluate(df, state)?;
                let new_name = series.name();
                let opt_agg = series.agg_list(groups);
                Ok(opt_agg.map(|mut s| {
                    s.rename(new_name);
                    vec![s]
                }))
            }
            _ => PhysicalAggregation::aggregate(self, df, groups, state)
                .map(|opt| opt.map(|s| vec![s])),
        }
    }

    fn evaluate_partitioned_final(
        &self,
        final_df: &DataFrame,
        groups: &GroupsProxy,
        state: &ExecutionState,
    ) -> Result<Option<Series>> {
        match self.agg_type {
            GroupByMethod::Mean => {
                let series = self.expr.evaluate(final_df, state)?;
                let count_name = format!("{}__POLARS_MEAN_COUNT", series.name());
                let new_name = series.name().to_string();
                let count = final_df.column(&count_name).unwrap();

                let (agg_count, agg_s) =
                    POOL.join(|| count.agg_sum(groups), || series.agg_sum(groups));
                let agg_s = agg_s.map(|agg_s| &agg_s / &agg_count.unwrap());
                Ok(rename_option_series(agg_s, &new_name))
            }
            GroupByMethod::List => {
                // the groups are scattered over multiple groups/sub dataframes.
                // we now must collect them into a single group
                let series = self.expr.evaluate(final_df, state)?;
                let ca = series.list().unwrap();
                let new_name = series.name().to_string();

                let mut values = Vec::with_capacity(groups.len());
                let mut can_fast_explode = true;

                let mut offsets = Vec::<i64>::with_capacity(groups.len() + 1);
                let mut length_so_far = 0i64;
                offsets.push(length_so_far);

                for (_, idx) in groups.idx_ref() {
                    let ca = unsafe {
                        // Safety
                        // The indexes of the groupby operation are never out of bounds
                        ca.take_unchecked(idx.into())
                    };
                    let s = ca.explode()?;
                    length_so_far += s.len() as i64;
                    offsets.push(length_so_far);
                    values.push(s.chunks()[0].clone());

                    if s.len() == 0 {
                        can_fast_explode = false;
                    }
                }
                let vals = values.iter().map(|arr| &**arr).collect::<Vec<_>>();
                let values: ArrayRef = concatenate(&vals).unwrap().into();

                let data_type = ListArray::<i64>::default_datatype(values.data_type().clone());
                let arr = Arc::new(ListArray::<i64>::from_data(
                    data_type,
                    offsets.into(),
                    values,
                    None,
                )) as ArrayRef;
                let mut ca = ListChunked::from_chunks(&new_name, vec![arr]);
                if can_fast_explode {
                    ca.set_fast_explode()
                }
                Ok(Some(ca.into_series()))
            }
            _ => PhysicalAggregation::aggregate(self, final_df, groups, state),
        }
    }
}

impl PhysicalAggregation for AggQuantileExpr {
    fn aggregate(
        &self,
        df: &DataFrame,
        groups: &GroupsProxy,
        state: &ExecutionState,
    ) -> Result<Option<Series>> {
        let series = self.expr.evaluate(df, state)?;
        let new_name = series.name().to_string();
        let opt_agg = series.agg_quantile(groups, self.quantile, self.interpol);

        let opt_agg = opt_agg.map(|mut agg| {
            agg.rename(&new_name);
            agg.into_series()
        });

        Ok(opt_agg)
    }
}
impl PhysicalAggregation for CastExpr {
    fn aggregate(
        &self,
        df: &DataFrame,
        groups: &GroupsProxy,
        state: &ExecutionState,
    ) -> Result<Option<Series>> {
        let agg_expr = self.input.as_agg_expr()?;
        let opt_agg = agg_expr.aggregate(df, groups, state)?;
        opt_agg.map(|agg| agg.cast(&self.data_type)).transpose()
    }
}

pub struct AggQuantileExpr {
    pub(crate) expr: Arc<dyn PhysicalExpr>,
    pub(crate) quantile: f64,
    pub(crate) interpol: QuantileInterpolOptions,
}

impl AggQuantileExpr {
    pub fn new(
        expr: Arc<dyn PhysicalExpr>,
        quantile: f64,
        interpol: QuantileInterpolOptions,
    ) -> Self {
        Self {
            expr,
            quantile,
            interpol,
        }
    }
}

impl PhysicalExpr for AggQuantileExpr {
    fn as_expression(&self) -> &Expr {
        unimplemented!()
    }

    fn evaluate(&self, _df: &DataFrame, _state: &ExecutionState) -> Result<Series> {
        unimplemented!()
    }
    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups<'a>(
        &self,
        _df: &DataFrame,
        _groups: &'a GroupsProxy,
        _state: &ExecutionState,
    ) -> Result<AggregationContext<'a>> {
        unimplemented!()
    }

    fn to_field(&self, input_schema: &Schema) -> Result<Field> {
        self.expr.to_field(input_schema)
    }

    fn as_agg_expr(&self) -> Result<&dyn PhysicalAggregation> {
        Ok(self)
    }
}
