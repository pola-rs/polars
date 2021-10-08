use crate::physical_plan::state::ExecutionState;
use crate::physical_plan::PhysicalAggregation;
use crate::prelude::*;
use polars_arrow::array::ValueSize;
use polars_core::chunked_array::builder::get_list_builder;
use polars_core::frame::groupby::{fmt_groupby_column, GroupByMethod, GroupTuples};
use polars_core::utils::NoNull;
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
        groups: &'a GroupTuples,
        state: &ExecutionState,
    ) -> Result<AggregationContext<'a>> {
        let out = self.aggregate(df, groups, state)?.ok_or_else(|| {
            PolarsError::ComputeError("Aggregation did not return a Series".into())
        })?;
        Ok(AggregationContext::new(out, Cow::Borrowed(groups)))
    }

    fn to_field(&self, input_schema: &Schema) -> Result<Field> {
        let field = self.expr.to_field(input_schema)?;
        let new_name = fmt_groupby_column(field.name(), self.agg_type);
        Ok(Field::new(&new_name, field.data_type().clone()))
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
        groups: &GroupTuples,
        state: &ExecutionState,
    ) -> Result<Option<Series>> {
        let mut ac = self.expr.evaluate_on_groups(df, groups, state)?;
        let new_name = fmt_groupby_column(ac.series().name(), self.agg_type);

        match self.agg_type {
            GroupByMethod::Min => {
                let agg_s = ac.flat().into_owned().agg_min(ac.groups());
                Ok(rename_option_series(agg_s, &new_name))
            }
            GroupByMethod::Max => {
                let agg_s = ac.flat().into_owned().agg_max(ac.groups());
                Ok(rename_option_series(agg_s, &new_name))
            }
            GroupByMethod::Median => {
                let agg_s = ac.flat().into_owned().agg_median(ac.groups());
                Ok(rename_option_series(agg_s, &new_name))
            }
            GroupByMethod::Mean => {
                let agg_s = ac.flat().into_owned().agg_mean(ac.groups());
                Ok(rename_option_series(agg_s, &new_name))
            }
            GroupByMethod::Sum => {
                let agg_s = ac.flat().into_owned().agg_sum(ac.groups());
                Ok(rename_option_series(agg_s, &new_name))
            }
            GroupByMethod::Count => {
                let mut ca: NoNull<UInt32Chunked> =
                    ac.groups().iter().map(|(_, g)| g.len() as u32).collect();
                ca.rename(&new_name);
                Ok(Some(ca.into_inner().into_series()))
            }
            GroupByMethod::First => {
                let mut agg_s = ac.flat().into_owned().agg_first(ac.groups());
                agg_s.rename(&new_name);
                Ok(Some(agg_s))
            }
            GroupByMethod::Last => {
                let mut agg_s = ac.flat().into_owned().agg_last(ac.groups());
                agg_s.rename(&new_name);
                Ok(Some(agg_s))
            }
            GroupByMethod::NUnique => {
                let opt_agg = ac.flat().into_owned().agg_n_unique(ac.groups());
                let opt_agg = opt_agg.map(|mut agg| {
                    agg.rename(&new_name);
                    agg.into_series()
                });
                Ok(opt_agg)
            }
            GroupByMethod::List => {
                let agg = ac.aggregated().into_owned();
                Ok(rename_option_series(Some(agg), &new_name))
            }
            GroupByMethod::Groups => {
                let mut column: ListChunked = ac
                    .groups()
                    .iter()
                    .map(|(_first, idx)| {
                        let ca: NoNull<UInt32Chunked> = idx.iter().map(|&v| v as u32).collect();
                        ca.into_inner().into_series()
                    })
                    .collect();

                column.rename(&new_name);
                Ok(Some(column.into_series()))
            }
            GroupByMethod::Std => {
                let agg_s = ac.flat().into_owned().agg_std(ac.groups());
                Ok(rename_option_series(agg_s, &new_name))
            }
            GroupByMethod::Var => {
                let agg_s = ac.flat().into_owned().agg_var(ac.groups());
                Ok(rename_option_series(agg_s, &new_name))
            }
            GroupByMethod::Quantile(_) => {
                // implemented explicitly in AggQuantile struct
                unimplemented!()
            }
        }
    }

    fn evaluate_partitioned(
        &self,
        df: &DataFrame,
        groups: &GroupTuples,
        state: &ExecutionState,
    ) -> Result<Option<Vec<Series>>> {
        match self.agg_type {
            GroupByMethod::Mean => {
                let series = self.expr.evaluate(df, state)?;
                let mut new_name = fmt_groupby_column(series.name(), self.agg_type);
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
                let new_name = fmt_groupby_column(series.name(), self.agg_type);
                let opt_agg = series.agg_list(groups);
                Ok(opt_agg.map(|mut s| {
                    s.rename(&new_name);
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
        groups: &GroupTuples,
        state: &ExecutionState,
    ) -> Result<Option<Series>> {
        match self.agg_type {
            GroupByMethod::Mean => {
                let series = self.expr.evaluate(final_df, state)?;
                let count_name = format!("{}__POLARS_MEAN_COUNT", series.name());
                let new_name = fmt_groupby_column(series.name(), self.agg_type);
                let count = final_df.column(&count_name).unwrap();

                let (agg_count, agg_s) =
                    POOL.join(|| count.agg_sum(groups), || series.agg_sum(groups));
                let agg_s = agg_s.map(|agg_s| &agg_s / &agg_count.unwrap());
                Ok(rename_option_series(agg_s, &new_name))
            }
            GroupByMethod::List => {
                let series = self.expr.evaluate(final_df, state)?;
                let ca = series.list().unwrap();
                let new_name = fmt_groupby_column(ca.name(), self.agg_type);

                let values_type = match ca.dtype() {
                    DataType::List(dt) => *dt.clone(),
                    _ => unreachable!(),
                };

                let mut builder =
                    get_list_builder(&values_type, ca.get_values_size(), ca.len(), &new_name);
                for (_, idx) in groups {
                    // Safety
                    // The indexes of the groupby operation are never out of bounds
                    let ca = unsafe { ca.take_unchecked(idx.iter().map(|i| *i as usize).into()) };
                    let s = ca.explode()?;
                    builder.append_series(&s);
                }
                let out = builder.finish();
                Ok(Some(out.into_series()))
            }
            _ => PhysicalAggregation::aggregate(self, final_df, groups, state),
        }
    }
}

impl PhysicalAggregation for AggQuantileExpr {
    fn aggregate(
        &self,
        df: &DataFrame,
        groups: &GroupTuples,
        state: &ExecutionState,
    ) -> Result<Option<Series>> {
        let series = self.expr.evaluate(df, state)?;
        let new_name = fmt_groupby_column(series.name(), GroupByMethod::Quantile(self.quantile));
        let opt_agg = series.agg_quantile(groups, self.quantile);

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
        groups: &GroupTuples,
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
}

impl AggQuantileExpr {
    pub fn new(expr: Arc<dyn PhysicalExpr>, quantile: f64) -> Self {
        Self { expr, quantile }
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
        _groups: &'a GroupTuples,
        _state: &ExecutionState,
    ) -> Result<AggregationContext<'a>> {
        unimplemented!()
    }

    fn to_field(&self, input_schema: &Schema) -> Result<Field> {
        let field = self.expr.to_field(input_schema)?;
        let new_name = fmt_groupby_column(field.name(), GroupByMethod::Quantile(self.quantile));
        Ok(Field::new(&new_name, field.data_type().clone()))
    }

    fn as_agg_expr(&self) -> Result<&dyn PhysicalAggregation> {
        Ok(self)
    }
}
