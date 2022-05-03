use crate::physical_plan::state::ExecutionState;
use crate::physical_plan::PhysicalAggregation;
use crate::prelude::*;
use polars_arrow::export::arrow::{array::*, compute::concatenate::concatenate};
use polars_arrow::prelude::QuantileInterpolOptions;
use polars_arrow::utils::CustomIterTools;
use polars_core::frame::groupby::{GroupByMethod, GroupsProxy};
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
        groups: &'a GroupsProxy,
        state: &ExecutionState,
    ) -> Result<AggregationContext<'a>> {
        let out = self.aggregate(df, groups, state)?;
        Ok(AggregationContext::new(out, Cow::Borrowed(groups), true))
    }

    fn to_field(&self, input_schema: &Schema) -> Result<Field> {
        self.expr.to_field(input_schema)
    }

    fn as_agg_expr(&self) -> Result<&dyn PhysicalAggregation> {
        Ok(self)
    }
}

fn rename_series(mut s: Series, name: &str) -> Series {
    s.rename(name);
    s
}

impl PhysicalAggregation for AggregationExpr {
    fn aggregate(
        &self,
        df: &DataFrame,
        groups: &GroupsProxy,
        state: &ExecutionState,
    ) -> Result<Series> {
        let mut ac = self.expr.evaluate_on_groups(df, groups, state)?;
        // don't change names by aggregations as is done in polars-core
        let keep_name = ac.series().name().to_string();

        match self.agg_type {
            GroupByMethod::Min => {
                let agg_s = ac.flat_naive().into_owned().agg_min(ac.groups());
                Ok(rename_series(agg_s, &keep_name))
            }
            GroupByMethod::Max => {
                let agg_s = ac.flat_naive().into_owned().agg_max(ac.groups());
                Ok(rename_series(agg_s, &keep_name))
            }
            GroupByMethod::Median => {
                let agg_s = ac.flat_naive().into_owned().agg_median(ac.groups());
                Ok(rename_series(agg_s, &keep_name))
            }
            GroupByMethod::Mean => {
                let agg_s = ac.flat_naive().into_owned().agg_mean(ac.groups());
                Ok(rename_series(agg_s, &keep_name))
            }
            GroupByMethod::Sum => {
                let agg_s = ac.flat_naive().into_owned().agg_sum(ac.groups());
                Ok(rename_series(agg_s, &keep_name))
            }
            GroupByMethod::Count => {
                // a few fast paths that prevent materializing new groups
                match ac.update_groups {
                    UpdateGroups::WithSeriesLen => {
                        let list = ac
                            .series()
                            .list()
                            .expect("impl error, should be a list at this point");

                        let mut s = match list.chunks().len() {
                            1 => {
                                let arr = list.downcast_iter().next().unwrap();
                                let offsets = arr.offsets().as_slice();

                                let mut previous = 0i64;
                                let counts: NoNull<IdxCa> = offsets[1..]
                                    .iter()
                                    .map(|&o| {
                                        let len = (o - previous) as IdxSize;
                                        previous = o;
                                        len
                                    })
                                    .collect_trusted();
                                counts.into_inner()
                            }
                            _ => {
                                let counts: NoNull<IdxCa> = list
                                    .amortized_iter()
                                    .map(|s| {
                                        if let Some(s) = s {
                                            s.as_ref().len() as IdxSize
                                        } else {
                                            1
                                        }
                                    })
                                    .collect_trusted();
                                counts.into_inner()
                            }
                        };
                        s.rename(&keep_name);
                        Ok(s.into_series())
                    }
                    UpdateGroups::WithGroupsLen => {
                        // no need to update the groups
                        // we can just get the attribute, because we only need the length,
                        // not the correct order
                        let mut ca = ac.groups.group_count();
                        ca.rename(&keep_name);
                        Ok(ca.into_series())
                    }
                    // materialize groups
                    _ => {
                        let mut ca = ac.groups().group_count();
                        ca.rename(&keep_name);
                        Ok(ca.into_series())
                    }
                }
            }
            GroupByMethod::First => {
                let mut agg_s = ac.flat_naive().into_owned().agg_first(ac.groups());
                agg_s.rename(&keep_name);
                Ok(agg_s)
            }
            GroupByMethod::Last => {
                let mut agg_s = ac.flat_naive().into_owned().agg_last(ac.groups());
                agg_s.rename(&keep_name);
                Ok(agg_s)
            }
            GroupByMethod::NUnique => {
                let agg_s = ac.flat_naive().into_owned().agg_n_unique(ac.groups());
                Ok(rename_series(agg_s, &keep_name))
            }
            GroupByMethod::List => {
                let agg = ac.aggregated();
                Ok(rename_series(agg, &keep_name))
            }
            GroupByMethod::Groups => {
                let mut column: ListChunked = ac.groups().as_list_chunked();
                column.rename(&keep_name);
                Ok(column.into_series())
            }
            GroupByMethod::Std => {
                let agg_s = ac.flat_naive().into_owned().agg_std(ac.groups());
                Ok(rename_series(agg_s, &keep_name))
            }
            GroupByMethod::Var => {
                let agg_s = ac.flat_naive().into_owned().agg_var(ac.groups());
                Ok(rename_series(agg_s, &keep_name))
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
    ) -> Result<Vec<Series>> {
        match self.agg_type {
            GroupByMethod::Mean => {
                let series = self.expr.evaluate(df, state)?;
                let mut new_name = series.name().to_string();
                let agg_s = series.agg_sum(groups);

                // the
                if !agg_s.dtype().is_numeric() {
                    Ok(vec![agg_s])
                } else {
                    let mut agg_s = match agg_s.dtype() {
                        DataType::Float32 => agg_s,
                        _ => agg_s.cast(&DataType::Float64).unwrap(),
                    };
                    agg_s.rename(&new_name);
                    new_name.push_str("__POLARS_MEAN_COUNT");
                    let mut count_s = series.agg_valid_count(groups);
                    count_s.rename(&new_name);
                    Ok(vec![agg_s, count_s])
                }
            }
            GroupByMethod::List => {
                let series = self.expr.evaluate(df, state)?;
                let new_name = series.name();
                let mut agg = series.agg_list(groups);
                agg.rename(new_name);
                Ok(vec![agg])
            }
            _ => Ok(vec![PhysicalAggregation::aggregate(
                self, df, groups, state,
            )?]),
        }
    }

    fn evaluate_partitioned_final(
        &self,
        final_df: &DataFrame,
        groups: &GroupsProxy,
        state: &ExecutionState,
    ) -> Result<Series> {
        match self.agg_type {
            GroupByMethod::Mean => {
                let series = self.expr.evaluate(final_df, state)?;
                let count_name = format!("{}__POLARS_MEAN_COUNT", series.name());
                let new_name = series.name().to_string();

                // The partitioned groupby yields
                // "foo" and "foo__POLARS_MEAN_COUNT"
                // if it can do the mean.
                // for instance on string columns, it only returns "foo" (with only null values)
                // the count column is not there and we can just yield the "foo" column
                match final_df.column(&count_name).ok() {
                    // we can finalize the aggregation
                    Some(count) => {
                        let (agg_count, agg_s) =
                            POOL.join(|| count.agg_sum(groups), || series.agg_sum(groups));
                        let agg_s = &agg_s / &agg_count;
                        Ok(rename_series(agg_s, &new_name))
                    }
                    None => Ok(Series::full_null(&new_name, groups.len(), series.dtype())),
                }
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
                Ok(ca.into_series())
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
    ) -> Result<Series> {
        let mut ac = self.expr.evaluate_on_groups(df, groups, state)?;
        // don't change names by aggregations as is done in polars-core
        let keep_name = ac.series().name().to_string();

        let mut agg =
            ac.flat_naive()
                .into_owned()
                .agg_quantile(ac.groups(), self.quantile, self.interpol);
        agg.rename(&keep_name);
        Ok(agg)
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
        df: &DataFrame,
        groups: &'a GroupsProxy,
        state: &ExecutionState,
    ) -> Result<AggregationContext<'a>> {
        let out = self.aggregate(df, groups, state)?;
        Ok(AggregationContext::new(out, Cow::Borrowed(groups), true))
    }

    fn to_field(&self, input_schema: &Schema) -> Result<Field> {
        self.expr.to_field(input_schema)
    }

    fn as_agg_expr(&self) -> Result<&dyn PhysicalAggregation> {
        Ok(self)
    }
}
