//! Implementations of PhysicalAggregation. These aggregations are called by the groupby context,
//! and nowhere else. Note, that this differes from evaluate on groups, which is also called in that
//! context, but typically before aggregation

use crate::physical_plan::state::ExecutionState;
use crate::physical_plan::PhysicalAggregation;
use crate::prelude::*;
use polars_arrow::array::ValueSize;
use polars_core::chunked_array::builder::get_list_builder;
use polars_core::frame::groupby::{fmt_groupby_column, GroupByMethod, GroupTuples};
use polars_core::prelude::*;
use polars_core::utils::NoNull;

impl PhysicalAggregation for AliasExpr {
    fn aggregate(
        &self,
        df: &DataFrame,
        groups: &GroupTuples,
        state: &ExecutionState,
    ) -> Result<Option<Series>> {
        let agg_expr = self.physical_expr.as_agg_expr()?;
        let opt_agg = agg_expr.aggregate(df, groups, state)?;
        Ok(opt_agg.map(|mut agg| {
            agg.rename(&self.name);
            agg
        }))
    }
}

impl PhysicalAggregation for SortExpr {
    // As a final aggregation a Sort returns a list array.
    fn aggregate(
        &self,
        df: &DataFrame,
        groups: &GroupTuples,
        state: &ExecutionState,
    ) -> Result<Option<Series>> {
        let s = self.physical_expr.evaluate(df, state)?;
        let agg_s = s.agg_list(groups);
        let out = agg_s.map(|s| {
            s.list()
                .unwrap()
                .into_iter()
                .map(|opt_s| opt_s.map(|s| s.sort(self.reverse)))
                .collect::<ListChunked>()
                .into_series()
        });
        Ok(out)
    }
}

impl PhysicalAggregation for SortByExpr {
    // As a final aggregation a Sort returns a list array.
    fn aggregate(
        &self,
        df: &DataFrame,
        groups: &GroupTuples,
        state: &ExecutionState,
    ) -> Result<Option<Series>> {
        let s = self.input.evaluate(df, state)?;
        let s_sort_by = self.by.evaluate(df, state)?;

        let s_sort_by = s_sort_by.agg_list(groups).ok_or_else(|| {
            PolarsError::Other(format!("cannot aggregate {:?} as list array", self.expr).into())
        })?;

        let agg_s = s.agg_list(groups);
        let out = agg_s.map(|s| {
            s.list()
                .unwrap()
                .into_iter()
                .zip(s_sort_by.list().unwrap())
                .map(|(opt_s, opt_sort_by)| {
                    match (opt_s, opt_sort_by) {
                        (Some(s), Some(sort_by)) => {
                            let sorted_idx = sort_by.argsort(self.reverse);
                            // Safety:
                            // sorted index are within bounds
                            unsafe { s.take_unchecked(&sorted_idx) }.ok()
                        }
                        _ => None,
                    }
                })
                .collect::<ListChunked>()
                .into_series()
        });
        Ok(out)
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
        let (series, groups) = self.expr.evaluate_on_groups(df, groups, state)?;
        let new_name = fmt_groupby_column(series.name(), self.agg_type);

        match self.agg_type {
            GroupByMethod::Min => {
                let agg_s = series.agg_min(&groups);
                Ok(rename_option_series(agg_s, &new_name))
            }
            GroupByMethod::Max => {
                let agg_s = series.agg_max(&groups);
                Ok(rename_option_series(agg_s, &new_name))
            }
            GroupByMethod::Median => {
                let agg_s = series.agg_median(&groups);
                Ok(rename_option_series(agg_s, &new_name))
            }
            GroupByMethod::Mean => {
                let agg_s = series.agg_mean(&groups);
                Ok(rename_option_series(agg_s, &new_name))
            }
            GroupByMethod::Sum => {
                let agg_s = series.agg_sum(&groups);
                Ok(rename_option_series(agg_s, &new_name))
            }
            GroupByMethod::Count => {
                let mut ca: NoNull<UInt32Chunked> =
                    groups.iter().map(|(_, g)| g.len() as u32).collect();
                ca.rename(&new_name);
                Ok(Some(ca.into_inner().into_series()))
            }
            GroupByMethod::First => {
                let mut agg_s = series.agg_first(&groups);
                agg_s.rename(&new_name);
                Ok(Some(agg_s))
            }
            GroupByMethod::Last => {
                let mut agg_s = series.agg_last(&groups);
                agg_s.rename(&new_name);
                Ok(Some(agg_s))
            }
            GroupByMethod::NUnique => {
                let opt_agg = series.agg_n_unique(&groups);
                let opt_agg = opt_agg.map(|mut agg| {
                    agg.rename(&new_name);
                    agg.into_series()
                });
                Ok(opt_agg)
            }
            GroupByMethod::List => {
                let opt_agg = series.agg_list(&groups);
                Ok(rename_option_series(opt_agg, &new_name))
            }
            GroupByMethod::Groups => {
                let mut column: ListChunked = groups
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
                let agg_s = series.agg_std(&groups);
                Ok(rename_option_series(agg_s, &new_name))
            }
            GroupByMethod::Var => {
                let agg_s = series.agg_var(&groups);
                Ok(rename_option_series(agg_s, &new_name))
            }
            GroupByMethod::Quantile(_) => {
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

                if let Some(mut agg_s) = agg_s {
                    agg_s.rename(&new_name);
                    new_name.push_str("__POLARS_MEAN_COUNT");
                    let ca: NoNull<UInt32Chunked> =
                        groups.iter().map(|t| t.1.len() as u32).collect();
                    let mut count_s = ca.into_inner().into_series();
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
                // divide by the count
                let series = &series / count;
                let agg_s = series.agg_sum(groups);
                Ok(rename_option_series(agg_s, &new_name))
            }
            GroupByMethod::List => {
                let series = self.expr.evaluate(final_df, state)?;
                let ca = series.list().unwrap();
                let new_name = fmt_groupby_column(ca.name(), self.agg_type);

                let values_type = match ca.dtype() {
                    DataType::List(dt) => DataType::from(dt),
                    _ => unreachable!(),
                };

                let mut builder =
                    get_list_builder(&values_type, ca.get_values_size(), ca.len(), &new_name);
                for (_, idx) in groups {
                    // Safety
                    // The indexes of the groupby operation are never out of bounds
                    let ca = unsafe { ca.take_unchecked(idx.iter().map(|i| *i as usize).into()) };
                    let s = ca.explode_and_offsets()?.0;
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
        opt_agg
            .map(|agg| agg.cast_with_datatype(&self.data_type))
            .transpose()
    }
}

impl PhysicalAggregation for ApplyExpr {
    fn aggregate(
        &self,
        df: &DataFrame,
        groups: &GroupTuples,
        state: &ExecutionState,
    ) -> Result<Option<Series>> {
        match self.input.as_agg_expr() {
            // layer below is also an aggregation expr.
            Ok(expr) => {
                let aggregated = expr.aggregate(df, groups, state)?;
                let out = aggregated.map(|s| self.function.call_udf(s));
                out.transpose()
            }
            Err(_) => {
                let series = self.input.evaluate(df, state)?;
                series
                    .agg_list(groups)
                    .map(|s| {
                        let s = self.function.call_udf(s);
                        s.map(|mut s| {
                            s.rename(series.name());
                            s
                        })
                    })
                    .map_or(Ok(None), |v| v.map(Some))
            }
        }
    }
}
impl PhysicalAggregation for SliceExpr {
    // As a final aggregation a Slice returns a list array.
    fn aggregate(
        &self,
        df: &DataFrame,
        groups: &GroupTuples,
        state: &ExecutionState,
    ) -> Result<Option<Series>> {
        let s = self.input.evaluate(df, state)?;
        let agg_s = s.agg_list(groups);
        let out = agg_s.map(|s| {
            s.list()
                .unwrap()
                .into_iter()
                .map(|opt_s| opt_s.map(|s| s.slice(self.offset, self.len)))
                .collect::<ListChunked>()
                .into_series()
        });
        Ok(out)
    }
}

impl PhysicalAggregation for BinaryFunctionExpr {
    fn aggregate(
        &self,
        df: &DataFrame,
        groups: &GroupTuples,
        state: &ExecutionState,
    ) -> Result<Option<Series>> {
        let a = self.input_a.evaluate(df, state)?;
        let b = self.input_b.evaluate(df, state)?;

        let agg_a = a.agg_list(groups).expect("no data?");
        let agg_b = b.agg_list(groups).expect("no data?");

        // keep track of the output lengths. If they are all unit length,
        // we can explode the array as it would have the same length as the no. of groups
        // if it is not all unit length it should remain a listarray

        let mut all_unit_length = true;

        let ca = agg_a
            .list()
            .unwrap()
            .into_iter()
            .zip(agg_b.list().unwrap())
            .map(|(opt_a, opt_b)| match (opt_a, opt_b) {
                (Some(a), Some(b)) => {
                    let out = self.function.call_udf(a, b).ok();

                    if let Some(s) = &out {
                        if s.len() != 1 {
                            all_unit_length = false;
                        }
                    }
                    out
                }
                _ => None,
            })
            .collect::<ListChunked>();

        if all_unit_length {
            return Ok(Some(ca.explode()?));
        }
        Ok(Some(ca.into_series()))
    }
}

impl PhysicalAggregation for BinaryExpr {
    fn aggregate(
        &self,
        df: &DataFrame,
        groups: &GroupTuples,
        state: &ExecutionState,
    ) -> Result<Option<Series>> {
        match (self.left.as_agg_expr(), self.right.as_agg_expr()) {
            (Ok(left), Err(_)) => {
                let opt_agg = left.aggregate(df, groups, state)?;
                let rhs = self.right.evaluate(df, state)?;
                opt_agg
                    .map(|agg| apply_operator(&agg, &rhs, self.op))
                    .transpose()
            }
            (Err(_), Ok(right)) => {
                let opt_agg = right.aggregate(df, groups, state)?;
                let lhs = self.left.evaluate(df, state)?;
                opt_agg
                    .map(|agg| apply_operator(&lhs, &agg, self.op))
                    .transpose()
            }
            (Ok(left), Ok(right)) => {
                let right_agg = right.aggregate(df, groups, state)?;
                left.aggregate(df, groups, state)?
                    .and_then(|left| right_agg.map(|right| apply_operator(&left, &right, self.op)))
                    .transpose()
            }
            (_, _) => Err(PolarsError::Other(
                "both expressions could not be used in an aggregation context.".into(),
            )),
        }
    }
}
