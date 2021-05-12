use crate::physical_plan::state::ExecutionState;
use crate::physical_plan::PhysicalAggregation;
use crate::prelude::*;
use polars_arrow::array::ValueSize;
use polars_core::chunked_array::builder::get_list_builder;
use polars_core::frame::groupby::{fmt_groupby_column, GroupByMethod, GroupTuples, GroupedMap};
use polars_core::prelude::*;
use polars_core::utils::NoNull;
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
    fn evaluate(&self, _df: &DataFrame, _state: &ExecutionState) -> Result<Series> {
        unimplemented!()
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

    #[allow(clippy::ptr_arg)]
    fn evaluate_partititioned2(
        &self,
        df: &DataFrame,
        g_maps: &[GroupedMap<Option<u64>>],
        state: &ExecutionState,
    ) -> Result<Option<Series>> {
        let series = self.expr.evaluate(df, state)?;

        // for every group map we run the aggregation
        // TODO: may switch parallelization order? par_iter here and normal iter in impl
        let mut iter = g_maps.iter().map(|g_map| series.part_agg_sum(g_map));

        // get the first result table
        let mut first = iter.next().unwrap();

        // maps over option, dtypes that cannot be accumulated return None
        Ok(first.map(|mut first| {
            // merge the table with the other tables in a recursive manner
            first.merge(iter.filter_map(|v| v).collect());

            first.finish(series.dtype().clone())
        }))
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
            .map(|agg| agg.cast_with_dtype(&self.data_type))
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
    fn evaluate(&self, _df: &DataFrame, _state: &ExecutionState) -> Result<Series> {
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
