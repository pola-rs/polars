use std::borrow::Cow;

use arrow::array::*;
use arrow::compute::concatenate::concatenate;
use arrow::legacy::utils::CustomIterTools;
use arrow::offset::Offsets;
use polars_core::chunked_array::metadata::MetadataEnv;
use polars_core::prelude::*;
use polars_core::series::IsSorted;
use polars_core::utils::{NoNull, _split_offsets};
use polars_core::POOL;
#[cfg(feature = "propagate_nans")]
use polars_ops::prelude::nan_propagating_aggregate;
use rayon::prelude::*;

use super::*;
use crate::expressions::AggState::{AggregatedList, AggregatedScalar};
use crate::expressions::{
    AggState, AggregationContext, PartitionedAggregation, PhysicalExpr, UpdateGroups,
};

#[derive(Debug, Clone, Copy)]
pub struct AggregationType {
    pub(crate) groupby: GroupByMethod,
    pub(crate) allow_threading: bool,
}

pub(crate) struct AggregationExpr {
    pub(crate) input: Arc<dyn PhysicalExpr>,
    pub(crate) agg_type: AggregationType,
    field: Option<Field>,
}

impl AggregationExpr {
    pub fn new(
        expr: Arc<dyn PhysicalExpr>,
        agg_type: AggregationType,
        field: Option<Field>,
    ) -> Self {
        Self {
            input: expr,
            agg_type,
            field,
        }
    }
}

impl PhysicalExpr for AggregationExpr {
    fn as_expression(&self) -> Option<&Expr> {
        None
    }

    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Series> {
        let s = self.input.evaluate(df, state)?;

        let AggregationType {
            groupby,
            allow_threading,
        } = self.agg_type;

        let is_float = s.dtype().is_float();
        let group_by = match groupby {
            GroupByMethod::NanMin if !is_float => GroupByMethod::Min,
            GroupByMethod::NanMax if !is_float => GroupByMethod::Max,
            gb => gb,
        };

        match group_by {
            GroupByMethod::Min => {
                if MetadataEnv::experimental_enabled() {
                    if let Some(sc) = s.get_metadata().and_then(|v| v.min_value()) {
                        return Ok(sc.into_series(s.name()));
                    }
                }

                match s.is_sorted_flag() {
                    IsSorted::Ascending | IsSorted::Descending => {
                        s.min_reduce().map(|sc| sc.into_series(s.name()))
                    },
                    IsSorted::Not => parallel_op_series(
                        |s| s.min_reduce().map(|sc| sc.into_series(s.name())),
                        s,
                        allow_threading,
                    ),
                }
            },
            #[cfg(feature = "propagate_nans")]
            GroupByMethod::NanMin => parallel_op_series(
                |s| {
                    Ok(polars_ops::prelude::nan_propagating_aggregate::nan_min_s(
                        &s,
                        s.name(),
                    ))
                },
                s,
                allow_threading,
            ),
            #[cfg(not(feature = "propagate_nans"))]
            GroupByMethod::NanMin => {
                panic!("activate 'propagate_nans' feature")
            },
            GroupByMethod::Max => {
                if MetadataEnv::experimental_enabled() {
                    if let Some(sc) = s.get_metadata().and_then(|v| v.max_value()) {
                        return Ok(sc.into_series(s.name()));
                    }
                }

                match s.is_sorted_flag() {
                    IsSorted::Ascending | IsSorted::Descending => {
                        s.max_reduce().map(|sc| sc.into_series(s.name()))
                    },
                    IsSorted::Not => parallel_op_series(
                        |s| s.max_reduce().map(|sc| sc.into_series(s.name())),
                        s,
                        allow_threading,
                    ),
                }
            },
            #[cfg(feature = "propagate_nans")]
            GroupByMethod::NanMax => parallel_op_series(
                |s| {
                    Ok(polars_ops::prelude::nan_propagating_aggregate::nan_max_s(
                        &s,
                        s.name(),
                    ))
                },
                s,
                allow_threading,
            ),
            #[cfg(not(feature = "propagate_nans"))]
            GroupByMethod::NanMax => {
                panic!("activate 'propagate_nans' feature")
            },
            GroupByMethod::Median => s.median_reduce().map(|sc| sc.into_series(s.name())),
            GroupByMethod::Mean => Ok(s.mean_reduce().into_series(s.name())),
            GroupByMethod::First => Ok(if s.is_empty() {
                Series::full_null(s.name(), 1, s.dtype())
            } else {
                s.head(Some(1))
            }),
            GroupByMethod::Last => Ok(if s.is_empty() {
                Series::full_null(s.name(), 1, s.dtype())
            } else {
                s.tail(Some(1))
            }),
            GroupByMethod::Sum => parallel_op_series(
                |s| s.sum_reduce().map(|sc| sc.into_series(s.name())),
                s,
                allow_threading,
            ),
            GroupByMethod::Groups => unreachable!(),
            GroupByMethod::NUnique => {
                if MetadataEnv::experimental_enabled() {
                    if let Some(count) = s.get_metadata().and_then(|v| v.distinct_count()) {
                        let count = count + IdxSize::from(s.null_count() > 0);
                        return Ok(IdxCa::from_slice(s.name(), &[count]).into_series());
                    }
                }

                s.n_unique()
                    .map(|count| IdxCa::from_slice(s.name(), &[count as IdxSize]).into_series())
            },
            GroupByMethod::Count { include_nulls } => {
                let count = s.len() - s.null_count() * !include_nulls as usize;

                Ok(IdxCa::from_slice(s.name(), &[count as IdxSize]).into_series())
            },
            GroupByMethod::Implode => s.implode().map(|ca| ca.into_series()),
            GroupByMethod::Std(ddof) => s.std_reduce(ddof).map(|sc| sc.into_series(s.name())),
            GroupByMethod::Var(ddof) => s.var_reduce(ddof).map(|sc| sc.into_series(s.name())),
            GroupByMethod::Quantile(_, _) => unimplemented!(),
        }
    }
    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupsProxy,
        state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        let mut ac = self.input.evaluate_on_groups(df, groups, state)?;
        // don't change names by aggregations as is done in polars-core
        let keep_name = ac.series().name().to_string();
        polars_ensure!(!matches!(ac.agg_state(), AggState::Literal(_)), ComputeError: "cannot aggregate a literal");

        if let AggregatedScalar(_) = ac.agg_state() {
            match self.agg_type.groupby {
                GroupByMethod::Implode => {},
                _ => {
                    polars_bail!(ComputeError: "cannot aggregate as {}, the column is already aggregated", self.agg_type.groupby);
                },
            }
        }

        // SAFETY:
        // groups must always be in bounds.
        let out = unsafe {
            match self.agg_type.groupby {
                GroupByMethod::Min => {
                    let (s, groups) = ac.get_final_aggregation();
                    let agg_s = s.agg_min(&groups);
                    AggregatedScalar(rename_series(agg_s, &keep_name))
                },
                GroupByMethod::Max => {
                    let (s, groups) = ac.get_final_aggregation();
                    let agg_s = s.agg_max(&groups);
                    AggregatedScalar(rename_series(agg_s, &keep_name))
                },
                GroupByMethod::Median => {
                    let (s, groups) = ac.get_final_aggregation();
                    let agg_s = s.agg_median(&groups);
                    AggregatedScalar(rename_series(agg_s, &keep_name))
                },
                GroupByMethod::Mean => {
                    let (s, groups) = ac.get_final_aggregation();
                    let agg_s = s.agg_mean(&groups);
                    AggregatedScalar(rename_series(agg_s, &keep_name))
                },
                GroupByMethod::Sum => {
                    let (s, groups) = ac.get_final_aggregation();
                    let agg_s = s.agg_sum(&groups);
                    AggregatedScalar(rename_series(agg_s, &keep_name))
                },
                GroupByMethod::Count { include_nulls } => {
                    if include_nulls || ac.series().null_count() == 0 {
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
                                    },
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
                                    },
                                };
                                s.rename(&keep_name);
                                AggregatedScalar(s.into_series())
                            },
                            UpdateGroups::WithGroupsLen => {
                                // no need to update the groups
                                // we can just get the attribute, because we only need the length,
                                // not the correct order
                                let mut ca = ac.groups.group_count();
                                ca.rename(&keep_name);
                                AggregatedScalar(ca.into_series())
                            },
                            // materialize groups
                            _ => {
                                let mut ca = ac.groups().group_count();
                                ca.rename(&keep_name);
                                AggregatedScalar(ca.into_series())
                            },
                        }
                    } else {
                        // TODO: optimize this/and write somewhere else.
                        match ac.agg_state() {
                            AggState::Literal(s) | AggState::AggregatedScalar(s) => {
                                AggregatedScalar(Series::new(
                                    &keep_name,
                                    [(s.len() as IdxSize - s.null_count() as IdxSize)],
                                ))
                            },
                            AggState::AggregatedList(s) => {
                                let ca = s.list()?;
                                let out: IdxCa = ca
                                    .into_iter()
                                    .map(|opt_s| {
                                        opt_s
                                            .map(|s| s.len() as IdxSize - s.null_count() as IdxSize)
                                    })
                                    .collect();
                                AggregatedScalar(rename_series(out.into_series(), &keep_name))
                            },
                            AggState::NotAggregated(s) => {
                                let s = s.clone();
                                let groups = ac.groups();
                                let out: IdxCa = if matches!(s.dtype(), &DataType::Null) {
                                    IdxCa::full(s.name(), 0, groups.len())
                                } else {
                                    match groups.as_ref() {
                                        GroupsProxy::Idx(idx) => {
                                            let s = s.rechunk();
                                            let array = &s.chunks()[0];
                                            let validity = array.validity().unwrap();
                                            idx.iter()
                                                .map(|(_, g)| {
                                                    let mut count = 0 as IdxSize;
                                                    // Count valid values
                                                    g.iter().for_each(|i| {
                                                        count += validity
                                                            .get_bit_unchecked(*i as usize)
                                                            as IdxSize;
                                                    });
                                                    count
                                                })
                                                .collect_ca_trusted_with_dtype(
                                                    &keep_name, IDX_DTYPE,
                                                )
                                        },
                                        GroupsProxy::Slice { groups, .. } => {
                                            // Slice and use computed null count
                                            groups
                                                .iter()
                                                .map(|g| {
                                                    let start = g[0];
                                                    let len = g[1];
                                                    len - s
                                                        .slice(start as i64, len as usize)
                                                        .null_count()
                                                        as IdxSize
                                                })
                                                .collect_ca_trusted_with_dtype(
                                                    &keep_name, IDX_DTYPE,
                                                )
                                        },
                                    }
                                };
                                AggregatedScalar(out.into_series())
                            },
                        }
                    }
                },
                GroupByMethod::First => {
                    let (s, groups) = ac.get_final_aggregation();
                    let agg_s = s.agg_first(&groups);
                    AggregatedScalar(rename_series(agg_s, &keep_name))
                },
                GroupByMethod::Last => {
                    let (s, groups) = ac.get_final_aggregation();
                    let agg_s = s.agg_last(&groups);
                    AggregatedScalar(rename_series(agg_s, &keep_name))
                },
                GroupByMethod::NUnique => {
                    let (s, groups) = ac.get_final_aggregation();
                    let agg_s = s.agg_n_unique(&groups);
                    AggregatedScalar(rename_series(agg_s, &keep_name))
                },
                GroupByMethod::Implode => {
                    // if the aggregation is already
                    // in an aggregate flat state for instance by
                    // a mean aggregation, we simply convert to list
                    //
                    // if it is not, we traverse the groups and create
                    // a list per group.
                    let s = match ac.agg_state() {
                        // mean agg:
                        // -> f64 -> list<f64>
                        AggState::AggregatedScalar(s) => s.reshape_list(&[-1, 1]).unwrap(),
                        _ => {
                            let agg = ac.aggregated();
                            agg.as_list().into_series()
                        },
                    };
                    AggregatedList(rename_series(s, &keep_name))
                },
                GroupByMethod::Groups => {
                    let mut column: ListChunked = ac.groups().as_list_chunked();
                    column.rename(&keep_name);
                    AggregatedScalar(column.into_series())
                },
                GroupByMethod::Std(ddof) => {
                    let (s, groups) = ac.get_final_aggregation();
                    let agg_s = s.agg_std(&groups, ddof);
                    AggregatedScalar(rename_series(agg_s, &keep_name))
                },
                GroupByMethod::Var(ddof) => {
                    let (s, groups) = ac.get_final_aggregation();
                    let agg_s = s.agg_var(&groups, ddof);
                    AggregatedScalar(rename_series(agg_s, &keep_name))
                },
                GroupByMethod::Quantile(_, _) => {
                    // implemented explicitly in AggQuantile struct
                    unimplemented!()
                },
                GroupByMethod::NanMin => {
                    #[cfg(feature = "propagate_nans")]
                    {
                        let (s, groups) = ac.get_final_aggregation();
                        let agg_s = if s.dtype().is_float() {
                            nan_propagating_aggregate::group_agg_nan_min_s(&s, &groups)
                        } else {
                            s.agg_min(&groups)
                        };
                        AggregatedScalar(rename_series(agg_s, &keep_name))
                    }
                    #[cfg(not(feature = "propagate_nans"))]
                    {
                        panic!("activate 'propagate_nans' feature")
                    }
                },
                GroupByMethod::NanMax => {
                    #[cfg(feature = "propagate_nans")]
                    {
                        let (s, groups) = ac.get_final_aggregation();
                        let agg_s = if s.dtype().is_float() {
                            nan_propagating_aggregate::group_agg_nan_max_s(&s, &groups)
                        } else {
                            s.agg_max(&groups)
                        };
                        AggregatedScalar(rename_series(agg_s, &keep_name))
                    }
                    #[cfg(not(feature = "propagate_nans"))]
                    {
                        panic!("activate 'propagate_nans' feature")
                    }
                },
            }
        };

        Ok(AggregationContext::from_agg_state(
            out,
            Cow::Borrowed(groups),
        ))
    }

    fn to_field(&self, input_schema: &Schema) -> PolarsResult<Field> {
        if let Some(field) = self.field.as_ref() {
            Ok(field.clone())
        } else {
            self.input.to_field(input_schema)
        }
    }

    fn as_partitioned_aggregator(&self) -> Option<&dyn PartitionedAggregation> {
        Some(self)
    }
}

fn rename_series(mut s: Series, name: &str) -> Series {
    s.rename(name);
    s
}

impl PartitionedAggregation for AggregationExpr {
    fn evaluate_partitioned(
        &self,
        df: &DataFrame,
        groups: &GroupsProxy,
        state: &ExecutionState,
    ) -> PolarsResult<Series> {
        let expr = self.input.as_partitioned_aggregator().unwrap();
        let series = expr.evaluate_partitioned(df, groups, state)?;

        // SAFETY:
        // groups are in bounds
        unsafe {
            match self.agg_type.groupby {
                #[cfg(feature = "dtype-struct")]
                GroupByMethod::Mean => {
                    let new_name = series.name().to_string();

                    // ensure we don't overflow
                    // the all 8 and 16 bits integers are already upcasted to int16 on `agg_sum`
                    let mut agg_s = if matches!(series.dtype(), DataType::Int32 | DataType::UInt32)
                    {
                        series.cast(&DataType::Int64).unwrap().agg_sum(groups)
                    } else {
                        series.agg_sum(groups)
                    };
                    agg_s.rename(&new_name);

                    if !agg_s.dtype().is_numeric() {
                        Ok(agg_s)
                    } else {
                        let agg_s = match agg_s.dtype() {
                            DataType::Float32 => agg_s,
                            _ => agg_s.cast(&DataType::Float64).unwrap(),
                        };
                        let mut count_s = series.agg_valid_count(groups);
                        count_s.rename("__POLARS_COUNT");
                        Ok(StructChunked::from_series(&new_name, &[agg_s, count_s])
                            .unwrap()
                            .into_series())
                    }
                },
                GroupByMethod::Implode => {
                    let new_name = series.name();
                    let mut agg = series.agg_list(groups);
                    agg.rename(new_name);
                    Ok(agg)
                },
                GroupByMethod::First => {
                    let mut agg = series.agg_first(groups);
                    agg.rename(series.name());
                    Ok(agg)
                },
                GroupByMethod::Last => {
                    let mut agg = series.agg_last(groups);
                    agg.rename(series.name());
                    Ok(agg)
                },
                GroupByMethod::Max => {
                    let mut agg = series.agg_max(groups);
                    agg.rename(series.name());
                    Ok(agg)
                },
                GroupByMethod::Min => {
                    let mut agg = series.agg_min(groups);
                    agg.rename(series.name());
                    Ok(agg)
                },
                GroupByMethod::Sum => {
                    let mut agg = series.agg_sum(groups);
                    agg.rename(series.name());
                    Ok(agg)
                },
                GroupByMethod::Count {
                    include_nulls: true,
                } => {
                    let mut ca = groups.group_count();
                    ca.rename(series.name());
                    Ok(ca.into_series())
                },
                _ => {
                    unimplemented!()
                },
            }
        }
    }

    fn finalize(
        &self,
        partitioned: Series,
        groups: &GroupsProxy,
        _state: &ExecutionState,
    ) -> PolarsResult<Series> {
        match self.agg_type.groupby {
            GroupByMethod::Count {
                include_nulls: true,
            }
            | GroupByMethod::Sum => {
                let mut agg = unsafe { partitioned.agg_sum(groups) };
                agg.rename(partitioned.name());
                Ok(agg)
            },
            #[cfg(feature = "dtype-struct")]
            GroupByMethod::Mean => {
                let new_name = partitioned.name();
                match partitioned.dtype() {
                    DataType::Struct(_) => {
                        let ca = partitioned.struct_().unwrap();
                        let fields = ca.fields_as_series();
                        let sum = &fields[0];
                        let count = &fields[1];
                        let (agg_count, agg_s) =
                            unsafe { POOL.join(|| count.agg_sum(groups), || sum.agg_sum(groups)) };
                        let agg_s = &agg_s / &agg_count;
                        Ok(rename_series(agg_s?, new_name))
                    },
                    _ => Ok(Series::full_null(
                        new_name,
                        groups.len(),
                        partitioned.dtype(),
                    )),
                }
            },
            GroupByMethod::Implode => {
                // the groups are scattered over multiple groups/sub dataframes.
                // we now must collect them into a single group
                let ca = partitioned.list().unwrap();
                let new_name = partitioned.name().to_string();

                let mut values = Vec::with_capacity(groups.len());
                let mut can_fast_explode = true;

                let mut offsets = Vec::<i64>::with_capacity(groups.len() + 1);
                let mut length_so_far = 0i64;
                offsets.push(length_so_far);

                let mut process_group = |ca: ListChunked| -> PolarsResult<()> {
                    let s = ca.explode()?;
                    length_so_far += s.len() as i64;
                    offsets.push(length_so_far);
                    values.push(s.chunks()[0].clone());

                    if s.len() == 0 {
                        can_fast_explode = false;
                    }
                    Ok(())
                };

                match groups {
                    GroupsProxy::Idx(groups) => {
                        for (_, idx) in groups {
                            let ca = unsafe {
                                // SAFETY:
                                // The indexes of the group_by operation are never out of bounds
                                ca.take_unchecked(idx)
                            };
                            process_group(ca)?;
                        }
                    },
                    GroupsProxy::Slice { groups, .. } => {
                        for [first, len] in groups {
                            let len = *len as usize;
                            let ca = ca.slice(*first as i64, len);
                            process_group(ca)?;
                        }
                    },
                }

                let vals = values.iter().map(|arr| &**arr).collect::<Vec<_>>();
                let values = concatenate(&vals).unwrap();

                let data_type = ListArray::<i64>::default_datatype(values.data_type().clone());
                // SAFETY: offsets are monotonically increasing.
                let arr = ListArray::<i64>::new(
                    data_type,
                    unsafe { Offsets::new_unchecked(offsets).into() },
                    values,
                    None,
                );
                let mut ca = ListChunked::with_chunk(&new_name, arr);
                if can_fast_explode {
                    ca.set_fast_explode()
                }
                Ok(ca.into_series().as_list().into_series())
            },
            GroupByMethod::First => {
                let mut agg = unsafe { partitioned.agg_first(groups) };
                agg.rename(partitioned.name());
                Ok(agg)
            },
            GroupByMethod::Last => {
                let mut agg = unsafe { partitioned.agg_last(groups) };
                agg.rename(partitioned.name());
                Ok(agg)
            },
            GroupByMethod::Max => {
                let mut agg = unsafe { partitioned.agg_max(groups) };
                agg.rename(partitioned.name());
                Ok(agg)
            },
            GroupByMethod::Min => {
                let mut agg = unsafe { partitioned.agg_min(groups) };
                agg.rename(partitioned.name());
                Ok(agg)
            },
            _ => unimplemented!(),
        }
    }
}

pub struct AggQuantileExpr {
    pub(crate) input: Arc<dyn PhysicalExpr>,
    pub(crate) quantile: Arc<dyn PhysicalExpr>,
    pub(crate) interpol: QuantileInterpolOptions,
}

impl AggQuantileExpr {
    pub fn new(
        input: Arc<dyn PhysicalExpr>,
        quantile: Arc<dyn PhysicalExpr>,
        interpol: QuantileInterpolOptions,
    ) -> Self {
        Self {
            input,
            quantile,
            interpol,
        }
    }

    fn get_quantile(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<f64> {
        let quantile = self.quantile.evaluate(df, state)?;
        polars_ensure!(quantile.len() <= 1, ComputeError:
            "polars only supports computing a single quantile; \
            make sure the 'quantile' expression input produces a single quantile"
        );
        quantile.get(0).unwrap().try_extract()
    }
}

impl PhysicalExpr for AggQuantileExpr {
    fn as_expression(&self) -> Option<&Expr> {
        None
    }

    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Series> {
        let input = self.input.evaluate(df, state)?;
        let quantile = self.get_quantile(df, state)?;
        input
            .quantile_reduce(quantile, self.interpol)
            .map(|sc| sc.into_series(input.name()))
    }
    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupsProxy,
        state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        let mut ac = self.input.evaluate_on_groups(df, groups, state)?;
        // don't change names by aggregations as is done in polars-core
        let keep_name = ac.series().name().to_string();

        let quantile = self.get_quantile(df, state)?;

        // SAFETY:
        // groups are in bounds
        let mut agg = unsafe {
            ac.flat_naive()
                .into_owned()
                .agg_quantile(ac.groups(), quantile, self.interpol)
        };
        agg.rename(&keep_name);
        Ok(AggregationContext::from_agg_state(
            AggregatedScalar(agg),
            Cow::Borrowed(groups),
        ))
    }

    fn to_field(&self, input_schema: &Schema) -> PolarsResult<Field> {
        self.input.to_field(input_schema)
    }
}

/// Simple wrapper to parallelize functions that can be divided over threads aggregated and
/// finally aggregated in the main thread. This can be done for sum, min, max, etc.
fn parallel_op_series<F>(f: F, s: Series, allow_threading: bool) -> PolarsResult<Series>
where
    F: Fn(Series) -> PolarsResult<Series> + Send + Sync,
{
    // set during debug low so
    // we mimic production size data behavior
    #[cfg(debug_assertions)]
    let thread_boundary = 0;

    #[cfg(not(debug_assertions))]
    let thread_boundary = 100_000;

    // threading overhead/ splitting work stealing is costly..
    if allow_threading
        || s.len() < thread_boundary
        || POOL.current_thread_has_pending_tasks().unwrap_or(false)
    {
        return f(s);
    }
    let n_threads = POOL.current_num_threads();
    let splits = _split_offsets(s.len(), n_threads);

    let chunks = POOL.install(|| {
        splits
            .into_par_iter()
            .map(|(offset, len)| {
                let s = s.slice(offset as i64, len);
                f(s)
            })
            .collect::<PolarsResult<Vec<_>>>()
    })?;

    let mut iter = chunks.into_iter();
    let first = iter.next().unwrap();
    let dtype = first.dtype();
    let out = iter.fold(first.to_physical_repr().into_owned(), |mut acc, s| {
        acc.append(&s.to_physical_repr()).unwrap();
        acc
    });

    unsafe { f(out.cast_unchecked(dtype).unwrap()) }
}
