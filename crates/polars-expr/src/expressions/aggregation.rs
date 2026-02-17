use std::borrow::Cow;

use arrow::legacy::utils::CustomIterTools;
use polars_compute::rolling::QuantileMethod;
use polars_core::POOL;
use polars_core::prelude::*;
use polars_core::series::IsSorted;
use polars_core::utils::{_split_offsets, NoNull};
use polars_ops::prelude::ArgAgg;
#[cfg(feature = "propagate_nans")]
use polars_ops::prelude::nan_propagating_aggregate;
use polars_utils::itertools::Itertools;
use rayon::prelude::*;

use super::*;
use crate::expressions::AggState::AggregatedScalar;
use crate::expressions::{AggState, AggregationContext, PhysicalExpr, UpdateGroups};
use crate::reduce::GroupedReduction;

#[derive(Debug, Clone, Copy)]
pub struct AggregationType {
    pub(crate) groupby: GroupByMethod,
    pub(crate) allow_threading: bool,
}

pub(crate) struct AggregationExpr {
    pub(crate) input: Arc<dyn PhysicalExpr>,
    pub(crate) agg_type: AggregationType,
    pub(crate) output_field: Field,
}

impl AggregationExpr {
    pub fn new(
        expr: Arc<dyn PhysicalExpr>,
        agg_type: AggregationType,
        output_field: Field,
    ) -> Self {
        Self {
            input: expr,
            agg_type,
            output_field,
        }
    }
}

impl PhysicalExpr for AggregationExpr {
    fn as_expression(&self) -> Option<&Expr> {
        None
    }

    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Column> {
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
            GroupByMethod::Min => match s.is_sorted_flag() {
                IsSorted::Ascending | IsSorted::Descending => {
                    s.min_reduce().map(|sc| sc.into_column(s.name().clone()))
                },
                IsSorted::Not => parallel_op_columns(
                    |s| s.min_reduce().map(|sc| sc.into_column(s.name().clone())),
                    s,
                    allow_threading,
                ),
            },
            #[cfg(feature = "propagate_nans")]
            GroupByMethod::NanMin => parallel_op_columns(
                |s| {
                    Ok(polars_ops::prelude::nan_propagating_aggregate::nan_min_s(
                        s.as_materialized_series(),
                        s.name().clone(),
                    )
                    .into_column())
                },
                s,
                allow_threading,
            ),
            #[cfg(not(feature = "propagate_nans"))]
            GroupByMethod::NanMin => {
                panic!("activate 'propagate_nans' feature")
            },
            GroupByMethod::Max => match s.is_sorted_flag() {
                IsSorted::Ascending | IsSorted::Descending => {
                    s.max_reduce().map(|sc| sc.into_column(s.name().clone()))
                },
                IsSorted::Not => parallel_op_columns(
                    |s| s.max_reduce().map(|sc| sc.into_column(s.name().clone())),
                    s,
                    allow_threading,
                ),
            },
            #[cfg(feature = "propagate_nans")]
            GroupByMethod::NanMax => parallel_op_columns(
                |s| {
                    Ok(polars_ops::prelude::nan_propagating_aggregate::nan_max_s(
                        s.as_materialized_series(),
                        s.name().clone(),
                    )
                    .into_column())
                },
                s,
                allow_threading,
            ),
            #[cfg(not(feature = "propagate_nans"))]
            GroupByMethod::NanMax => {
                panic!("activate 'propagate_nans' feature")
            },
            GroupByMethod::Median => s.median_reduce().map(|sc| sc.into_column(s.name().clone())),
            GroupByMethod::Mean => s.mean_reduce().map(|sc| sc.into_column(s.name().clone())),
            GroupByMethod::First => Ok(if s.is_empty() {
                Column::full_null(s.name().clone(), 1, s.dtype())
            } else {
                s.head(Some(1))
            }),
            GroupByMethod::FirstNonNull => Ok(s
                .as_materialized_series_maintain_scalar()
                .first_non_null()
                .into_column(s.name().clone())),
            GroupByMethod::Last => Ok(if s.is_empty() {
                Column::full_null(s.name().clone(), 1, s.dtype())
            } else {
                s.tail(Some(1))
            }),
            GroupByMethod::LastNonNull => Ok(s
                .as_materialized_series_maintain_scalar()
                .last_non_null()
                .into_column(s.name().clone())),
            GroupByMethod::Item { allow_empty } => Ok(match s.len() {
                0 if allow_empty => Column::full_null(s.name().clone(), 1, s.dtype()),
                1 => s,
                n => polars_bail!(item_agg_count_not_one = n, allow_empty = allow_empty),
            }),
            GroupByMethod::Sum => parallel_op_columns(
                |s| s.sum_reduce().map(|sc| sc.into_column(s.name().clone())),
                s,
                allow_threading,
            ),
            GroupByMethod::Groups => unreachable!(),
            GroupByMethod::NUnique => s.n_unique().map(|count| {
                IdxCa::from_slice(s.name().clone(), &[count as IdxSize]).into_column()
            }),
            GroupByMethod::Count { include_nulls } => {
                let count = s.len() - s.null_count() * !include_nulls as usize;

                Ok(IdxCa::from_slice(s.name().clone(), &[count as IdxSize]).into_column())
            },
            GroupByMethod::Implode => s.implode().map(|ca| ca.into_column()),
            GroupByMethod::Std(ddof) => s
                .std_reduce(ddof)
                .map(|sc| sc.into_column(s.name().clone())),
            GroupByMethod::Var(ddof) => s
                .var_reduce(ddof)
                .map(|sc| sc.into_column(s.name().clone())),
            GroupByMethod::Quantile(_, _) => unimplemented!(),
            GroupByMethod::ArgMin => {
                let opt = s.as_materialized_series().arg_min();
                Ok(opt.map_or_else(
                    || Column::full_null(s.name().clone(), 1, &IDX_DTYPE),
                    |idx| {
                        Column::new_scalar(
                            s.name().clone(),
                            Scalar::new_idxsize(idx.try_into().unwrap()),
                            1,
                        )
                    },
                ))
            },
            GroupByMethod::ArgMax => {
                let opt = s.as_materialized_series().arg_max();
                Ok(opt.map_or_else(
                    || Column::full_null(s.name().clone(), 1, &IDX_DTYPE),
                    |idx| {
                        Column::new_scalar(
                            s.name().clone(),
                            Scalar::new_idxsize(idx.try_into().unwrap()),
                            1,
                        )
                    },
                ))
            },
        }
    }

    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupPositions,
        state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        let mut ac = self.input.evaluate_on_groups(df, groups, state)?;

        // don't change names by aggregations as is done in polars-core
        let keep_name = ac.get_values().name().clone();

        if let AggState::LiteralScalar(c) = &mut ac.state {
            *c = self.evaluate(df, state)?;
            return Ok(ac);
        }

        // AggregatedScalar has no defined group structure. We fix it up here, so that we can
        // reliably call `agg_*` functions with the groups.
        ac.set_groups_for_undefined_agg_states();

        // SAFETY:
        // groups must always be in bounds.
        let out = unsafe {
            match self.agg_type.groupby {
                GroupByMethod::Min => {
                    let (c, groups) = ac.get_final_aggregation();
                    let agg_c = c.agg_min(&groups);
                    AggregatedScalar(agg_c.with_name(keep_name))
                },
                GroupByMethod::Max => {
                    let (c, groups) = ac.get_final_aggregation();
                    let agg_c = c.agg_max(&groups);
                    AggregatedScalar(agg_c.with_name(keep_name))
                },
                GroupByMethod::ArgMin => {
                    let (c, groups) = ac.get_final_aggregation();
                    let agg_c = c.agg_arg_min(&groups);
                    AggregatedScalar(agg_c.with_name(keep_name))
                },
                GroupByMethod::ArgMax => {
                    let (c, groups) = ac.get_final_aggregation();
                    let agg_c = c.agg_arg_max(&groups);
                    AggregatedScalar(agg_c.with_name(keep_name))
                },
                GroupByMethod::Median => {
                    let (c, groups) = ac.get_final_aggregation();
                    let agg_c = c.agg_median(&groups);
                    AggregatedScalar(agg_c.with_name(keep_name))
                },
                GroupByMethod::Mean => {
                    let (c, groups) = ac.get_final_aggregation();
                    let agg_c = c.agg_mean(&groups);
                    AggregatedScalar(agg_c.with_name(keep_name))
                },
                GroupByMethod::Sum => {
                    let (c, groups) = ac.get_final_aggregation();
                    let agg_c = c.agg_sum(&groups);
                    AggregatedScalar(agg_c.with_name(keep_name))
                },
                GroupByMethod::Count { include_nulls } => {
                    if include_nulls || ac.get_values().null_count() == 0 {
                        // a few fast paths that prevent materializing new groups
                        match ac.update_groups {
                            UpdateGroups::WithSeriesLen => {
                                let list = ac
                                    .get_values()
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
                                s.rename(keep_name);
                                AggregatedScalar(s.into_column())
                            },
                            UpdateGroups::WithGroupsLen => {
                                // no need to update the groups
                                // we can just get the attribute, because we only need the length,
                                // not the correct order
                                let mut ca = ac.groups.group_count();
                                ca.rename(keep_name);
                                AggregatedScalar(ca.into_column())
                            },
                            // materialize groups
                            _ => {
                                let mut ca = ac.groups().group_count();
                                ca.rename(keep_name);
                                AggregatedScalar(ca.into_column())
                            },
                        }
                    } else {
                        // TODO: optimize this/and write somewhere else.
                        match ac.agg_state() {
                            AggState::LiteralScalar(_) => unreachable!(),
                            AggState::AggregatedScalar(c) => AggregatedScalar(
                                c.is_not_null().cast(&IDX_DTYPE).unwrap().into_column(),
                            ),
                            AggState::AggregatedList(s) => {
                                let ca = s.list()?;
                                let out: IdxCa = ca
                                    .into_iter()
                                    .map(|opt_s| {
                                        opt_s
                                            .map(|s| s.len() as IdxSize - s.null_count() as IdxSize)
                                    })
                                    .collect();
                                AggregatedScalar(out.into_column().with_name(keep_name))
                            },
                            AggState::NotAggregated(s) => {
                                let s = s.clone();
                                let groups = ac.groups();
                                let out: IdxCa = if matches!(s.dtype(), &DataType::Null) {
                                    IdxCa::full(s.name().clone(), 0, groups.len())
                                } else {
                                    match groups.as_ref().as_ref() {
                                        GroupsType::Idx(idx) => {
                                            let s = s.rechunk();
                                            // @scalar-opt
                                            // @partition-opt
                                            let array = &s.as_materialized_series().chunks()[0];
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
                                                .collect_ca_trusted_with_dtype(keep_name, IDX_DTYPE)
                                        },
                                        GroupsType::Slice { groups, .. } => {
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
                                                .collect_ca_trusted_with_dtype(keep_name, IDX_DTYPE)
                                        },
                                    }
                                };
                                AggregatedScalar(out.into_column())
                            },
                        }
                    }
                },
                GroupByMethod::First => {
                    let (s, groups) = ac.get_final_aggregation();
                    let agg_s = s.agg_first(&groups);
                    AggregatedScalar(agg_s.with_name(keep_name))
                },
                GroupByMethod::FirstNonNull => {
                    let (s, groups) = ac.get_final_aggregation();
                    let agg_s = s.agg_first_non_null(&groups);
                    AggregatedScalar(agg_s.with_name(keep_name))
                },
                GroupByMethod::Last => {
                    let (s, groups) = ac.get_final_aggregation();
                    let agg_s = s.agg_last(&groups);
                    AggregatedScalar(agg_s.with_name(keep_name))
                },
                GroupByMethod::LastNonNull => {
                    let (s, groups) = ac.get_final_aggregation();
                    let agg_s = s.agg_last_non_null(&groups);
                    AggregatedScalar(agg_s.with_name(keep_name))
                },
                GroupByMethod::Item { allow_empty } => {
                    let (s, groups) = ac.get_final_aggregation();
                    for gc in groups.group_count().iter() {
                        match gc {
                            Some(0) if allow_empty => continue,
                            None | Some(1) => continue,
                            Some(n) => {
                                polars_bail!(item_agg_count_not_one = n, allow_empty = allow_empty);
                            },
                        }
                    }
                    let agg_s = s.agg_first(&groups);
                    AggregatedScalar(agg_s.with_name(keep_name))
                },
                GroupByMethod::NUnique => {
                    let (s, groups) = ac.get_final_aggregation();
                    let agg_s = s.agg_n_unique(&groups);
                    AggregatedScalar(agg_s.with_name(keep_name))
                },
                GroupByMethod::Implode => AggregatedScalar(match ac.agg_state() {
                    AggState::LiteralScalar(_) => unreachable!(), // handled above
                    AggState::AggregatedScalar(c) => c.as_list().into_column(),
                    AggState::NotAggregated(_) | AggState::AggregatedList(_) => ac.aggregated(),
                }),
                GroupByMethod::Groups => {
                    let mut column: ListChunked = ac.groups().as_list_chunked();
                    column.rename(keep_name);
                    AggregatedScalar(column.into_column())
                },
                GroupByMethod::Std(ddof) => {
                    let (c, groups) = ac.get_final_aggregation();
                    let agg_c = c.agg_std(&groups, ddof);
                    AggregatedScalar(agg_c.with_name(keep_name))
                },
                GroupByMethod::Var(ddof) => {
                    let (c, groups) = ac.get_final_aggregation();
                    let agg_c = c.agg_var(&groups, ddof);
                    AggregatedScalar(agg_c.with_name(keep_name))
                },
                GroupByMethod::Quantile(_, _) => {
                    // implemented explicitly in AggQuantile struct
                    unimplemented!()
                },
                GroupByMethod::NanMin => {
                    #[cfg(feature = "propagate_nans")]
                    {
                        let (c, groups) = ac.get_final_aggregation();
                        let agg_c = if c.dtype().is_float() {
                            nan_propagating_aggregate::group_agg_nan_min_s(
                                c.as_materialized_series(),
                                &groups,
                            )
                            .into_column()
                        } else {
                            c.agg_min(&groups)
                        };
                        AggregatedScalar(agg_c.with_name(keep_name))
                    }
                    #[cfg(not(feature = "propagate_nans"))]
                    {
                        panic!("activate 'propagate_nans' feature")
                    }
                },
                GroupByMethod::NanMax => {
                    #[cfg(feature = "propagate_nans")]
                    {
                        let (c, groups) = ac.get_final_aggregation();
                        let agg_c = if c.dtype().is_float() {
                            nan_propagating_aggregate::group_agg_nan_max_s(
                                c.as_materialized_series(),
                                &groups,
                            )
                            .into_column()
                        } else {
                            c.agg_max(&groups)
                        };
                        AggregatedScalar(agg_c.with_name(keep_name))
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

    fn to_field(&self, _input_schema: &Schema) -> PolarsResult<Field> {
        Ok(self.output_field.clone())
    }

    fn is_scalar(&self) -> bool {
        true
    }
}

pub struct AggQuantileExpr {
    input: Arc<dyn PhysicalExpr>,
    quantile: Arc<dyn PhysicalExpr>,
    method: QuantileMethod,
}

impl AggQuantileExpr {
    pub fn new(
        input: Arc<dyn PhysicalExpr>,
        quantile: Arc<dyn PhysicalExpr>,
        method: QuantileMethod,
    ) -> Self {
        Self {
            input,
            quantile,
            method,
        }
    }
}

impl PhysicalExpr for AggQuantileExpr {
    fn as_expression(&self) -> Option<&Expr> {
        None
    }

    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Column> {
        let input = self.input.evaluate(df, state)?;

        let quantile = self.quantile.evaluate(df, state)?;

        polars_ensure!(quantile.len() <= 1, ComputeError:
            "polars does not support varying quantiles yet, \
            make sure the 'quantile' expression input produces a single quantile or a list of quantiles"
        );

        let s = quantile.as_materialized_series();

        match s.dtype() {
            DataType::List(_) => {
                let list = s.list()?;
                let inner_s = list.get_as_series(0).unwrap();
                if inner_s.has_nulls() {
                    polars_bail!(ComputeError: "quantile expression contains null values");
                }

                let v: Vec<f64> = inner_s
                    .cast(&DataType::Float64)?
                    .f64()?
                    .into_no_null_iter()
                    .collect();

                input
                    .quantiles_reduce(&v, self.method)
                    .map(|sc| sc.into_column(input.name().clone()))
            },
            _ => {
                let q: f64 = quantile.get(0).unwrap().try_extract()?;
                input
                    .quantile_reduce(q, self.method)
                    .map(|sc| sc.into_column(input.name().clone()))
            },
        }
    }

    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupPositions,
        state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        let mut ac = self.input.evaluate_on_groups(df, groups, state)?;

        // AggregatedScalar has no defined group structure. We fix it up here, so that we can
        // reliably call `agg_quantile` functions with the groups.
        ac.set_groups_for_undefined_agg_states();

        // don't change names by aggregations as is done in polars-core
        let keep_name = ac.get_values().name().clone();

        let quantile_column = self.quantile.evaluate(df, state)?;
        polars_ensure!(quantile_column.len() <= 1, ComputeError:
            "polars only supports computing a single quantile in a groupby aggregation context"
        );
        let quantile: f64 = quantile_column.get(0).unwrap().try_extract()?;

        if let AggState::LiteralScalar(c) = &mut ac.state {
            *c = c
                .quantile_reduce(quantile, self.method)?
                .into_column(keep_name);
            return Ok(ac);
        }

        // SAFETY:
        // groups are in bounds
        let mut agg = unsafe {
            ac.flat_naive()
                .into_owned()
                .agg_quantile(ac.groups(), quantile, self.method)
        };
        agg.rename(keep_name);
        Ok(AggregationContext::from_agg_state(
            AggregatedScalar(agg),
            Cow::Borrowed(groups),
        ))
    }

    fn to_field(&self, input_schema: &Schema) -> PolarsResult<Field> {
        // If the quantile expression is a literal that yields a list of floats,
        // the aggregation returns a list of quantiles (one list per row/group).
        // In that case, report `List(Float64)` as the output field.
        let input_field = self.input.to_field(input_schema)?;
        match self.quantile.to_field(input_schema) {
            Ok(qf) => match qf.dtype() {
                DataType::List(inner) => {
                    if inner.is_float() {
                        Ok(Field::new(
                            input_field.name().clone(),
                            DataType::List(Box::new(DataType::Float64)),
                        ))
                    } else {
                        // fallback to input field
                        Ok(input_field)
                    }
                },
                _ => Ok(input_field),
            },
            Err(_) => Ok(input_field),
        }
    }

    fn is_scalar(&self) -> bool {
        true
    }
}

pub struct AggMinMaxByExpr {
    input: Arc<dyn PhysicalExpr>,
    by: Arc<dyn PhysicalExpr>,
    is_max_by: bool,
}

impl AggMinMaxByExpr {
    pub fn new_min_by(input: Arc<dyn PhysicalExpr>, by: Arc<dyn PhysicalExpr>) -> Self {
        Self {
            input,
            by,
            is_max_by: false,
        }
    }

    pub fn new_max_by(input: Arc<dyn PhysicalExpr>, by: Arc<dyn PhysicalExpr>) -> Self {
        Self {
            input,
            by,
            is_max_by: true,
        }
    }
}

impl PhysicalExpr for AggMinMaxByExpr {
    fn as_expression(&self) -> Option<&Expr> {
        None
    }

    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Column> {
        let input = self.input.evaluate(df, state)?;
        let by = self.by.evaluate(df, state)?;
        let name = if self.is_max_by { "max_by" } else { "min_by" };
        polars_ensure!(
            input.len() == by.len(),
            ShapeMismatch: "'by' column in {} expression has incorrect length: expected {}, got {}",
            name, input.len(), by.len()
        );
        let arg_extremum = if self.is_max_by {
            by.as_materialized_series_maintain_scalar().arg_max()
        } else {
            by.as_materialized_series_maintain_scalar().arg_min()
        };
        let out = if let Some(idx) = arg_extremum {
            input.slice(idx as i64, 1)
        } else {
            let dtype = input.dtype().clone();
            Column::new_scalar(input.name().clone(), Scalar::null(dtype), 1)
        };
        Ok(out)
    }

    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupPositions,
        state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        let ac = self.input.evaluate_on_groups(df, groups, state)?;
        let ac_by = self.by.evaluate_on_groups(df, groups, state)?;
        assert!(ac.groups.len() == ac_by.groups.len());

        // Don't change names by aggregations as is done in polars-core
        let keep_name = ac.get_values().name().clone();

        let (input_col, input_groups) = ac.get_final_aggregation();
        let (by_col, by_groups) = ac_by.get_final_aggregation();
        GroupsType::check_lengths(&input_groups, &by_groups)?;

        // Dispatch to arg_min/arg_max and then gather
        // SAFETY: Groups are correct.
        let idxs_in_groups = if self.is_max_by {
            unsafe { by_col.agg_arg_max(&by_groups) }
        } else {
            unsafe { by_col.agg_arg_min(&by_groups) }
        };
        let idxs_in_groups: &IdxCa = idxs_in_groups.as_materialized_series().as_ref().as_ref();
        let flat_gather_idxs = match input_groups.as_ref().as_ref() {
            GroupsType::Idx(g) => idxs_in_groups
                .into_no_null_iter()
                .enumerate()
                .map(|(group_idx, idx_in_group)| g.all()[group_idx][idx_in_group as usize])
                .collect_vec(),
            GroupsType::Slice { groups, .. } => idxs_in_groups
                .into_no_null_iter()
                .enumerate()
                .map(|(group_idx, idx_in_group)| groups[group_idx][0] + idx_in_group)
                .collect_vec(),
        };

        // SAFETY: All indices are within input_col's groups.
        let gathered = unsafe { input_col.take_slice_unchecked(&flat_gather_idxs) };
        let agg_state = AggregatedScalar(gathered.with_name(keep_name));
        Ok(AggregationContext::from_agg_state(
            agg_state,
            Cow::Borrowed(groups),
        ))
    }

    fn to_field(&self, input_schema: &Schema) -> PolarsResult<Field> {
        self.input.to_field(input_schema)
    }

    fn is_scalar(&self) -> bool {
        true
    }
}

pub(crate) struct AnonymousAggregationExpr {
    pub(crate) inputs: Vec<Arc<dyn PhysicalExpr>>,
    pub(crate) grouped_reduction: Box<dyn GroupedReduction>,
    pub(crate) output_field: Field,
}

impl AnonymousAggregationExpr {
    pub fn new(
        inputs: Vec<Arc<dyn PhysicalExpr>>,
        grouped_reduction: Box<dyn GroupedReduction>,
        output_field: Field,
    ) -> Self {
        Self {
            inputs,
            grouped_reduction,
            output_field,
        }
    }
}

impl PhysicalExpr for AnonymousAggregationExpr {
    fn as_expression(&self) -> Option<&Expr> {
        None
    }

    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Column> {
        polars_ensure!(
            self.inputs.len() == 1,
            ComputeError: "AnonymousAggregationExpr with more than one input is not supported"
        );

        let col = self.inputs[0].evaluate(df, state)?;
        let mut gr = self.grouped_reduction.new_empty();
        gr.resize(1);
        gr.update_group(&[&col], 0, 0)?;
        let out_series = gr.finalize()?;
        Ok(Column::new(col.name().clone(), out_series))
    }

    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupPositions,
        state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        polars_ensure!(
            self.inputs.len() == 1,
            ComputeError: "AnonymousAggregationExpr with more than one input is not supported"
        );

        let input = &self.inputs[0];
        let mut ac = input.evaluate_on_groups(df, groups, state)?;

        // don't change names by aggregations as is done in polars-core
        let input_column_name = ac.get_values().name().clone();

        if let AggState::LiteralScalar(input_column) = &mut ac.state {
            *input_column = self.evaluate(df, state)?;
            return Ok(ac);
        }

        let (input_column, resolved_groups) = ac.get_final_aggregation();

        let mut gr = self.grouped_reduction.new_empty();
        gr.resize(groups.len() as IdxSize);

        assert!(
            !resolved_groups.is_overlapping(),
            "Aggregating with overlapping groups is a logic error"
        );

        let subset = (0..input_column.len() as IdxSize).collect::<Vec<IdxSize>>();

        let mut group_idxs = Vec::with_capacity(input_column.len());
        match &**resolved_groups {
            GroupsType::Idx(group_indices) => {
                group_idxs.resize(input_column.len(), 0);
                for (group_idx, indices_in_group) in group_indices.all().iter().enumerate() {
                    for pos in indices_in_group.iter() {
                        group_idxs[*pos as usize] = group_idx as IdxSize;
                    }
                }
            },
            GroupsType::Slice { groups, .. } => {
                for (group_idx, [_start, len]) in groups.iter().enumerate() {
                    group_idxs.extend(std::iter::repeat_n(group_idx as IdxSize, *len as usize));
                }
            },
        };
        assert_eq!(group_idxs.len(), input_column.len());

        // `update_groups_subset` needs a single chunk.
        let input_column_rechunked = input_column.rechunk();

        // Single call so no need to resolve ordering.
        let seq_id = 0;

        // SAFETY:
        // - `subset` is in-bounds because it is 0..N
        // - `group_idxs` is in-bounds because we checked that it matches `input_column.len()` *and*
        //   is filled with values <= `input_column.len()` since they are derived from it via
        //   `enumerate`.
        unsafe {
            gr.update_groups_subset(&[&input_column_rechunked], &subset, &group_idxs, seq_id)?;
        }

        let out_series = gr.finalize()?;
        let out = AggregatedScalar(Column::new(input_column_name, out_series));

        Ok(AggregationContext::from_agg_state(out, resolved_groups))
    }

    fn to_field(&self, _input_schema: &Schema) -> PolarsResult<Field> {
        Ok(self.output_field.clone())
    }

    fn is_scalar(&self) -> bool {
        true
    }
}

/// Simple wrapper to parallelize functions that can be divided over threads aggregated and
/// finally aggregated in the main thread. This can be done for sum, min, max, etc.
fn parallel_op_columns<F>(f: F, s: Column, allow_threading: bool) -> PolarsResult<Column>
where
    F: Fn(Column) -> PolarsResult<Column> + Send + Sync,
{
    // set during debug low so
    // we mimic production size data behavior
    #[cfg(debug_assertions)]
    let thread_boundary = 0;

    #[cfg(not(debug_assertions))]
    let thread_boundary = 100_000;

    // threading overhead/ splitting work stealing is costly..

    if !allow_threading
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
    let out = iter.fold(first.to_physical_repr(), |mut acc, s| {
        acc.append(&s.to_physical_repr()).unwrap();
        acc
    });

    unsafe { f(out.from_physical_unchecked(dtype).unwrap()) }
}
