use std::borrow::Cow;
use std::sync::{Arc, Mutex};

use arrow::array::{Array, ListArray};
use polars_core::POOL;
use polars_core::chunked_array::builder::AnonymousOwnedListBuilder;
use polars_core::chunked_array::from_iterator_par::ChunkedCollectParIterExt;
use polars_core::error::{PolarsResult, polars_ensure};
use polars_core::frame::DataFrame;
use polars_core::prelude::{
    AnyValue, ChunkCast, ChunkNestingUtils, Column, CompatLevel, DataType, Field, GroupPositions,
    GroupsType, IntoColumn, ListBuilderTrait, ListChunked,
};
use polars_core::schema::Schema;
use polars_core::series::Series;
use polars_core::utils::CustomIterTools;
use polars_plan::dsl::{EvalVariant, Expr};
use polars_plan::plans::ExprPushdownGroup;
use polars_utils::IdxSize;
use polars_utils::pl_str::PlSmallStr;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use super::{AggState, AggregationContext, PhysicalExpr};
use crate::state::ExecutionState;

#[derive(Clone)]
pub struct EvalExpr {
    input: Arc<dyn PhysicalExpr>,
    evaluation: Arc<dyn PhysicalExpr>,
    variant: EvalVariant,
    expr: Expr,
    allow_threading: bool,
    //  `output_field_with_ctx`` accounts for the aggregation context, if any
    // It will 'auto-implode/expplode' if needed.
    output_field_with_ctx: Field,
    // `non_aggregated_output_dtype`` ignores any aggregation context
    non_aggregated_output_dtype: DataType,
    is_scalar: bool,
    pd_group: ExprPushdownGroup,
    evaluation_is_scalar: bool,
}

fn offsets_to_groups(offsets: &[i64]) -> Option<GroupPositions> {
    let mut start = offsets[0];
    let end = *offsets.last().unwrap();
    if IdxSize::try_from(end - start).is_err() {
        return None;
    }
    let groups = offsets
        .iter()
        .skip(1)
        .map(|end| {
            let offset = start as IdxSize;
            let len = (*end - start) as IdxSize;
            start = *end;
            [offset, len]
        })
        .collect();
    Some(
        GroupsType::Slice {
            groups,
            rolling: false,
        }
        .into_sliceable(),
    )
}

impl EvalExpr {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        input: Arc<dyn PhysicalExpr>,
        evaluation: Arc<dyn PhysicalExpr>,
        variant: EvalVariant,
        expr: Expr,
        allow_threading: bool,
        output_field_with_ctx: Field,
        non_aggregated_output_dtype: DataType,
        is_scalar: bool,
        pd_group: ExprPushdownGroup,
        evaluation_is_scalar: bool,
    ) -> Self {
        Self {
            input,
            evaluation,
            variant,
            expr,
            allow_threading,
            output_field_with_ctx,
            non_aggregated_output_dtype,
            is_scalar,
            pd_group,
            evaluation_is_scalar,
        }
    }

    fn run_elementwise_on_values(
        &self,
        lst: &ListChunked,
        state: &ExecutionState,
    ) -> PolarsResult<Column> {
        if lst.chunks().is_empty() {
            return Ok(Column::new_empty(
                self.output_field_with_ctx.name.clone(),
                &self.non_aggregated_output_dtype,
            ));
        }

        let lst = lst
            .trim_lists_to_normalized_offsets()
            .map_or(Cow::Borrowed(lst), Cow::Owned);

        let output_arrow_dtype = self
            .non_aggregated_output_dtype
            .clone()
            .to_arrow(CompatLevel::newest());
        let output_arrow_dtype_physical = output_arrow_dtype.underlying_physical_type();

        let apply_to_chunk = |arr: &dyn Array| {
            let arr: &ListArray<i64> = arr.as_any().downcast_ref().unwrap();

            let values = unsafe {
                Series::from_chunks_and_dtype_unchecked(
                    PlSmallStr::EMPTY,
                    vec![arr.values().clone()],
                    lst.inner_dtype(),
                )
            };

            let df = values.into_frame();

            self.evaluation.evaluate(&df, state).map(|values| {
                let values = values.take_materialized_series().rechunk().chunks()[0].clone();

                ListArray::<i64>::new(
                    output_arrow_dtype_physical.clone(),
                    arr.offsets().clone(),
                    values,
                    arr.validity().cloned(),
                )
                .boxed()
            })
        };

        let chunks = if self.allow_threading && lst.chunks().len() > 1 {
            POOL.install(|| {
                lst.chunks()
                    .into_par_iter()
                    .map(|x| apply_to_chunk(&**x))
                    .collect::<PolarsResult<Vec<Box<dyn Array>>>>()
            })?
        } else {
            lst.chunks()
                .iter()
                .map(|x| apply_to_chunk(&**x))
                .collect::<PolarsResult<Vec<Box<dyn Array>>>>()?
        };

        Ok(unsafe {
            ListChunked::from_chunks(self.output_field_with_ctx.name.clone(), chunks)
                .cast_unchecked(&self.non_aggregated_output_dtype)
                .unwrap()
        }
        .into_column())
    }

    fn run_per_sublist(&self, lst: &ListChunked, state: &ExecutionState) -> PolarsResult<Column> {
        let mut err = None;
        let mut ca: ListChunked = if self.allow_threading {
            let m_err = Mutex::new(None);
            let ca: ListChunked = POOL.install(|| {
                lst.par_iter()
                    .map(|opt_s| {
                        opt_s.and_then(|s| {
                            let df = s.into_frame();
                            let out = self.evaluation.evaluate(&df, state);
                            match out {
                                Ok(s) => Some(s.take_materialized_series()),
                                Err(e) => {
                                    *m_err.lock().unwrap() = Some(e);
                                    None
                                },
                            }
                        })
                    })
                    .collect_ca_with_dtype(
                        PlSmallStr::EMPTY,
                        self.non_aggregated_output_dtype.clone(),
                    )
            });
            err = m_err.into_inner().unwrap();
            ca
        } else {
            let mut df_container = DataFrame::empty();

            lst.into_iter()
                .map(|s| {
                    s.and_then(|s| unsafe {
                        df_container.with_column_unchecked(s.into_column());
                        let out = self.evaluation.evaluate(&df_container, state);
                        df_container.clear_columns();
                        match out {
                            Ok(s) => Some(s.take_materialized_series()),
                            Err(e) => {
                                err = Some(e);
                                None
                            },
                        }
                    })
                })
                .collect_trusted()
        };
        if let Some(err) = err {
            return Err(err);
        }

        ca.rename(lst.name().clone());

        // Cast may still be required in some cases, e.g. for an empty frame when running single-threaded
        if ca.dtype() != &self.non_aggregated_output_dtype {
            ca.cast(&self.non_aggregated_output_dtype).map(Column::from)
        } else {
            Ok(ca.into_column())
        }
    }

    fn run_on_group_by_engine(
        &self,
        lst: &ListChunked,
        state: &ExecutionState,
    ) -> PolarsResult<Column> {
        let lst = lst.rechunk();
        let arr = lst.downcast_as_array();
        let groups = offsets_to_groups(arr.offsets()).unwrap();

        // List elements in a series.
        let values = Series::try_from((PlSmallStr::EMPTY, arr.values().clone())).unwrap();
        let inner_dtype = lst.inner_dtype();
        // SAFETY:
        // Invariant in List means values physicals can be cast to inner dtype
        let values = unsafe { values.from_physical_unchecked(inner_dtype).unwrap() };

        let df_context = values.into_frame();

        let mut ac = self
            .evaluation
            .evaluate_on_groups(&df_context, &groups, state)?;
        let out = match ac.agg_state() {
            AggState::AggregatedScalar(_) => {
                let out = ac.aggregated();
                out.as_list().into_column()
            },
            _ => ac.aggregated(),
        };
        Ok(out
            .with_name(self.output_field_with_ctx.name.clone())
            .into_column())
    }

    fn evaluate_on_list_chunked(
        &self,
        lst: &ListChunked,
        state: &ExecutionState,
    ) -> PolarsResult<Column> {
        let fits_idx_size = lst.get_inner().len() < (IdxSize::MAX as usize);
        if match self.pd_group {
            ExprPushdownGroup::Pushable => true,
            ExprPushdownGroup::Fallible => !lst.has_nulls(),
            ExprPushdownGroup::Barrier => false,
        } && !self.evaluation_is_scalar
        {
            self.run_elementwise_on_values(lst, state)
        } else if fits_idx_size && lst.null_count() == 0 && self.evaluation_is_scalar {
            self.run_on_group_by_engine(lst, state)
        } else {
            self.run_per_sublist(lst, state)
        }
    }

    fn evaluate_cumulative_eval(
        &self,
        input: &Series,
        min_samples: usize,
        state: &ExecutionState,
    ) -> PolarsResult<Series> {
        let finish = |out: Series| {
            polars_ensure!(
                out.len() <= 1,
                ComputeError:
                "expected single value, got a result with length {}, {:?}",
                out.len(), out,
            );
            Ok(out.get(0).unwrap().into_static())
        };

        let input = input.clone().with_name(PlSmallStr::EMPTY);
        let avs = if self.allow_threading {
            POOL.install(|| {
                (1..input.len() + 1)
                    .into_par_iter()
                    .map(|len| {
                        let c = input.slice(0, len);
                        if (len - c.null_count()) >= min_samples {
                            let df = c.into_frame();
                            let out = self
                                .evaluation
                                .evaluate(&df, state)?
                                .take_materialized_series();
                            finish(out)
                        } else {
                            Ok(AnyValue::Null)
                        }
                    })
                    .collect::<PolarsResult<Vec<_>>>()
            })?
        } else {
            let mut df_container = DataFrame::empty();
            (1..input.len() + 1)
                .map(|len| {
                    let c = input.slice(0, len);
                    if (len - c.null_count()) >= min_samples {
                        unsafe {
                            df_container.with_column_unchecked(c.into_column());
                            let out = self
                                .evaluation
                                .evaluate(&df_container, state)?
                                .take_materialized_series();
                            df_container.clear_columns();
                            finish(out)
                        }
                    } else {
                        Ok(AnyValue::Null)
                    }
                })
                .collect::<PolarsResult<Vec<_>>>()?
        };

        Series::from_any_values_and_dtype(
            self.output_field_with_ctx.name().clone(),
            &avs,
            &self.non_aggregated_output_dtype,
            true,
        )
    }
}

impl PhysicalExpr for EvalExpr {
    fn as_expression(&self) -> Option<&Expr> {
        Some(&self.expr)
    }

    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Column> {
        let input = self.input.evaluate(df, state)?;
        match self.variant {
            EvalVariant::List => {
                let lst = input.list()?;
                self.evaluate_on_list_chunked(lst, state)
            },
            EvalVariant::Cumulative { min_samples } => self
                .evaluate_cumulative_eval(input.as_materialized_series(), min_samples, state)
                .map(Column::from),
        }
    }

    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupPositions,
        state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        let mut input = self.input.evaluate_on_groups(df, groups, state)?;
        match self.variant {
            EvalVariant::List => {
                let out = self.evaluate_on_list_chunked(input.get_values().list()?, state)?;
                input.with_values(out, false, Some(&self.expr))?;
            },
            EvalVariant::Cumulative { min_samples } => {
                let mut builder = AnonymousOwnedListBuilder::new(
                    self.output_field_with_ctx.name().clone(),
                    input.groups().len(),
                    Some(self.non_aggregated_output_dtype.clone()),
                );
                for group in input.iter_groups(false) {
                    match group {
                        None => {},
                        Some(group) => {
                            let out =
                                self.evaluate_cumulative_eval(group.as_ref(), min_samples, state)?;
                            builder.append_series(&out)?;
                        },
                    }
                }

                input.with_values(builder.finish().into_column(), true, Some(&self.expr))?;
            },
        }
        Ok(input)
    }

    fn to_field(&self, _input_schema: &Schema) -> PolarsResult<Field> {
        Ok(self.output_field_with_ctx.clone())
    }

    fn is_scalar(&self) -> bool {
        self.is_scalar
    }
}
