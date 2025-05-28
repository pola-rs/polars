use std::borrow::Cow;
use std::sync::{Arc, Mutex};

use arrow::array::{Array, ListArray};
use polars_core::POOL;
use polars_core::chunked_array::from_iterator_par::ChunkedCollectParIterExt;
use polars_core::error::PolarsResult;
use polars_core::frame::DataFrame;
use polars_core::prelude::{
    ChunkCast, ChunkNestingUtils, Column, CompatLevel, Field, GroupPositions, GroupsType,
    IntoColumn, ListChunked,
};
use polars_core::schema::Schema;
use polars_core::series::Series;
use polars_core::utils::CustomIterTools;
use polars_plan::dsl::Expr;
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
    expr: Expr,
    allow_threading: bool,
    output_field: Field,
    is_scalar: bool,
    pd_group: ExprPushdownGroup,
    evaluation_is_scalar: bool,
    is_user_apply: bool,
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
        expr: Expr,
        allow_threading: bool,
        output_field: Field,
        is_scalar: bool,
        pd_group: ExprPushdownGroup,
        evaluation_is_scalar: bool,
        is_user_apply: bool,
    ) -> Self {
        Self {
            input,
            evaluation,
            expr,
            allow_threading,
            output_field,
            is_scalar,
            pd_group,
            evaluation_is_scalar,
            is_user_apply,
        }
    }

    fn run_elementwise_on_values(
        &self,
        lst: &ListChunked,
        state: &ExecutionState,
    ) -> PolarsResult<Column> {
        if lst.chunks().is_empty() {
            return Ok(Column::new_empty(
                self.output_field.name.clone(),
                &self.output_field.dtype,
            ));
        }

        let lst = lst
            .trim_lists_to_normalized_offsets()
            .map_or(Cow::Borrowed(lst), Cow::Owned);

        let output_arrow_dtype = self
            .output_field
            .dtype()
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
            ListChunked::from_chunks(self.output_field.name.clone(), chunks)
                .cast_unchecked(self.output_field.dtype())
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
                    .collect_ca_with_dtype(PlSmallStr::EMPTY, self.output_field.dtype.clone())
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

        if ca.dtype() != self.output_field.dtype() {
            ca.cast(self.output_field.dtype()).map(Column::from)
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
        Ok(out.with_name(self.output_field.name.clone()).into_column())
    }

    fn evaluate_on_list_chunked(
        &self,
        lst: &ListChunked,
        state: &ExecutionState,
    ) -> PolarsResult<Column> {
        let fits_idx_size = lst.get_inner().len() < (IdxSize::MAX as usize);
        // If a users passes a return type to `apply`, e.g. `return_dtype=pl.Int64`,
        // this fails as the list builder expects `List<Int64>`, so let's skip that for now.
        let is_user_apply = self.is_user_apply;

        if match self.pd_group {
            ExprPushdownGroup::Pushable => true,
            ExprPushdownGroup::Fallible => !lst.has_nulls(),
            ExprPushdownGroup::Barrier => false,
        } && !self.evaluation_is_scalar
        {
            self.run_elementwise_on_values(lst, state)
        } else if fits_idx_size
            && lst.null_count() == 0
            && !is_user_apply
            && self.evaluation_is_scalar
        {
            self.run_on_group_by_engine(lst, state)
        } else {
            self.run_per_sublist(lst, state)
        }
    }
}

impl PhysicalExpr for EvalExpr {
    fn as_expression(&self) -> Option<&Expr> {
        Some(&self.expr)
    }

    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Column> {
        let input = self.input.evaluate(df, state)?;
        let lst = input.list()?;
        self.evaluate_on_list_chunked(lst, state)
    }

    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupPositions,
        state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        let mut input = self.input.evaluate_on_groups(df, groups, state)?;
        let out = self.evaluate_on_list_chunked(input.get_values().list()?, state)?;
        input.with_values(out, false, Some(&self.expr))?;
        Ok(input)
    }

    fn to_field(&self, _input_schema: &Schema) -> PolarsResult<Field> {
        Ok(self.output_field.clone())
    }

    fn is_scalar(&self) -> bool {
        self.is_scalar
    }
}
