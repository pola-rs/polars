use std::borrow::Cow;

use polars_core::POOL;
use polars_core::chunked_array::builder::get_list_builder;
use polars_core::chunked_array::from_iterator_par::{
    ChunkedCollectParIterExt, try_list_from_par_iter,
};
use polars_core::prelude::*;
use rayon::prelude::*;

use super::*;
use crate::expressions::{
    AggState, AggregationContext, PartitionedAggregation, PhysicalExpr, UpdateGroups,
};

#[derive(Clone)]
pub struct ApplyExpr {
    inputs: Vec<Arc<dyn PhysicalExpr>>,
    function: SpecialEq<Arc<dyn ColumnsUdf>>,
    expr: Expr,
    flags: FunctionFlags,
    function_operates_on_scalar: bool,
    input_schema: SchemaRef,
    allow_threading: bool,
    check_lengths: bool,
    output_field: Field,
}

impl ApplyExpr {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        inputs: Vec<Arc<dyn PhysicalExpr>>,
        function: SpecialEq<Arc<dyn ColumnsUdf>>,
        expr: Expr,
        options: FunctionOptions,
        allow_threading: bool,
        input_schema: SchemaRef,
        output_field: Field,
        function_operates_on_scalar: bool,
    ) -> Self {
        debug_assert!(
            !options.is_length_preserving()
                || !options.flags.contains(FunctionFlags::RETURNS_SCALAR),
            "expr {expr:?} is not implemented correctly. 'returns_scalar' and 'elementwise' are mutually exclusive",
        );

        Self {
            inputs,
            function,
            expr,
            flags: options.flags,
            function_operates_on_scalar,
            input_schema,
            allow_threading,
            check_lengths: options.check_lengths(),
            output_field,
        }
    }

    #[allow(clippy::ptr_arg)]
    fn prepare_multiple_inputs<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupPositions,
        state: &ExecutionState,
    ) -> PolarsResult<Vec<AggregationContext<'a>>> {
        let f = |e: &Arc<dyn PhysicalExpr>| e.evaluate_on_groups(df, groups, state);
        if self.allow_threading {
            POOL.install(|| self.inputs.par_iter().map(f).collect())
        } else {
            self.inputs.iter().map(f).collect()
        }
    }

    fn finish_apply_groups<'a>(
        &self,
        mut ac: AggregationContext<'a>,
        ca: ListChunked,
    ) -> PolarsResult<AggregationContext<'a>> {
        let c = if self.flags.returns_scalar() {
            let out = ca.explode(false).unwrap();
            // if the explode doesn't return the same len, it wasn't scalar.
            polars_ensure!(out.len() == ca.len(), InvalidOperation: "expected scalar for expr: {}, got {}", self.expr, &out);
            ac.update_groups = UpdateGroups::No;
            out.into_column()
        } else {
            ac.with_update_groups(UpdateGroups::WithSeriesLen);
            ca.into_series().into()
        };

        ac.with_values_and_args(c, true, None, false, self.flags.returns_scalar())?;

        Ok(ac)
    }

    fn get_input_schema(&self, _df: &DataFrame) -> Cow<'_, Schema> {
        Cow::Borrowed(self.input_schema.as_ref())
    }

    /// Evaluates and flattens `Option<Column>` to `Column`.
    fn eval_and_flatten(&self, inputs: &mut [Column]) -> PolarsResult<Column> {
        self.function.call_udf(inputs)
    }
    fn apply_single_group_aware<'a>(
        &self,
        mut ac: AggregationContext<'a>,
    ) -> PolarsResult<AggregationContext<'a>> {
        let s = ac.get_values();

        #[allow(clippy::nonminimal_bool)]
        {
            polars_ensure!(
                !(matches!(ac.agg_state(), AggState::AggregatedScalar(_)) && !s.dtype().is_list() ) ,
                expr = self.expr,
                ComputeError: "cannot aggregate, the column is already aggregated",
            );
        }

        let name = s.name().clone();
        let agg = ac.aggregated();
        // Collection of empty list leads to a null dtype. See: #3687.
        if agg.is_empty() {
            // Create input for the function to determine the output dtype, see #3946.
            let agg = agg.list().unwrap();
            let input_dtype = agg.inner_dtype();
            let input = Column::full_null(name.clone(), 0, input_dtype);

            let output = self.eval_and_flatten(&mut [input])?;
            let ca = ListChunked::full(name, output.as_materialized_series(), 0);
            return self.finish_apply_groups(ac, ca);
        }

        let f = |opt_s: Option<Series>| match opt_s {
            None => Ok(None),
            Some(mut s) => {
                if self.flags.contains(FunctionFlags::PASS_NAME_TO_APPLY) {
                    s.rename(name.clone());
                }
                Ok(Some(
                    self.function
                        .call_udf(&mut [Column::from(s)])?
                        .take_materialized_series(),
                ))
            },
        };

        let ca: ListChunked = if self.allow_threading {
            let dtype = if self.output_field.dtype.is_known() && !self.output_field.dtype.is_null()
            {
                Some(self.output_field.dtype.clone())
            } else {
                None
            };

            let lst = agg.list().unwrap();
            let iter = lst.par_iter().map(f);

            if let Some(dtype) = dtype {
                // @NOTE: Since the output type for scalars does an implicit explode, we need to
                // patch up the type here to also be a list.
                let out_dtype = if self.is_scalar() {
                    DataType::List(Box::new(dtype))
                } else {
                    dtype
                };

                let out: ListChunked = POOL.install(|| {
                    iter.collect_ca_with_dtype::<PolarsResult<_>>(PlSmallStr::EMPTY, out_dtype)
                })?;
                out
            } else {
                POOL.install(|| try_list_from_par_iter(iter, PlSmallStr::EMPTY))?
            }
        } else {
            agg.list()
                .unwrap()
                .into_iter()
                .map(f)
                .collect::<PolarsResult<_>>()?
        };

        self.finish_apply_groups(ac, ca.with_name(name))
    }

    /// Apply elementwise e.g. ignore the group/list indices.
    fn apply_single_elementwise<'a>(
        &self,
        mut ac: AggregationContext<'a>,
    ) -> PolarsResult<AggregationContext<'a>> {
        let (c, aggregated) = match ac.agg_state() {
            AggState::AggregatedList(c) => {
                let ca = c.list().unwrap();
                let out = ca.apply_to_inner(&|s| {
                    Ok(self
                        .eval_and_flatten(&mut [s.into_column()])?
                        .take_materialized_series())
                })?;
                (out.into_column(), true)
            },
            AggState::NotAggregated(c) => {
                let (out, aggregated) = (self.eval_and_flatten(&mut [c.clone()])?, false);
                check_map_output_len(c.len(), out.len(), &self.expr)?;
                (out, aggregated)
            },
            agg_state => {
                ac.with_agg_state(agg_state.try_map(|s| self.eval_and_flatten(&mut [s.clone()]))?);
                return Ok(ac);
            },
        };

        ac.with_values_and_args(c, aggregated, Some(&self.expr), true, self.is_scalar())?;
        Ok(ac)
    }
    fn apply_multiple_group_aware<'a>(
        &self,
        mut acs: Vec<AggregationContext<'a>>,
        df: &DataFrame,
    ) -> PolarsResult<AggregationContext<'a>> {
        let mut container = vec![Default::default(); acs.len()];
        let schema = self.get_input_schema(df);
        let field = self.to_field(&schema)?;

        // Aggregate representation of the aggregation contexts,
        // then unpack the lists and finally create iterators from this list chunked arrays.
        let mut iters = acs
            .iter_mut()
            .map(|ac| ac.iter_groups(self.flags.contains(FunctionFlags::PASS_NAME_TO_APPLY)))
            .collect::<Vec<_>>();

        // Length of the items to iterate over.
        let len = iters[0].size_hint().0;

        let ca = if len == 0 {
            let mut builder = get_list_builder(&field.dtype, len * 5, len, field.name);
            for _ in 0..len {
                container.clear();
                for iter in &mut iters {
                    match iter.next().unwrap() {
                        None => {
                            builder.append_null();
                        },
                        Some(s) => container.push(s.deep_clone().into()),
                    }
                }
                let out = self
                    .function
                    .call_udf(&mut container)
                    .map(|c| c.take_materialized_series())?;

                builder.append_series(&out)?
            }
            builder.finish()
        } else {
            // We still need this branch to materialize unknown/ data dependent types in eager. :(
            (0..len)
                .map(|_| {
                    container.clear();
                    for iter in &mut iters {
                        match iter.next().unwrap() {
                            None => return Ok(None),
                            Some(s) => container.push(s.deep_clone().into()),
                        }
                    }
                    Ok(Some(
                        self.function
                            .call_udf(&mut container)?
                            .take_materialized_series(),
                    ))
                })
                .collect::<PolarsResult<ListChunked>>()?
                .with_name(field.name.clone())
        };
        #[cfg(debug_assertions)]
        {
            let inner = ca.dtype().inner_dtype().unwrap();
            if field.dtype.is_known() {
                assert_eq!(inner, &field.dtype);
            }
        }

        drop(iters);

        // Take the first aggregation context that as that is the input series.
        let ac = acs.swap_remove(0);
        self.finish_apply_groups(ac, ca)
    }
}

fn check_map_output_len(input_len: usize, output_len: usize, expr: &Expr) -> PolarsResult<()> {
    polars_ensure!(
        input_len == output_len, expr = expr, InvalidOperation:
        "output length of `map` ({}) must be equal to the input length ({}); \
        consider using `apply` instead", output_len, input_len
    );
    Ok(())
}

impl PhysicalExpr for ApplyExpr {
    fn as_expression(&self) -> Option<&Expr> {
        Some(&self.expr)
    }

    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Column> {
        let f = |e: &Arc<dyn PhysicalExpr>| e.evaluate(df, state);
        let mut inputs = if self.allow_threading && self.inputs.len() > 1 {
            POOL.install(|| {
                self.inputs
                    .par_iter()
                    .map(f)
                    .collect::<PolarsResult<Vec<_>>>()
            })
        } else {
            self.inputs.iter().map(f).collect::<PolarsResult<Vec<_>>>()
        }?;

        if self.flags.contains(FunctionFlags::ALLOW_RENAME) {
            self.eval_and_flatten(&mut inputs)
        } else {
            let in_name = inputs[0].name().clone();
            Ok(self.eval_and_flatten(&mut inputs)?.with_name(in_name))
        }
    }

    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupPositions,
        state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        if self.inputs.len() == 1 {
            let ac = self.inputs[0].evaluate_on_groups(df, groups, state)?;

            match self.flags.is_elementwise() {
                false => self.apply_single_group_aware(ac),
                true => self.apply_single_elementwise(ac),
            }
        } else {
            let acs = self.prepare_multiple_inputs(df, groups, state)?;

            match self.flags.is_elementwise() {
                false => self.apply_multiple_group_aware(acs, df),
                true => {
                    let mut has_agg_list = false;
                    let mut has_agg_scalar = false;
                    let mut has_not_agg = false;
                    for ac in &acs {
                        match ac.state {
                            AggState::AggregatedList(_) => has_agg_list = true,
                            AggState::AggregatedScalar(_) => has_agg_scalar = true,
                            AggState::NotAggregated(_) => has_not_agg = true,
                            _ => {},
                        }
                    }
                    if has_agg_list || (has_agg_scalar && has_not_agg) {
                        self.apply_multiple_group_aware(acs, df)
                    } else {
                        apply_multiple_elementwise(
                            acs,
                            self.function.as_ref(),
                            &self.expr,
                            self.check_lengths,
                            self.is_scalar(),
                        )
                    }
                },
            }
        }
    }

    fn to_field(&self, input_schema: &Schema) -> PolarsResult<Field> {
        self.expr.to_field(input_schema)
    }
    fn as_partitioned_aggregator(&self) -> Option<&dyn PartitionedAggregation> {
        if self.inputs.len() == 1 && self.flags.is_elementwise() {
            Some(self)
        } else {
            None
        }
    }
    fn is_scalar(&self) -> bool {
        self.flags.returns_scalar()
            || (self.function_operates_on_scalar && self.flags.is_length_preserving())
    }
}

fn apply_multiple_elementwise<'a>(
    mut acs: Vec<AggregationContext<'a>>,
    function: &dyn ColumnsUdf,
    expr: &Expr,
    check_lengths: bool,
    returns_scalar: bool,
) -> PolarsResult<AggregationContext<'a>> {
    match acs.first().unwrap().agg_state() {
        // A fast path that doesn't drop groups of the first arg.
        // This doesn't require group re-computation.
        AggState::AggregatedList(s) => {
            let ca = s.list().unwrap();

            let other = acs[1..]
                .iter()
                .map(|ac| ac.flat_naive().into_owned())
                .collect::<Vec<_>>();

            let out = ca.apply_to_inner(&|s| {
                let mut args = Vec::with_capacity(other.len() + 1);
                args.push(s.into());
                args.extend_from_slice(&other);
                Ok(function
                    .call_udf(&mut args)?
                    .as_materialized_series()
                    .clone())
            })?;
            let mut ac = acs.swap_remove(0);
            ac.with_values(out.into_column(), true, None)?;
            Ok(ac)
        },
        first_as => {
            let check_lengths = check_lengths && !matches!(first_as, AggState::LiteralScalar(_));
            let aggregated = acs.iter().all(|ac| ac.is_aggregated() | ac.is_literal())
                && acs.iter().any(|ac| ac.is_aggregated());
            let mut c = acs
                .iter_mut()
                .enumerate()
                .map(|(i, ac)| {
                    // Make sure the groups are updated because we are about to throw away
                    // the series length information, only on the first iteration.
                    if let (0, UpdateGroups::WithSeriesLen) = (i, &ac.update_groups) {
                        ac.groups();
                    }

                    ac.flat_naive().into_owned()
                })
                .collect::<Vec<_>>();

            let input_len = c[0].len();
            let c = function.call_udf(&mut c)?;
            if check_lengths {
                check_map_output_len(input_len, c.len(), expr)?;
            }

            // Take the first aggregation context that as that is the input series.
            let mut ac = acs.swap_remove(0);
            ac.with_values_and_args(c, aggregated, None, true, returns_scalar)?;
            Ok(ac)
        },
    }
}

impl PartitionedAggregation for ApplyExpr {
    fn evaluate_partitioned(
        &self,
        df: &DataFrame,
        groups: &GroupPositions,
        state: &ExecutionState,
    ) -> PolarsResult<Column> {
        let a = self.inputs[0].as_partitioned_aggregator().unwrap();
        let s = a.evaluate_partitioned(df, groups, state)?;

        if self.flags.contains(FunctionFlags::ALLOW_RENAME) {
            self.eval_and_flatten(&mut [s])
        } else {
            let in_name = s.name().clone();
            Ok(self.eval_and_flatten(&mut [s])?.with_name(in_name))
        }
    }

    fn finalize(
        &self,
        partitioned: Column,
        _groups: &GroupPositions,
        _state: &ExecutionState,
    ) -> PolarsResult<Column> {
        Ok(partitioned)
    }
}
