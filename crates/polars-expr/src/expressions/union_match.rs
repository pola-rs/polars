use std::borrow::Cow;
use std::sync::Arc;

use polars_arrow::bitmap::MutableBitmap;
use polars_core::prelude::*;
use polars_core::runtime::RAYON;
use polars_plan::prelude::*;
use rayon::prelude::*;

use super::*;
use crate::expressions::{AggregationContext, PhysicalExpr};

/// An arm or the default of `union.match`, evaluated on the rows it covers.
pub struct MatchBranch {
    pub expr: Arc<dyn PhysicalExpr>,
    /// The columns the branch reads, which are gathered at its rows.
    pub columns: Vec<PlSmallStr>,
    /// Whether the branch reads `pl.element()` of an enclosing `eval`.
    pub reads_element: bool,
    /// For an arm, its variant and the column holding the variant's values, if it reads them.
    pub variant: Option<(usize, Option<PlSmallStr>)>,
}

/// `union.match`: every arm runs on the values of its variant only, and the default on the rows
/// whose variant has no arm.
pub struct UnionMatchExpr {
    input: Arc<dyn PhysicalExpr>,
    arms: Vec<MatchBranch>,
    default: Option<MatchBranch>,
    expr: Expr,
    output_field: Field,
    is_scalar: bool,
    allow_threading: bool,
}

impl UnionMatchExpr {
    pub fn new(
        input: Arc<dyn PhysicalExpr>,
        arms: Vec<MatchBranch>,
        default: Option<MatchBranch>,
        expr: Expr,
        output_field: Field,
        is_scalar: bool,
        allow_threading: bool,
    ) -> Self {
        Self {
            input,
            arms,
            default,
            expr,
            output_field,
            is_scalar,
            allow_threading,
        }
    }

    fn branches(&self) -> impl Iterator<Item = &MatchBranch> {
        self.arms.iter().chain(&self.default)
    }

    /// The height of the frame the branches read; a unit-length union is broadcast to it.
    fn frame_height(&self, df: &DataFrame, state: &ExecutionState) -> Option<usize> {
        if self.branches().any(|b| !b.columns.is_empty()) {
            Some(df.height())
        } else if self.branches().any(|b| b.reads_element) {
            state.element.as_ref().as_ref().map(|(e, _)| e.len())
        } else {
            None
        }
    }

    /// Evaluates `branch` on `rows`, which hold `values` if it is an arm.
    fn evaluate_branch(
        &self,
        branch: &MatchBranch,
        rows: &[IdxSize],
        values: Option<&Series>,
        df: &DataFrame,
        state: &ExecutionState,
    ) -> PolarsResult<Column> {
        let height = rows.len();
        let idx = IdxCa::from_slice(PlSmallStr::EMPTY, rows);
        // Unit-length columns, such as those common subexpression elimination adds for scalars,
        // are broadcast.
        let gather = |c: &Column| {
            if c.len() == 1 {
                Ok(c.clone())
            } else {
                c.take(&idx)
            }
        };
        let mut columns = branch
            .columns
            .iter()
            .map(|name| gather(df.column(name)?))
            .collect::<PolarsResult<Vec<_>>>()?;
        if let (Some((_, Some(name))), Some(values)) = (&branch.variant, values) {
            columns.push(values.clone().with_name(name.clone()).into_column());
        }
        let frame = DataFrame::new_with_broadcast(height, columns)?;

        let mut state = state.split();
        if branch.reads_element {
            if let Some((element, _)) = state.element.as_ref() {
                state.element = Arc::new(Some((gather(element)?, None)));
            }
        }
        let out = branch.expr.evaluate(&frame, &state)?;
        polars_ensure!(
            out.len() == height || out.len() == 1,
            ShapeMismatch: "an arm of `union.match` returned {} rows for {} rows of its variant",
            out.len(), height
        );
        let dtype = self.output_field.dtype();
        if out.dtype() != dtype && !dtype.contains_unknown() {
            out.strict_cast(dtype)
        } else {
            Ok(out)
        }
    }
}

impl PhysicalExpr for UnionMatchExpr {
    fn as_expression(&self) -> Option<&Expr> {
        Some(&self.expr)
    }

    fn evaluate_impl(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Column> {
        let mut input = self.input.evaluate(df, state)?;
        if input.len() == 1 && !self.is_scalar {
            if let Some(height) = self.frame_height(df, state) {
                input = input.new_from_index(0, height);
            }
        }
        let input = input.as_materialized_series();
        let ca = input.union()?;
        let n_variants = ca.union_fields().len();

        let n_groups = self.arms.len() + self.default.is_some() as usize;
        let mut group_of = vec![self.default.as_ref().map(|_| self.arms.len()); n_variants];
        let mut values_of = vec![false; n_variants];
        for (g, arm) in self.arms.iter().enumerate() {
            let (t, values) = arm.variant.as_ref().unwrap();
            group_of[*t] = Some(g);
            values_of[*t] = values.is_some();
        }
        let (rows, values) = ca.split_by_variant(&group_of, n_groups, &values_of);

        let evaluate = |(g, branch): (usize, &MatchBranch)| -> PolarsResult<Option<Column>> {
            if rows[g].is_empty() {
                return Ok(None);
            }
            let values = branch
                .variant
                .as_ref()
                .and_then(|(t, _)| values[*t].as_ref());
            self.evaluate_branch(branch, &rows[g], values, df, state)
                .map(Some)
        };
        let branches: Vec<_> = self.branches().enumerate().collect();
        let outputs: Vec<Option<Column>> = if self.allow_threading && n_groups > 1 {
            RAYON.install(|| {
                branches
                    .into_par_iter()
                    .map(evaluate)
                    .collect::<PolarsResult<_>>()
            })
        } else {
            branches
                .into_iter()
                .map(evaluate)
                .collect::<PolarsResult<_>>()
        }?;

        let name = input.name().clone();
        let dtype = self.output_field.dtype();
        let len = input.len();

        // One branch that covers every row in order is the output.
        if let Some(g) = rows.iter().position(|r| r.len() == len) {
            if let Some(out) = &outputs[g] {
                if out.len() == len {
                    return Ok(out.clone().with_name(name));
                }
            }
        }

        // Gather every row from the output of its branch; null rows and rows without a branch
        // stay null.
        let mut idx = vec![0 as IdxSize; len];
        let mut validity = MutableBitmap::from_len_zeroed(len);
        let mut concatenated: Option<Column> = None;
        for (rows, out) in rows.iter().zip(outputs) {
            let Some(out) = out else { continue };
            let base = concatenated.as_ref().map_or(0, |c| c.len()) as IdxSize;
            let broadcast = out.len() != rows.len();
            for (rank, &row) in rows.iter().enumerate() {
                idx[row as usize] = base + if broadcast { 0 } else { rank as IdxSize };
                validity.set(row as usize, true);
            }
            match &mut concatenated {
                None => concatenated = Some(out),
                Some(c) => {
                    c.append_owned(out)?;
                },
            }
        }
        let Some(concatenated) = concatenated else {
            return Ok(Column::full_null(name, len, dtype));
        };
        let idx = IdxCa::from_vec_validity(PlSmallStr::EMPTY, idx, Some(validity.freeze()));
        Ok(concatenated.take(&idx)?.with_name(name))
    }

    fn evaluate_on_groups_impl<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupPositions,
        state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        // The lowering only builds this expression when it is elementwise, so it can run on the
        // flat values and keep the groups.
        let out = self.evaluate(df, state)?;
        if self.is_scalar {
            Ok(AggregationContext::from_agg_state(
                AggState::LiteralScalar(out),
                Cow::Borrowed(groups),
            ))
        } else {
            Ok(AggregationContext::new(out, Cow::Borrowed(groups), false))
        }
    }

    fn to_field(&self, _input_schema: &Schema) -> PolarsResult<Field> {
        Ok(self.output_field.clone())
    }

    fn is_scalar(&self) -> bool {
        self.is_scalar
    }
}
