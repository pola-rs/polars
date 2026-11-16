use std::sync::Arc;

use polars_arrow::bitmap::MutableBitmap;
use polars_core::prelude::*;
use polars_core::runtime::RAYON;
use polars_plan::dsl::ColumnsUdf;
use rayon::prelude::*;

use crate::expressions::PhysicalExpr;
use crate::state::ExecutionState;

/// An arm or the default of `union.match`, evaluated on the rows it covers.
pub struct MatchBranch {
    pub expr: Arc<dyn PhysicalExpr>,
    /// For an arm, its variant, and whether it reads the variant's value as `pl.element()`.
    pub variant: Option<(usize, bool)>,
}

/// The kernel of `union.match`, a function of the union, then the values of `columns`, of the
/// struct fields `fields` (`pl.field`), and of the enclosing `pl.element()` if `reads_element`.
///
/// Every arm runs on the rows of its variant only, and the default on the non-null rows whose
/// variant has no arm.
pub struct UnionMatchKernel {
    pub arms: Vec<MatchBranch>,
    pub default: Option<MatchBranch>,
    pub columns: Vec<PlSmallStr>,
    pub fields: Vec<PlSmallStr>,
    pub reads_element: bool,
    pub dtype: DataType,
    pub allow_threading: bool,
}

impl UnionMatchKernel {
    fn branches(&self) -> impl Iterator<Item = &MatchBranch> {
        self.arms.iter().chain(&self.default)
    }

    /// Evaluates `branch` on `rows` of the `inputs` after the union, where the variant has
    /// `values` if the branch reads them.
    fn evaluate_branch(
        &self,
        branch: &MatchBranch,
        rows: &[IdxSize],
        values: Option<&Series>,
        inputs: &[Column],
    ) -> PolarsResult<Column> {
        let height = rows.len();
        let idx = IdxCa::from_slice(PlSmallStr::EMPTY, rows);
        let gather = |c: &Column| {
            if c.len() == 1 {
                Ok(c.new_from_index(0, height))
            } else {
                c.take(&idx)
            }
        };
        let (columns, rest) = inputs.split_at(self.columns.len());

        let frame = columns
            .iter()
            .zip(&self.columns)
            .map(|(c, name)| Ok(gather(c)?.with_name(name.clone())))
            .collect::<PolarsResult<Vec<_>>>()?;
        let frame = DataFrame::new_with_broadcast(height, frame)?;

        let mut state = ExecutionState::new();
        #[cfg(feature = "dtype-struct")]
        if !self.fields.is_empty() {
            let fields = rest
                .iter()
                .zip(&self.fields)
                .map(|(c, name)| Ok(gather(c)?.with_name(name.clone())))
                .collect::<PolarsResult<Vec<_>>>()?;
            let ca = StructChunked::from_columns(PlSmallStr::EMPTY, height, &fields)?;
            state.with_fields = Some(Arc::new(ca));
        }
        match (&branch.variant, values) {
            (Some(_), Some(values)) => {
                state.element = Arc::new(Some((values.clone().into_column(), None)));
            },
            (None, _) if self.reads_element => {
                let element = &rest[self.fields.len()];
                state.element = Arc::new(Some((gather(element)?, None)));
            },
            _ => {},
        }

        let out = branch.expr.evaluate(&frame, &state)?;
        polars_ensure!(
            out.len() == height || out.len() == 1,
            ShapeMismatch: "an arm of `union.match` returned {} rows for {} rows of its variant",
            out.len(), height
        );
        if out.dtype() != &self.dtype {
            out.strict_cast(&self.dtype)
        } else {
            Ok(out)
        }
    }
}

impl ColumnsUdf for UnionMatchKernel {
    fn call_udf(&self, inputs: &mut [Column]) -> PolarsResult<Column> {
        let len = inputs.iter().map(|c| c.len()).max().unwrap();
        polars_ensure!(
            inputs.iter().all(|c| c.len() == len || c.len() == 1),
            ShapeMismatch: "the union and the columns read by the arms of `union.match` must have \
            the same length or length 1"
        );
        let union = &inputs[0];
        let union = if union.len() == len {
            union.clone()
        } else {
            union.new_from_index(0, len)
        };
        let union = union.as_materialized_series();
        let ca = union.union()?;
        let n_variants = ca.union_fields().len();

        let n_groups = self.arms.len() + self.default.is_some() as usize;
        let mut group_of = vec![self.default.as_ref().map(|_| self.arms.len()); n_variants];
        let mut values_of = vec![false; n_variants];
        for (g, arm) in self.arms.iter().enumerate() {
            let (t, reads_values) = arm.variant.unwrap();
            group_of[t] = Some(g);
            values_of[t] = reads_values;
        }
        let (rows, values) = ca.split_by_variant(&group_of, n_groups, &values_of);

        let inputs = &inputs[1..];
        let evaluate = |(g, branch): (usize, &MatchBranch)| -> PolarsResult<Option<Column>> {
            if rows[g].is_empty() {
                return Ok(None);
            }
            let values = branch.variant.and_then(|(t, _)| values[t].as_ref());
            self.evaluate_branch(branch, &rows[g], values, inputs)
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

        let name = union.name().clone();

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
            return Ok(Column::full_null(name, len, &self.dtype));
        };
        let idx = IdxCa::from_vec_validity(PlSmallStr::EMPTY, idx, Some(validity.freeze()));
        Ok(concatenated.take(&idx)?.with_name(name))
    }
}
