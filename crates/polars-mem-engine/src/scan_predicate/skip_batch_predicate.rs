//! Evaluation of a statistics-based batch-skip predicate.

use std::sync::Arc;

use polars_arrow::array::BooleanArray;
use polars_arrow::bitmap::MutableBitmap;
use polars_core::prelude::*;
use polars_expr::prelude::{AggregationContext, PhysicalExpr};
use polars_expr::state::ExecutionState;
use polars_expr::{ExpressionConversionState, create_physical_expr};
use polars_plan::dsl::Operator;
use polars_plan::plans::AExpr;
use polars_plan::plans::expr_ir::{ExprIR, OutputName};
use polars_utils::arena::{Arena, Node};
use polars_utils::pl_str::PlSmallStr;

/// Build the physical expression of a batch-skip predicate, with `&` and `|` evaluated strictly
/// left to right.
///
/// A batch-skip predicate is a proof attempt on a batch's statistics: the plan layer rewrites the
/// row predicate into terms over `<col>_min` / `<col>_max` / `<col>_nc` (see
/// `aexpr_to_skip_batch_predicate`). Some of those terms substitute a statistic for a column
/// value, guarded by the terms the plan layer conjoins to their left. A statistic is a bound of
/// the batch's values, not necessarily a value the batch contains, so such a term — an arbitrary
/// expression, e.g. `str.json_decode()`, `str.slice(..).cast(i64)` or a WKB parser — can fail on
/// the batches whose statistics it is substituted into.
///
/// Evaluating `&` left to right confines every term to the batches its left-hand side holds for,
/// and `|` confines the right-hand side to the batches the left-hand side does not already skip.
/// A term whose guard does not hold is never evaluated, and the batches it would have settled are
/// read instead. This decides every batch exactly as evaluating the whole predicate would:
///
///  - `false & x` is `false` and `true | x` is `true` whatever `x` is, so a right-hand side that
///    is not evaluated cannot change the answer, and
///  - an operand that is not evaluated does not skip: a `null` operand and a `null` result both
///    mean "do not skip".
pub(crate) fn create_skip_batch_expr(
    node: Node,
    expr_arena: &mut Arena<AExpr>,
    schema: &Arc<Schema>,
    state: &mut ExpressionConversionState,
) -> PolarsResult<Arc<dyn PhysicalExpr>> {
    Ok(Arc::new(SkipBatchExpr {
        root: build(node, expr_arena, schema, state)?,
    }))
}

enum SkipBatchExprNode {
    /// A term of the predicate, evaluated for every batch.
    Leaf(Arc<dyn PhysicalExpr>),
    And(Box<Self>, Box<Self>),
    Or(Box<Self>, Box<Self>),
}

fn build(
    node: Node,
    expr_arena: &mut Arena<AExpr>,
    schema: &Arc<Schema>,
    state: &mut ExpressionConversionState,
) -> PolarsResult<SkipBatchExprNode> {
    let binary = match expr_arena.get(node) {
        AExpr::BinaryExpr { left, op, right } => Some((*left, *op, *right)),
        _ => None,
    };

    match binary {
        Some((left, op, right)) if matches!(op, Operator::And | Operator::Or) => {
            let left = build(left, expr_arena, schema, state)?;
            let right = build(right, expr_arena, schema, state)?;
            Ok(match op {
                Operator::And => SkipBatchExprNode::And(Box::new(left), Box::new(right)),
                _ => SkipBatchExprNode::Or(Box::new(left), Box::new(right)),
            })
        },
        _ => {
            let expr = ExprIR::new(node, OutputName::Alias(PlSmallStr::EMPTY));
            Ok(SkipBatchExprNode::Leaf(create_physical_expr(
                &expr, expr_arena, schema, state,
            )?))
        },
    }
}

impl SkipBatchExprNode {
    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Column> {
        let (left, right, skip_on_left) = match self {
            Self::Leaf(expr) => return expr.evaluate(df, state),
            Self::And(left, right) => (left, right, false),
            Self::Or(left, right) => (left, right, true),
        };

        let height = df.height();
        let left = verdicts(&left.evaluate(df, state)?, height)?;
        let mut out = MutableBitmap::from_len_zeroed(height);
        let mut rows: Vec<IdxSize> = Vec::new();

        for (i, verdict) in left.iter().enumerate() {
            match (skip_on_left, verdict) {
                // `|`: the left-hand side skips the batch, the right-hand side cannot
                // change that.
                (true, Some(true)) => out.set(i, true),
                // `|`: the left-hand side skips nothing, the right-hand side decides.
                // `&`: the left-hand side skips nothing already, so the batch is not
                // skipped and the right-hand side must not be evaluated for it.
                (true, _) | (false, Some(true)) => rows.push(i as IdxSize),
                (false, _) => {},
            }
        }

        if !rows.is_empty() {
            let indices = IdxCa::from_vec(PlSmallStr::EMPTY, rows);
            let sub = df.take(&indices)?;
            let right = verdicts(&right.evaluate(&sub, state)?, sub.height())?;

            for (row, verdict) in indices.iter().flatten().zip(right.iter()) {
                if verdict == Some(true) {
                    out.set(row as usize, true);
                }
            }
        }

        Ok(
            BooleanChunked::with_chunk(PlSmallStr::EMPTY, BooleanArray::from(out.freeze()))
                .into_column(),
        )
    }
}

/// The verdict of a term of the skip predicate for each of the `height` batches: `Some(true)`
/// where the term skips the batch.
///
/// A term that is `null` for a batch skips nothing there, exactly like a term that is `false`.
fn verdicts(mask: &Column, height: usize) -> PolarsResult<BooleanChunked> {
    match (mask.len(), height) {
        (len, height) if len == height => Ok(mask.bool()?.clone()),
        (1, height) if height != 1 => Ok(mask.new_from_index(0, height).bool()?.clone()),
        (len, height) => {
            polars_bail!(
                ComputeError:
                "batch-skip statistics predicate has length {} for {} batches",
                len, height,
            )
        },
    }
}

struct SkipBatchExpr {
    root: SkipBatchExprNode,
}

#[cfg(test)]
mod tests {
    use polars_core::chunked_array::cast::CastOptions;
    use polars_plan::plans::AExprBuilder;

    use super::*;

    /// The statistics frame of `min.len()` batches, with one string column.
    fn statistics_frame(
        min: &[Option<&str>],
        max: &[Option<&str>],
        null_count: &[Option<IdxSize>],
    ) -> DataFrame {
        let height = min.len();
        DataFrame::new(
            height,
            vec![
                Column::new(
                    PlSmallStr::from_static("len"),
                    vec![height as IdxSize; height],
                ),
                Series::new(PlSmallStr::from_static("s_min"), min).into_column(),
                Series::new(PlSmallStr::from_static("s_max"), max).into_column(),
                Series::new(PlSmallStr::from_static("s_nc"), null_count).into_column(),
            ],
        )
        .unwrap()
    }

    fn schema() -> Arc<Schema> {
        Arc::new(Schema::from_iter([
            (PlSmallStr::from_static("len"), IDX_DTYPE),
            (PlSmallStr::from_static("s_min"), DataType::String),
            (PlSmallStr::from_static("s_max"), DataType::String),
            (PlSmallStr::from_static("s_nc"), IDX_DTYPE),
        ]))
    }

    fn column(arena: &mut Arena<AExpr>, name: &str) -> Node {
        arena.add(AExpr::Column(PlSmallStr::from_string(name.to_string())))
    }

    fn binary(arena: &mut Arena<AExpr>, left: Node, op: Operator, right: Node) -> Node {
        arena.add(AExpr::BinaryExpr { left, op, right })
    }

    fn zero(arena: &mut Arena<AExpr>) -> Node {
        arena.add(AExpr::Literal(
            Scalar::new(IDX_DTYPE, (0 as IdxSize).into()).into(),
        ))
    }

    /// `min == max && null_count == 0`: the plan layer's proof that a batch is the value its
    /// statistics hold.
    fn guard(arena: &mut Arena<AExpr>) -> Node {
        let min = column(arena, "s_min");
        let max = column(arena, "s_max");
        let null_count = column(arena, "s_nc");
        let min_is_max = binary(arena, min, Operator::Eq, max);
        let zero = zero(arena);
        let no_nulls = binary(arena, null_count, Operator::Eq, zero);
        binary(arena, min_is_max, Operator::And, no_nulls)
    }

    /// `!(s_min > 5)`: a term substituted with a statistic, which fails on a bound that is not
    /// a value, exactly like the expressions of issue #29447.
    fn substituted_term(arena: &mut Arena<AExpr>) -> Node {
        let min = column(arena, "s_min");
        let cast = arena.add(AExpr::Cast {
            expr: min,
            dtype: DataType::Int64,
            options: CastOptions::Strict,
        });
        let five = arena.add(AExpr::Literal(Scalar::from(5i64).into()));
        let greater = binary(arena, cast, Operator::Gt, five);
        AExprBuilder::new_from_node(greater).not(arena).node()
    }

    /// The verdicts of `node` for the batches of `df`.
    fn verdicts_of(node: Node, arena: &mut Arena<AExpr>, df: &DataFrame) -> Vec<Option<bool>> {
        let expr = create_skip_batch_expr(
            node,
            arena,
            &schema(),
            &mut ExpressionConversionState::new(true),
        )
        .unwrap();
        expr.evaluate(df, &Default::default())
            .unwrap()
            .bool()
            .unwrap()
            .iter()
            .collect()
    }

    /// A guarded term is only evaluated for the batches its guard holds for: the second batch's
    /// `"abc"` cannot be cast, and is never looked at.
    #[test]
    fn and_evaluates_a_guarded_term_only_where_the_guard_holds() {
        let mut arena = Arena::new();
        let guard = guard(&mut arena);
        let substituted = substituted_term(&mut arena);
        let node = binary(&mut arena, guard, Operator::And, substituted);
        let df = statistics_frame(
            &[Some("3"), Some("abc")],
            &[Some("3"), Some("bcd")],
            &[Some(0), Some(0)],
        );

        assert_eq!(
            verdicts_of(node, &mut arena, &df),
            vec![Some(true), Some(false)]
        );
    }

    /// `|` only evaluates its right-hand side for the batches its left-hand side does not
    /// already skip: the first batch's left-hand side skips it, and its guarded term — whose
    /// own guard holds there — is never looked at.
    #[test]
    fn or_evaluates_a_guarded_term_only_where_the_left_does_not_skip() {
        let mut arena = Arena::new();
        let min = column(&mut arena, "s_min");
        let zzz = arena.add(AExpr::Literal(
            Scalar::new(DataType::String, AnyValue::String("zzz")).into(),
        ));
        let is_zzz = binary(&mut arena, min, Operator::Eq, zzz);
        let guard = guard(&mut arena);
        let substituted = substituted_term(&mut arena);
        let guarded = binary(&mut arena, guard, Operator::And, substituted);
        let node = binary(&mut arena, is_zzz, Operator::Or, guarded);
        let df = statistics_frame(
            &[Some("zzz"), Some("3")],
            &[Some("zzz"), Some("3")],
            &[Some(0), Some(0)],
        );

        assert_eq!(
            verdicts_of(node, &mut arena, &df),
            vec![Some(true), Some(true)]
        );
    }

    /// A `null` verdict settles nothing: the batch is read, and the guarded term is not
    /// evaluated — `min == max` holds here, but the null count is unknown.
    #[test]
    fn null_verdicts_skip_nothing() {
        let mut arena = Arena::new();
        let guard = guard(&mut arena);
        let substituted = substituted_term(&mut arena);
        let node = binary(&mut arena, guard, Operator::And, substituted);
        let df = statistics_frame(&[Some("abc")], &[Some("abc")], &[None]);

        assert_eq!(verdicts_of(node, &mut arena, &df), vec![Some(false)]);
    }

    /// A literal operand is one verdict for every batch, and `true` on the left of `|` skips
    /// them without evaluating the right-hand side.
    #[test]
    fn literal_operands_are_broadcast() {
        let mut arena = Arena::new();
        let always = arena.add(AExpr::Literal(Scalar::from(true).into()));
        let substituted = substituted_term(&mut arena);
        let node = binary(&mut arena, always, Operator::Or, substituted);
        let df = statistics_frame(
            &[Some("abc"), Some("def")],
            &[Some("bcd"), Some("efg")],
            &[Some(0), Some(0)],
        );

        assert_eq!(
            verdicts_of(node, &mut arena, &df),
            vec![Some(true), Some(true)]
        );
    }
}

impl PhysicalExpr for SkipBatchExpr {
    fn evaluate_impl(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Column> {
        self.root.evaluate(df, state)
    }

    fn evaluate_on_groups_impl<'a>(
        &self,
        _df: &DataFrame,
        _groups: &'a GroupPositions,
        _state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        polars_bail!(
            InvalidOperation: "batch-skip statistics predicates are not evaluated on groups"
        )
    }

    fn to_field(&self, _input_schema: &Schema) -> PolarsResult<Field> {
        Ok(Field::new(PlSmallStr::EMPTY, DataType::Boolean))
    }

    fn is_scalar(&self) -> bool {
        false
    }
}
