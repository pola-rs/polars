//! This module creates predicates that can skip record batches of rows based on statistics about
//! that record batch.

use std::borrow::Cow;

use polars_core::prelude::{AnyValue, DataType, Scalar};
use polars_core::schema::Schema;
use polars_utils::aliases::PlIndexMap;
use polars_utils::arena::{Arena, Node};
use polars_utils::format_pl_smallstr;
use polars_utils::pl_str::PlSmallStr;

#[cfg(feature = "dtype-struct")]
use super::super::IRStructFunction;
use super::super::evaluate::constant_evaluate;
use super::super::{AExpr, IRBooleanFunction, IRFunctionExpr, LiteralValue, Operator};
use crate::plans::aexpr::builder::IntoAExprBuilder;
#[cfg(feature = "dtype-struct")]
use crate::plans::expr_ir::ExprIR;
#[cfg(feature = "is_in")]
use crate::plans::predicates::try_extract_is_in_haystack;
use crate::plans::{AExprBuilder, aexpr_to_leaf_names_iter, is_scalar_ae, rename_columns};

/// A resolved reference to a (possibly nested) statistics target: a top-level column,
/// optionally followed by a struct-field path.
///
/// The skip-batch statistics frame carries `<col>_min` / `<col>_max` / `<col>_nc` columns
/// whose shape mirrors the data column. A struct field `s.a` is therefore read out of the
/// struct-typed statistics column as `col("s_min").struct.field("a")` (and likewise for
/// `_max` / `_nc`), letting the existing scalar handlers prune on an individual field even
/// when sibling fields vary.
#[derive(Clone)]
struct StatTarget {
    col: PlSmallStr,
    /// Field path into a struct column, outermost first. Empty for a whole column.
    #[cfg_attr(not(feature = "dtype-struct"), allow(dead_code))]
    path: Vec<PlSmallStr>,
}

impl StatTarget {
    /// Build the `<col>_<suffix>` statistics accessor, indexing into the struct field path.
    fn stat(&self, suffix: &str, arena: &mut Arena<AExpr>) -> AExprBuilder {
        #[allow(unused_mut)]
        let mut b = AExprBuilder::new_from_node(
            arena.add(AExpr::Column(format_pl_smallstr!("{}_{suffix}", self.col))),
        );
        #[cfg(feature = "dtype-struct")]
        {
            for field in &self.path {
                b = struct_field(b, field.clone(), arena);
            }
        }
        b
    }

    fn min(&self, arena: &mut Arena<AExpr>) -> AExprBuilder {
        self.stat("min", arena)
    }
    fn max(&self, arena: &mut Arena<AExpr>) -> AExprBuilder {
        self.stat("max", arena)
    }
    fn null_count(&self, arena: &mut Arena<AExpr>) -> AExprBuilder {
        self.stat("nc", arena)
    }
}

#[cfg(feature = "dtype-struct")]
fn struct_field(base: AExprBuilder, name: PlSmallStr, arena: &mut Arena<AExpr>) -> AExprBuilder {
    AExprBuilder::function(
        vec![ExprIR::from_node(base.node(), arena)],
        IRFunctionExpr::StructExpr(IRStructFunction::FieldByName(name)),
        arena,
    )
}

/// Resolve a `col(..)` or chained `col(..).struct.field(..)` expression to a [`StatTarget`].
/// Returns `None` for anything else (literals, computed expressions, ...).
fn resolve_stat_target(e: Node, arena: &Arena<AExpr>) -> Option<StatTarget> {
    match arena.get(e) {
        AExpr::Column(c) => Some(StatTarget {
            col: c.clone(),
            path: Vec::new(),
        }),
        #[cfg(feature = "dtype-struct")]
        AExpr::Function {
            input,
            function: IRFunctionExpr::StructExpr(IRStructFunction::FieldByName(name)),
            ..
        } if input.len() == 1 => {
            let mut inner = resolve_stat_target(input[0].node(), arena)?;
            inner.path.push(name.clone());
            Some(inner)
        },
        _ => None,
    }
}

/// The leaf dtype a [`StatTarget`] points at, walking the struct-field path through `schema`.
fn target_leaf_dtype<'a>(target: &StatTarget, schema: &'a Schema) -> Option<&'a DataType> {
    #[allow(unused_mut)]
    let mut dtype = schema.get(&target.col)?;
    #[cfg(feature = "dtype-struct")]
    {
        for field in &target.path {
            let DataType::Struct(fields) = dtype else {
                return None;
            };
            dtype = fields
                .iter()
                .find(|f| f.name() == field)
                .map(|f| f.dtype())?;
        }
    }
    Some(dtype)
}

/// Collect every scalar leaf reachable from `base` under a struct `dtype`. Used to express
/// whole-struct null tests as a conjunction over the per-field null counts.
#[cfg(feature = "dtype-struct")]
fn struct_leaf_targets(base: &StatTarget, dtype: &DataType, out: &mut Vec<StatTarget>) {
    match dtype {
        DataType::Struct(fields) => {
            for f in fields {
                let mut child = base.clone();
                child.path.push(f.name().clone());
                struct_leaf_targets(&child, f.dtype(), out);
            }
        },
        _ => out.push(base.clone()),
    }
}

/// Like `get_binary_expr_col_and_lv`, but resolving the column side to a [`StatTarget`] so a
/// `struct.field(..)` comparison is recognised too.
#[allow(clippy::type_complexity)]
fn get_binary_expr_target_and_lv<'a>(
    left: Node,
    right: Node,
    arena: &'a Arena<AExpr>,
    schema: &Schema,
) -> Option<((StatTarget, Node), (Option<Cow<'a, LiteralValue>>, Node))> {
    match (
        resolve_stat_target(left, arena),
        resolve_stat_target(right, arena),
        constant_evaluate(left, arena, schema, 0),
        constant_evaluate(right, arena, schema, 0),
    ) {
        (Some(target), _, _, Some(lv)) => Some(((target, left), (lv, right))),
        (_, Some(target), Some(lv), _) => Some(((target, right), (lv, left))),
        _ => None,
    }
}

/// Return a new boolean expression determines whether a batch can be skipped based on min, max and
/// null count statistics.
///
/// This is conversative and may return `None` or `false` when an expression is not yet supported.
///
/// To evaluate, the expression it is given all the original column appended with `_min` and
/// `_max`. The `min` or `max` cannot be null and when they are null it is assumed they are not
/// known.
pub fn aexpr_to_skip_batch_predicate(
    e: Node,
    expr_arena: &mut Arena<AExpr>,
    schema: &Schema,
) -> Option<Node> {
    aexpr_to_skip_batch_predicate_rec(e, expr_arena, schema, 0)
}

/// Whether min/max statistics are usable for the given dtype, operator, and literal.
///
/// Rejects nested, null, and categorical types. For floats, Parquet stats exclude NaN
/// but data may contain it. Since NaN is largest under TotalOrd, `col < x` is safe
/// (NaN never matches) but `col > x` is not (NaN always matches).
/// col >= NaN matches NaN == NaN, but stats exclude NaN so max < NaN can't detect them -> unsafe
fn can_use_min_max_stats(
    dtype: &DataType,
    op: Option<&Operator>,
    lv: Option<&LiteralValue>,
) -> bool {
    if dtype.is_nested() || dtype.is_null() || dtype.is_categorical() {
        return false;
    }

    if !dtype.is_float() {
        return true;
    }

    let lv_is_nan = lv.is_some_and(|lv| lv.is_nan());

    use Operator as O;
    match op {
        Some(O::Lt | O::LtEq) => true,
        None | Some(O::Eq | O::EqValidity) => !lv_is_nan && lv.is_some(),
        Some(O::Gt) => lv_is_nan,
        _ => false,
    }
}

fn is_stat_defined(
    expr: impl IntoAExprBuilder,
    dtype: &DataType,
    arena: &mut Arena<AExpr>,
) -> AExprBuilder {
    let expr = expr.into_aexpr_builder();
    let mut result = expr.is_not_null(arena);
    if dtype.is_float() {
        let is_not_nan = expr.is_not_nan(arena);
        result = result.and(is_not_nan, arena);
    }
    result
}

#[recursive::recursive]
fn aexpr_to_skip_batch_predicate_rec(
    e: Node,
    arena: &mut Arena<AExpr>,
    schema: &Schema,
    depth: usize,
) -> Option<Node> {
    use Operator as O;

    macro_rules! rec {
        ($node:expr) => {{ aexpr_to_skip_batch_predicate_rec($node, arena, schema, depth + 1) }};
    }
    macro_rules! lv_cases {
        (
            $lv:expr, $lv_node:expr,
            null: $null_case:expr,
            not_null: $non_null_case:expr $(,)?
        ) => {{
            if let Some(lv) = $lv {
                if lv.is_null() {
                    $null_case
                } else {
                    $non_null_case
                }
            } else {
                let lv_node = AExprBuilder::new_from_node($lv_node);

                let lv_is_null = lv_node.has_nulls(arena);
                let lv_not_null = lv_node.has_no_nulls(arena);

                let null_case = lv_is_null.and($null_case, arena);
                let non_null_case = lv_not_null.and($non_null_case, arena);

                null_case.or(non_null_case, arena).node()
            }
        }};
    }
    macro_rules! col {
        (len) => {{ col!(PlSmallStr::from_static("len")) }};
        ($name:expr) => {{ AExprBuilder::new_from_node(arena.add(AExpr::Column($name))) }};
        (min: $name:expr) => {{ col!(format_pl_smallstr!("{}_min", $name)) }};
        (max: $name:expr) => {{ col!(format_pl_smallstr!("{}_max", $name)) }};
        (null_count: $name:expr) => {{ col!(format_pl_smallstr!("{}_nc", $name)) }};
    }
    macro_rules! lv {
        ($lv:expr) => {{ AExprBuilder::lit_scalar(Scalar::from($lv), arena) }};
        (idx: $lv:expr) => {{ AExprBuilder::lit_scalar(Scalar::new_idxsize($lv), arena) }};
    }

    let specialized = (|| {
        if let Some(Some(lv)) = constant_evaluate(e, arena, schema, 0) {
            if let Some(av) = lv.to_any_value() {
                return match av {
                    AnyValue::Null => Some(lv!(true).node()),
                    AnyValue::Boolean(b) => Some(lv!(!b).node()),
                    _ => None,
                };
            }
        }

        match arena.get(e) {
            AExpr::Element => None,
            AExpr::Explode { .. } => None,
            AExpr::Column(_) => None,
            #[cfg(feature = "dtype-struct")]
            AExpr::StructField(_) => None,
            AExpr::Literal(_) => None,
            AExpr::BinaryExpr { left, op, right } => {
                let left = *left;
                let right = *right;

                match op {
                    O::Eq | O::EqValidity => {
                        let ((target, _), (lv, lv_node)) =
                            get_binary_expr_target_and_lv(left, right, arena, schema)?;
                        let dtype = target_leaf_dtype(&target, schema)?;

                        if !can_use_min_max_stats(dtype, Some(op), lv.as_deref()) {
                            return None;
                        }

                        let op = *op;

                        // col(A) == B -> {
                        //     null_count(A) == 0                              , if B.is_null(),
                        //     null_count(A) == LEN || min(A) > B || max(A) < B, if B.is_not_null(),
                        // }

                        Some(lv_cases!(
                            lv, lv_node,
                            null: {
                                if matches!(op, O::Eq) {
                                    lv!(false).node()
                                } else {
                                    let col_nc = target.null_count(arena);
                                    let idx_zero = lv!(idx: 0);
                                    col_nc.eq(idx_zero, arena).node()
                                }
                            },
                            not_null: {
                                let col_min = target.min(arena);
                                let col_max = target.max(arena);

                                let min_is_defined = is_stat_defined(col_min.node(), dtype, arena);
                                let max_is_defined = is_stat_defined(col_max.node(), dtype, arena);

                                let min_gt = col_min.gt(lv_node, arena);
                                let min_gt = min_gt.and(min_is_defined, arena);

                                let max_lt = col_max.lt(lv_node, arena);
                                let max_lt = max_lt.and(max_is_defined, arena);

                                let col_nc = target.null_count(arena);
                                let len = col!(len);
                                let all_nulls = col_nc.eq(len, arena);

                                all_nulls.or(min_gt, arena).or(max_lt, arena).node()
                            }
                        ))
                    },
                    O::NotEq | O::NotEqValidity => {
                        let ((target, _), (lv, lv_node)) =
                            get_binary_expr_target_and_lv(left, right, arena, schema)?;
                        let dtype = target_leaf_dtype(&target, schema)?;

                        if !can_use_min_max_stats(dtype, Some(op), lv.as_deref()) {
                            return None;
                        }

                        let op = *op;

                        // col(A) != B -> {
                        //     null_count(A) == LEN                            , if B.is_null(),
                        //     null_count(A) == 0 && min(A) == B && max(A) == B, if B.is_not_null(),
                        // }

                        Some(lv_cases!(
                            lv, lv_node,
                            null: {
                                if matches!(op, O::NotEq) {
                                    lv!(false).node()
                                } else {
                                    let col_nc = target.null_count(arena);
                                    let len = col!(len);
                                    col_nc.eq(len, arena).node()
                                }
                            },
                            not_null: {
                                let col_min = target.min(arena);
                                let col_max = target.max(arena);
                                let min_eq = col_min.eq(lv_node, arena);
                                let max_eq = col_max.eq(lv_node, arena);

                                let col_nc = target.null_count(arena);
                                let idx_zero = lv!(idx: 0);
                                let no_nulls = col_nc.eq(idx_zero, arena);

                                no_nulls.and(min_eq, arena).and(max_eq, arena).node()
                            }
                        ))
                    },
                    O::Lt | O::Gt | O::LtEq | O::GtEq => {
                        let ((target, col_node), (lv, lv_node)) =
                            get_binary_expr_target_and_lv(left, right, arena, schema)?;
                        let dtype = target_leaf_dtype(&target, schema)?;
                        let col_is_left = col_node == left;

                        let effective_op = if col_is_left { *op } else { op.swap_operands() };
                        if !can_use_min_max_stats(dtype, Some(&effective_op), lv.as_deref()) {
                            return None;
                        }

                        let op = *op;
                        let lv_may_be_null = lv.is_none_or(|lv| lv.is_null());

                        // If B is null, this is always true.
                        //
                        // col(A) < B       ~       B > col(A)  ->
                        //     null_count(A) == LEN || min(A) >= B
                        //
                        // col(A) > B       ~       B < col(A)  ->
                        //     null_count(A) == LEN || max(A) <= B
                        //
                        // col(A) <= B      ~       B >= col(A) ->
                        //     null_count(A) == LEN || min(A) > B
                        //
                        // col(A) >= B      ~       B <= col(A) ->
                        //     null_count(A) == LEN || max(A) < B

                        let stat = match (op, col_is_left) {
                            (O::Lt | O::LtEq, true) | (O::Gt | O::GtEq, false) => target.min(arena),
                            (O::Lt | O::LtEq, false) | (O::Gt | O::GtEq, true) => target.max(arena),
                            _ => unreachable!(),
                        };
                        let cmp_op = match (op, col_is_left) {
                            (O::Lt, true) | (O::Gt, false) => O::GtEq,
                            (O::Lt, false) | (O::Gt, true) => O::LtEq,

                            (O::LtEq, true) | (O::GtEq, false) => O::Gt,
                            (O::LtEq, false) | (O::GtEq, true) => O::Lt,

                            _ => unreachable!(),
                        };

                        let stat_is_defined = is_stat_defined(stat, dtype, arena);
                        let cmp_op = stat.binary_op(lv_node, cmp_op, arena);
                        let mut expr = stat_is_defined.and(cmp_op, arena);

                        if lv_may_be_null {
                            let has_nulls = lv_node.into_aexpr_builder().has_nulls(arena);
                            expr = has_nulls.or(expr, arena);
                        }
                        Some(expr.node())
                    },

                    O::And | O::LogicalAnd => match (rec!(left), rec!(right)) {
                        (Some(left), Some(right)) => {
                            Some(AExprBuilder::new_from_node(left).or(right, arena).node())
                        },
                        (Some(n), None) | (None, Some(n)) => Some(n),
                        (None, None) => None,
                    },
                    O::Or | O::LogicalOr => {
                        let left = rec!(left)?;
                        let right = rec!(right)?;
                        Some(AExprBuilder::new_from_node(left).and(right, arena).node())
                    },

                    O::Plus
                    | O::Minus
                    | O::Multiply
                    | O::RustDivide
                    | O::TrueDivide
                    | O::FloorDivide
                    | O::Modulus
                    | O::Xor => None,
                }
            },
            AExpr::Cast { .. } => None,
            AExpr::Sort { .. } => None,
            AExpr::Gather { .. } => None,
            AExpr::SortBy { .. } => None,
            AExpr::Filter { .. } => None,
            AExpr::Agg(..) | AExpr::AnonymousAgg { .. } => None,
            AExpr::Ternary { .. } => None,
            AExpr::AnonymousFunction { .. } => None,
            AExpr::Eval { .. } => None,
            #[cfg(feature = "dtype-struct")]
            AExpr::StructEval { .. } => None,
            AExpr::Function {
                input, function, ..
            } => match function {
                IRFunctionExpr::Boolean(f) => match f {
                    #[cfg(feature = "is_in")]
                    IRBooleanFunction::IsIn { nulls_equal } => {
                        if !is_scalar_ae(input[1].node(), arena) {
                            return None;
                        }

                        let nulls_equal = *nulls_equal;
                        let lv_node = input[1].node();
                        match (
                            resolve_stat_target(input[0].node(), arena),
                            constant_evaluate(lv_node, arena, schema, 0),
                        ) {
                            (Some(target), Some(_)) => {
                                use polars_core::prelude::ExplodeOptions;

                                let dtype = target_leaf_dtype(&target, schema)?;
                                if !can_use_min_max_stats(dtype, None, None) {
                                    return None;
                                }

                                // col(A).is_in([B1, ..., Bn]) -> {
                                //     min(A) == max(A) && null_count(A) == 0 && !min(A).is_in(lv),
                                //     (!lv.has_nulls() || null_count(A) == 0) &&
                                //         (null_count(A) == LEN ||
                                //             AND_i (min(A) > Bi || max(A) < Bi)) , if N <= LIMIT,
                                //         (null_count(A) == LEN ||
                                //             min(A) > max(lv) || max(A) < min(lv)) , otherwise,
                                // }
                                // Branch 1 mirrors the generic min==max fallback for const-col rgs.
                                // `min(A) > X` / `max(A) < X` elide an `is_defined(stat) &&` guard.
                                // Helper drops haystack nulls; the `(!lv.has_nulls() || null_count(A)
                                // == 0)` gate is applied only when needed: per-case under
                                // nulls_equal=true in the unrolled path (when a null was dropped),
                                // and conditionally in the fallback path.
                                const LIST_ITEM_LIMIT: usize = 100;

                                let values = try_extract_is_in_haystack(
                                    lv_node,
                                    arena,
                                    schema,
                                    dtype,
                                    LIST_ITEM_LIMIT,
                                );

                                let lv_node = lv_node.into_aexpr_builder();
                                let lv_node_exploded = lv_node.explode(
                                    arena,
                                    ExplodeOptions {
                                        empty_as_null: false,
                                        keep_nulls: true,
                                    },
                                );

                                let col_min = target.min(arena);
                                let col_max = target.max(arena);
                                let min_is_defined = is_stat_defined(col_min, dtype, arena);
                                let max_is_defined = is_stat_defined(col_max, dtype, arena);

                                // all_nulls disjunct skips row groups whose column is entirely
                                // null in this group. Sound for both paths; the per-case
                                // null_count(A) == 0 guards suppress skip when haystack and column
                                // both have nulls under nulls_equal=true.
                                let col_nc = target.null_count(arena);
                                let len = col!(len);
                                let idx_zero = lv!(idx: 0);
                                let all_nulls = col_nc.eq(len, arena);
                                let col_has_no_nulls = col_nc.eq(idx_zero, arena);

                                let min_max_not_in = if let Some((values, had_nulls)) = values {
                                    // Per-element AND. Empty list folds to `true` (AND identity);
                                    // `is_in([])` is unconditionally false. The helper has dropped
                                    // any haystack nulls; the per-case `null_count(A) == 0` guard
                                    // below handles the column-has-nulls × haystack-has-null
                                    // interaction under nulls_equal=true.
                                    let inner = values.iter().fold(lv!(true), |acc, av| {
                                        let scalar = Scalar::new(dtype.clone(), av.into_static());
                                        let bi = AExprBuilder::lit_scalar(scalar, arena);
                                        let below =
                                            min_is_defined.and(col_min.gt(bi, arena), arena);
                                        let above =
                                            max_is_defined.and(col_max.lt(bi, arena), arena);
                                        acc.and(below.or(above, arena), arena)
                                    });
                                    let expr = all_nulls.or(inner, arena);
                                    if had_nulls && nulls_equal {
                                        expr.and(col_has_no_nulls, arena)
                                    } else {
                                        expr
                                    }
                                } else {
                                    let lv_min = lv_node_exploded.min(arena);
                                    let lv_max = lv_node_exploded.max(arena);
                                    let below =
                                        min_is_defined.and(col_min.gt(lv_max, arena), arena);
                                    let above =
                                        max_is_defined.and(col_max.lt(lv_min, arena), arena);
                                    let inner = below.or(above, arena);
                                    let expr = all_nulls.or(inner, arena);

                                    if nulls_equal {
                                        let lv_has_not_nulls = lv_node_exploded.has_no_nulls(arena);
                                        let null_case =
                                            lv_has_not_nulls.or(col_has_no_nulls, arena);
                                        null_case.and(expr, arena)
                                    } else {
                                        expr
                                    }
                                };

                                let min_is_max = col_min.eq(col_max, arena); // Eq so that (None == None) == None

                                // The above case does always cover the fallback path. Since there
                                // is code that relies on the `min==max` always filtering normally,
                                // we add it here.
                                let exact_not_in =
                                    col_min.is_in(lv_node, nulls_equal, arena).not(arena);
                                let exact_not_in = min_is_max
                                    .and(col_has_no_nulls, arena)
                                    .and(exact_not_in, arena);

                                Some(exact_not_in.or(min_max_not_in, arena).node())
                            },
                            _ => None,
                        }
                    },
                    IRBooleanFunction::Not => {
                        let target = resolve_stat_target(input[0].node(), arena)?;
                        let dtype = target_leaf_dtype(&target, schema)?;
                        if !matches!(dtype, DataType::Boolean) {
                            return None;
                        }

                        let col_nc = target.null_count(arena);
                        let len = col!(len);
                        let col_min = target.min(arena);
                        let col_min_is_true = col_min.eq(lv!(true), arena);
                        let min_is_defined = is_stat_defined(col_min, dtype, arena);

                        // col(A).Not() ->
                        //     null_count(A) == LEN || min(A) == True
                        let all_null = col_nc.eq(len, arena);
                        let min = min_is_defined.and(col_min_is_true, arena);

                        Some(all_null.or(min, arena).node())
                    },
                    IRBooleanFunction::IsNull => {
                        let target = resolve_stat_target(input[0].node(), arena)?;

                        // col(A).is_null() -> null_count(A) == 0
                        //
                        // For a struct target the row-level null count is unrecoverable from
                        // per-field counts. But a null struct nulls *every* field, so a single
                        // field that is never null proves no struct row is null either
                        // (null_count(field) bounds the struct null count from above). Skip when
                        // *any* per-field `null_count == 0`: the disjunction over the leaves.
                        #[cfg(feature = "dtype-struct")]
                        {
                            let dtype = target_leaf_dtype(&target, schema)?;
                            if matches!(dtype, DataType::Struct(_)) {
                                let mut leaves = Vec::new();
                                struct_leaf_targets(&target, dtype, &mut leaves);
                                let mut acc: Option<AExprBuilder> = None;
                                for leaf in &leaves {
                                    let leaf_nc = leaf.null_count(arena);
                                    let idx_zero = lv!(idx: 0);
                                    let term = leaf_nc.eq(idx_zero, arena);
                                    acc = Some(match acc {
                                        Some(acc) => acc.or(term, arena),
                                        None => term,
                                    });
                                }
                                // Empty struct: no leaves to reason about -> cannot skip.
                                return acc.map(|acc| acc.node());
                            }
                        }

                        let col_nc = target.null_count(arena);
                        let idx_zero = lv!(idx: 0);
                        Some(col_nc.eq(idx_zero, arena).node())
                    },
                    IRBooleanFunction::IsNotNull => {
                        let target = resolve_stat_target(input[0].node(), arena)?;

                        // A struct row with null *fields* is still a non-null *struct*, and the
                        // struct's row-level null count is unrecoverable from per-field counts, so
                        // whole-struct is_not_null cannot be pruned. A scalar struct *field* is
                        // fine: its per-field null count is exact.
                        #[cfg(feature = "dtype-struct")]
                        {
                            let dtype = target_leaf_dtype(&target, schema)?;
                            if matches!(dtype, DataType::Struct(_)) {
                                return None;
                            }
                        }

                        // col(A).is_not_null() -> null_count(A) == LEN
                        let col_nc = target.null_count(arena);
                        let len = col!(len);
                        Some(col_nc.eq(len, arena).node())
                    },
                    #[cfg(feature = "is_between")]
                    IRBooleanFunction::IsBetween { closed } => {
                        let target = resolve_stat_target(input[0].node(), arena)?;
                        let dtype = target_leaf_dtype(&target, schema)?;

                        // col(A).is_between(X, Y) ->
                        //     null_count(A) == LEN ||
                        //         min(A) >(=) Y ||
                        //         max(A) <(=) X

                        let left_node = input[1].node();
                        let right_node = input[2].node();

                        let left_lv = constant_evaluate(left_node, arena, schema, 0)?;
                        let right_lv = constant_evaluate(right_node, arena, schema, 0)?;

                        if !can_use_min_max_stats(dtype, None, left_lv.as_deref())
                            || !can_use_min_max_stats(dtype, None, right_lv.as_deref())
                        {
                            return None;
                        }

                        let closed = *closed;

                        let lhs_no_nulls = left_node.into_aexpr_builder().has_no_nulls(arena);
                        let rhs_no_nulls = right_node.into_aexpr_builder().has_no_nulls(arena);

                        let col_min = target.min(arena);
                        let col_max = target.max(arena);

                        use polars_ops::series::ClosedInterval;
                        let (left, right) = match closed {
                            ClosedInterval::Both => (O::Lt, O::Gt),
                            ClosedInterval::Left => (O::Lt, O::GtEq),
                            ClosedInterval::Right => (O::LtEq, O::Gt),
                            ClosedInterval::None => (O::LtEq, O::GtEq),
                        };

                        let left = col_max.binary_op(left_node, left, arena);
                        let right = col_min.binary_op(right_node, right, arena);

                        let min_is_defined = is_stat_defined(col_min, dtype, arena);
                        let max_is_defined = is_stat_defined(col_max, dtype, arena);

                        let left = max_is_defined.and(left, arena);
                        let right = min_is_defined.and(right, arena);

                        let interval = left.or(right, arena);
                        Some(
                            lhs_no_nulls
                                .and(rhs_no_nulls, arena)
                                .and(interval, arena)
                                .node(),
                        )
                    },
                    _ => None,
                },
                _ => None,
            },
            #[cfg(feature = "dynamic_group_by")]
            AExpr::Rolling { .. } => None,
            AExpr::Over { .. } => None,
            AExpr::Slice { .. } => None,
            AExpr::Len => None,
        }
    })();

    if let Some(specialized) = specialized {
        return Some(specialized);
    }

    // If we don't have a specialized implementation we can check if the whole block is constant
    // and fill that value in. This is especially useful when filtering hive partitions which are
    // filtered using this expression and which set their min == max.
    //
    // Essentially, what this does is
    //     E -> all(col(A_min) == col(A_max) & col(A_nc) == 0 for A in LIVE(E)) & ~(E)

    let live_columns = PlIndexMap::from_iter(aexpr_to_leaf_names_iter(e, arena).map(|col| {
        let min_name = format_pl_smallstr!("{col}_min");
        (col.clone(), min_name)
    }));

    // We cannot do proper equalities for these. For floats, min/max stats exclude
    // NaN, so substituting col=min doesn't account for hidden NaN values. Nested columns
    // (e.g. structs) carry struct-shaped stats that cannot be substituted as a whole; their
    // individual fields are handled by the specialized per-field handlers above.
    if live_columns.iter().any(|(c, _)| {
        schema
            .get(c)
            .is_none_or(|dt| dt.is_categorical() || dt.is_float() || dt.is_nested())
    }) {
        return None;
    }

    // Rename all uses of column names with the min value.
    let expr = rename_columns(e, arena, &live_columns);
    let mut expr = expr.into_aexpr_builder().not(arena);
    for col in live_columns.keys() {
        let col_min = col!(min: col);
        let col_max = col!(max: col);
        let col_nc = col!(null_count: col);

        let min_is_max = col_min.eq(col_max, arena); // Eq so that (None == None) == None
        let idx_zero = lv!(idx: 0);
        let has_no_nulls = col_nc.eq(idx_zero, arena);

        expr = min_is_max.and(has_no_nulls, arena).and(expr, arena);
    }
    Some(expr.node())
}
