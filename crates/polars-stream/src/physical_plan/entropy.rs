//! Building blocks of the single-pass `entropy` rewrite, shared by select and group-by lowering.
use polars_core::chunked_array::cast::CastOptions;
use polars_core::prelude::DataType;
use polars_core::scalar::Scalar;
use polars_plan::plans::{AExpr, AExprBuilder, IRFunctionExpr};
use polars_utils::arena::Arena;

/// A literal in `dtype`, so that it doesn't widen the expression it is used in.
fn float_lit(v: f64, dtype: &DataType, arena: &mut Arena<AExpr>) -> AExprBuilder {
    let scalar = Scalar::from(v)
        .cast_with_options(dtype, CastOptions::Strict)
        .unwrap();
    AExprBuilder::lit_scalar(scalar, arena)
}

/// Casts the entropy input `x` to `dtype`, the float dtype the result is computed in.
pub(crate) fn entropy_input(
    x: AExprBuilder,
    x_dtype: &DataType,
    dtype: &DataType,
    arena: &mut Arena<AExpr>,
) -> AExprBuilder {
    if x_dtype == dtype {
        x
    } else {
        x.cast(dtype.clone(), arena)
    }
}

/// `x * log(x, base)`, with `0 * log(0) = 0`.
pub(crate) fn x_log_x(
    x: AExprBuilder,
    base: f64,
    dtype: &DataType,
    arena: &mut Arena<AExpr>,
) -> AExprBuilder {
    let zero = float_lit(0.0, dtype, arena);
    let base = float_lit(base, dtype, arena);
    let log_x = AExprBuilder::function(
        vec![x.expr_ir_unnamed(), base.expr_ir_unnamed()],
        IRFunctionExpr::Log,
        arena,
    );
    let x_log_x = x.multiply(log_x, arena);
    let is_zero = x.eq(zero, arena);
    AExprBuilder::when_then_otherwise(is_zero, zero, x_log_x, arena)
}

/// Combines `sum(x_log_x(x))` and, if normalizing, `(sum(x), count(x))` into the entropy.
pub(crate) fn entropy_from_sums(
    sum_x_log_x: AExprBuilder,
    normalize: Option<(AExprBuilder, AExprBuilder)>,
    base: f64,
    dtype: &DataType,
    arena: &mut Arena<AExpr>,
) -> AExprBuilder {
    let zero = float_lit(0.0, dtype, arena);
    let Some((sum_x, count)) = normalize else {
        // `0 - t` instead of `-t`, so that a zero result is `+0.0`.
        return zero.minus(sum_x_log_x, arena);
    };

    // -sum(x/s * log(x/s)) = log(s) - sum(x * log(x)) / s, with s = sum(x).
    let base = float_lit(base, dtype, arena);
    let log_sum_x = AExprBuilder::function(
        vec![sum_x.expr_ir_unnamed(), base.expr_ir_unnamed()],
        IRFunctionExpr::Log,
        arena,
    );
    let entropy = log_sum_x.minus(sum_x_log_x.true_divide(sum_x, arena), arena);
    let is_empty = count.eq(
        AExprBuilder::lit_scalar(Scalar::new_idxsize(0), arena),
        arena,
    );
    // Adding `+0.0` turns `-0.0` (`log(1)` for a base below 1) into `+0.0`.
    AExprBuilder::when_then_otherwise(is_empty, zero, entropy, arena).plus(zero, arena)
}
