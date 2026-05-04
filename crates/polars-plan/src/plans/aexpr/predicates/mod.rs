mod column_expr;
mod skip_batches;

use std::borrow::Cow;

pub use column_expr::*;
#[cfg(feature = "is_in")]
use polars_core::prelude::{AnyValue, DataType, Series};
use polars_core::schema::Schema;
use polars_utils::arena::{Arena, Node};
use polars_utils::pl_str::PlSmallStr;
pub use skip_batches::*;

use super::evaluate::{constant_evaluate, into_column};
use super::{AExpr, LiteralValue};

#[allow(clippy::type_complexity)]
fn get_binary_expr_col_and_lv<'a>(
    left: Node,
    right: Node,
    expr_arena: &'a Arena<AExpr>,
    schema: &Schema,
) -> Option<(
    (&'a PlSmallStr, Node),
    (Option<Cow<'a, LiteralValue>>, Node),
)> {
    match (
        into_column(left, expr_arena),
        into_column(right, expr_arena),
        constant_evaluate(left, expr_arena, schema, 0),
        constant_evaluate(right, expr_arena, schema, 0),
    ) {
        (Some(col), _, _, Some(lv)) => Some(((col, left), (lv, right))),
        (_, Some(col), Some(lv), _) => Some(((col, right), (lv, left))),
        _ => None,
    }
}

/// Extract the haystack of `col.is_in(haystack)` as a flat element [`Series`]
/// at planner time, dropping any nulls (pyarrow-style). The returned `bool`
/// is `true` iff a null was dropped; under `nulls_equal=true` the caller must
/// compensate (typically AND'ing with `null_count(col) == 0`).
///
/// Returns `None` (signals the caller to use its fallback path) when any of:
/// - `lv_node` is not an `AExpr::Literal` whose value is `AnyValue::List(_)`
///   (or `AnyValue::Array(_, _)` under `dtype-array`).
/// - The list's element dtype is not equal to `column_dtype`.
/// - The list has more than `max_len` elements after null filtering.
///
/// Shared by [`skip_batches`] (skip-batch predicate) and [`column_expr`]
/// (per-column predicate).
#[cfg(feature = "is_in")]
pub(super) fn try_extract_is_in_haystack(
    lv_node: Node,
    expr_arena: &Arena<AExpr>,
    schema: &Schema,
    column_dtype: &DataType,
    max_len: usize,
) -> Option<(Series, bool)> {
    let lv = constant_evaluate(lv_node, expr_arena, schema, 0)??;
    let av = lv.to_any_value()?;
    let s = match av {
        AnyValue::List(s) => s,
        #[cfg(feature = "dtype-array")]
        AnyValue::Array(s, _) => s,
        _ => return None,
    };
    if s.dtype() != column_dtype {
        return None;
    }
    let had_nulls = s.has_nulls();
    let s = if had_nulls { s.drop_nulls() } else { s };
    if s.len() > max_len {
        return None;
    }
    Some((s, had_nulls))
}
