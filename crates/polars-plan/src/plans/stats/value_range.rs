use polars_defs::join::JoinType;
use polars_utils::arena::{Arena, Node};
use recursive::recursive;

use crate::plans::schema::join_right_output_names;
use crate::plans::{AExpr, IR, JoinTypeOptionsIR, into_column};

/// An inclusive range that holds every non-null value of column `name` of `node`, on
/// its physical values.
///
/// The range comes from the statistics of the scan the column is read from, when they
/// cover all of its data. Filters, slices, sorts, group-by keys and joins only drop or
/// repeat values, so the range of the scan still holds above them. A computed column
/// has no range.
///
/// This doesn't use `node_stats`, whose ranges may be estimates. A range here must
/// hold every value, as packed keys rely on it.
#[recursive]
pub fn column_value_range(
    node: Node,
    name: &str,
    ir_arena: &Arena<IR>,
    expr_arena: &Arena<AExpr>,
) -> Option<(i128, i128)> {
    let recurse = |node: Node, name: &str| column_value_range(node, name, ir_arena, expr_arena);
    match ir_arena.get(node) {
        IR::Scan {
            file_info,
            hive_parts,
            ..
        } => {
            // Hive values replace the column values of the files.
            if hive_parts
                .as_ref()
                .is_some_and(|h| h.schema().contains(name))
            {
                return None;
            }
            // Files of one scan may store a `Datetime`, `Duration` or `Time` in
            // different units, and their ranges are merged as stored.
            let dtype = file_info.schema.get(name)?;
            if !(dtype.is_integer() || dtype.is_date()) {
                return None;
            }
            let stats = file_info.stats.columns.as_ref()?.get(name)?;
            let (min, max) = stats.int_range?;
            (min <= max && !stats.int_range_partial && !stats.int_range_sampled)
                .then_some((min, max))
        },
        IR::Filter { input, .. }
        | IR::Slice { input, .. }
        | IR::Sort { input, .. }
        | IR::Cache { input, .. }
        | IR::SimpleProjection { input, .. }
        | IR::Distinct { input, .. } => recurse(*input, name),
        IR::Select { input, expr, .. } => {
            let e = expr.iter().find(|e| e.output_name() == name)?;
            recurse(*input, into_column(e.node(), expr_arena)?)
        },
        IR::HStack { input, exprs, .. } => match exprs.iter().find(|e| e.output_name() == name) {
            Some(e) => recurse(*input, into_column(e.node(), expr_arena)?),
            None => recurse(*input, name),
        },
        IR::GroupBy {
            input,
            keys,
            options,
            apply,
            ..
        } => {
            if apply.is_some() || options.is_rolling() || options.is_dynamic() {
                return None;
            }
            let key = keys.iter().find(|k| k.output_name() == name)?;
            recurse(*input, into_column(key.node(), expr_arena)?)
        },
        IR::Union { inputs, .. } => inputs.iter().try_fold(None, |acc, input| {
            let (min, max) = recurse(*input, name)?;
            Some(Some(match acc {
                None => (min, max),
                Some((lo, hi)) => (min.min(lo), max.max(hi)),
            }))
        })?,
        IR::Join {
            input_left,
            input_right,
            options,
            ..
        } => {
            let how = &options.args.how;
            if how.is_semi_anti() {
                return recurse(*input_left, name);
            }
            if !how.is_equi() {
                return None;
            }
            let JoinTypeOptionsIR::Equi { on, .. } = &options.options else {
                return None;
            };
            let schema_left = ir_arena.get(*input_left).schema(ir_arena);
            let schema_right = ir_arena.get(*input_right).schema(ir_arena);
            let right_names = join_right_output_names(&schema_left, &schema_right, options).ok()?;
            let from_right = || {
                let (right_name, _) = schema_right
                    .iter_names()
                    .zip(&right_names)
                    .find(|(_, out)| out.as_deref() == Some(name))?;
                recurse(*input_right, right_name)
            };
            let coalesced = options.args.should_coalesce();
            if coalesced && matches!(how, JoinType::Right) {
                // The left key columns are dropped, so a right column wins.
                if right_names.iter().flatten().any(|out| out == name) {
                    return from_right();
                }
                if on.iter().any(|(l, _)| l.output_name() == name) {
                    return None;
                }
                return recurse(*input_left, name);
            }
            if !schema_left.contains(name) {
                return from_right();
            }
            let left = recurse(*input_left, name)?;
            match on.iter().find(|(l, _)| l.output_name() == name) {
                // A full join's coalesced key holds the right keys of its unmatched rows.
                Some((l, r)) if coalesced && matches!(how, JoinType::Full) => {
                    into_column(l.node(), expr_arena)?;
                    let right = recurse(*input_right, into_column(r.node(), expr_arena)?)?;
                    Some((left.0.min(right.0), left.1.max(right.1)))
                },
                _ => Some(left),
            }
        },
        _ => None,
    }
}
