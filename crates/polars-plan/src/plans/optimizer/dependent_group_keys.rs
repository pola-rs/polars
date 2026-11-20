//! A group-by key that is computed row by row from other key columns can't split a group, so it
//! is computed from those columns after the aggregation.

use polars_core::prelude::*;
use polars_utils::arena::{Arena, Node};

use super::OptimizationRule;
use crate::plans::iterator::ArenaExprIter;
use crate::plans::{
    AExpr, ExprIR, IR, IRFunctionExpr, OutputName, is_elementwise_rec,
    is_inherently_nondeterministic,
};
use crate::prelude::ProjectionOptions;

pub struct DependentGroupKeys;

impl OptimizationRule for DependentGroupKeys {
    fn optimize_plan(
        &mut self,
        lp_arena: &mut Arena<IR>,
        expr_arena: &mut Arena<AExpr>,
        node: Node,
    ) -> PolarsResult<Option<IR>> {
        let IR::GroupBy {
            input,
            keys,
            aggs,
            schema,
            maintain_order,
            options,
            apply: None,
        } = lp_arena.get(node)
        else {
            return Ok(None);
        };
        if keys.len() < 2 || options.is_dynamic() || options.is_rolling() {
            return Ok(None);
        }

        let with_columns = match lp_arena.get(*input) {
            IR::HStack { exprs, .. } => exprs.as_slice(),
            _ => &[],
        };
        let computed_below =
            |name: &PlSmallStr| with_columns.iter().find(|e| e.output_name() == name);
        // Keys that are input columns, which are the same in the input of the `with_columns`.
        let column_keys: PlIndexSet<PlSmallStr> = keys
            .iter()
            .filter_map(|k| k.plain_column(expr_arena))
            .filter(|name| {
                computed_below(name).is_none() && schema.get(name).is_some_and(equal_is_identical)
            })
            .cloned()
            .collect();
        if column_keys.is_empty() {
            return Ok(None);
        }

        // The expression each dependent key is computed with after the aggregation.
        let mut dependent: PlIndexMap<PlSmallStr, Node> = PlIndexMap::new();
        for k in keys {
            let computed = match k.plain_column(expr_arena) {
                Some(name) => match computed_below(name) {
                    Some(e) => e.node(),
                    None => continue,
                },
                None => k.node(),
            };
            if reads_only(computed, &column_keys, expr_arena) {
                dependent.insert(k.output_name().clone(), computed);
            }
        }
        if dependent.is_empty() {
            return Ok(None);
        }

        let group_by = IR::GroupBy {
            input: *input,
            keys: keys
                .iter()
                .filter(|k| !dependent.contains_key(k.output_name()))
                .cloned()
                .collect(),
            aggs: aggs.clone(),
            schema: Arc::new(
                schema
                    .iter()
                    .filter(|(name, _)| !dependent.contains_key(*name))
                    .map(|(name, dtype)| (name.clone(), dtype.clone()))
                    .collect(),
            ),
            maintain_order: *maintain_order,
            options: options.clone(),
            apply: None,
        };
        let schema = schema.clone();
        let expr = schema
            .iter_names()
            .map(|name| match dependent.get(name) {
                Some(node) => ExprIR::new(*node, OutputName::Alias(name.clone())),
                None => ExprIR::from_column_name(name.clone(), expr_arena),
            })
            .collect();
        Ok(Some(IR::Select {
            input: lp_arena.add(group_by),
            expr,
            schema,
            options: ProjectionOptions::default(),
        }))
    }
}

/// Whether `node` is a row-by-row expression without random values or user functions that reads
/// one or more of `columns` and nothing else.
fn reads_only(node: Node, columns: &PlIndexSet<PlSmallStr>, expr_arena: &Arena<AExpr>) -> bool {
    let mut reads_column = false;
    for (_, ae) in expr_arena.iter(node) {
        match ae {
            AExpr::Column(name) if columns.contains(name) => reads_column = true,
            AExpr::Literal(lv) if lv.is_scalar() => {},
            AExpr::Column(_) | AExpr::Literal(_) | AExpr::Element | AExpr::Len => return false,
            // Only holds in the order of the rows before the aggregation.
            AExpr::Function {
                function: IRFunctionExpr::SetSortedFlag(_),
                ..
            } => return false,
            _ => {},
        }
    }
    reads_column
        && is_elementwise_rec(node, expr_arena)
        && !is_inherently_nondeterministic(node, expr_arena)
}

/// Types where values in one group are the same value. Not floats: `-0.0` and `0.0` are one group,
/// but an expression can tell them apart.
fn equal_is_identical(dtype: &DataType) -> bool {
    dtype.is_integer()
        || dtype.is_bool()
        || dtype.is_string()
        || dtype.is_binary()
        || dtype.is_temporal()
        || dtype.is_decimal()
        || dtype.is_categorical()
        || dtype.is_enum()
}
