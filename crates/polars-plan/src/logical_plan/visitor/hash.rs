use std::hash::{Hash, Hasher};
use std::sync::Arc;
use polars_utils::arena::Arena;
use crate::logical_plan::{AExpr, ALogicalPlan};
use crate::prelude::ExprIR;
use super::*;

struct HashableAlpNode<'a> {
    node: ALogicalPlanNode,
    expr_arena: &'a Arena<AExpr>
}

fn hash_option_expr<H: Hasher>(expr: &Option<ExprIR>, expr_arena: &Arena<AExpr>, state: &mut H)  {
    if let Some(e) = expr {
        e.traverse_and_hash(expr_arena, state)
    }
}

fn hash_exprs<H: Hasher>(exprs: &[ExprIR], expr_arena: &Arena<AExpr>, state: &mut H) {
    for e in exprs {
        e.traverse_and_hash(expr_arena, state);
    }
}

impl Hash for HashableAlpNode<'_> {
    // This hashes the variant, not the whole plan
    fn hash<H: Hasher>(&self, state: &mut H) {
        let alp = self.node.to_alp();
        std::mem::discriminant(alp).hash(state);
        match alp {
            #[cfg(feature = "python")]
            ALogicalPlan::PythonScan {.. } => {}
            ALogicalPlan::Slice { offset, len, input: _ } => {
                len.hash(state);
                offset.hash(state);
            }
            ALogicalPlan::Selection { input: _, predicate } => {
                predicate.traverse_and_hash(self.expr_arena, state);
            }
            ALogicalPlan::Scan { paths, file_info: _, predicate, output_schema: _, scan_type, file_options, } => {
                // We don't have to traverse the schema, hive partitions etc. as they are derivative from the paths.
                scan_type.hash(state);
                paths.hash(state);
                hash_option_expr(predicate, self.expr_arena, state);
                file_options.hash(state);
            }
            ALogicalPlan::DataFrameScan { df, schema:_, output_schema: _, projection, selection } => {
                (Arc::as_ptr(df) as usize).hash(state);
                projection.hash(state);
                hash_option_expr(selection, self.expr_arena, state);
            }
            ALogicalPlan::SimpleProjection { columns, duplicate_check, input: _ } => {
                columns.hash(state);
                duplicate_check.hash(state);
            }
            ALogicalPlan::Projection { input: _, expr, schema: _, options } => {
                hash_exprs(expr.default_exprs(), self.expr_arena, state);
                options.hash(state);
            }
            ALogicalPlan::Sort { input: _, by_column, args } => {
                hash_exprs(by_column, self.expr_arena, state);
                args.hash(state);
            }
            ALogicalPlan::Aggregate { input:_, keys, aggs, schema: _, apply, maintain_order, options } => {
                hash_exprs(keys, self.expr_arena, state);
                hash_exprs(aggs, self.expr_arena, state);
                apply.is_none().hash(state);
                maintain_order.hash(state);
                options.hash(state);
            }
            ALogicalPlan::Join { input_left: _, input_right: _, schema: _, left_on, right_on, options } => {
                hash_exprs(left_on, self.expr_arena, state);
                hash_exprs(right_on, self.expr_arena, state);
                options.hash(state);
            }
            ALogicalPlan::HStack { input:_, exprs, schema: _, options } => {
                hash_exprs(exprs.default_exprs(), self.expr_arena, state);
                options.hash(state);
            }
            ALogicalPlan::Distinct { input: _, options } => {
                options.hash(state);
            }
            ALogicalPlan::MapFunction { input: _, function } => {
                function.has
            }
            ALogicalPlan::Union { .. } => {}
            ALogicalPlan::HConcat { .. } => {}
            ALogicalPlan::ExtContext { .. } => {}
            ALogicalPlan::Sink { .. } => {}
            ALogicalPlan::Invalid => {}
            ALogicalPlan::Cache { input: _, id, count } => {
                id.hash(state);
                count.hash(state);
            }
        }
    }
}
