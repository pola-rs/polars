use std::hash::{Hash, Hasher};
use std::sync::Arc;

use polars_utils::arena::Arena;

use super::*;
use crate::plans::{AExpr, IR};
use crate::prelude::aexpr::traverse_and_hash_aexpr;
use crate::prelude::ExprIR;

impl IRNode {
    pub(crate) fn hashable_and_cmp<'a>(
        &'a self,
        lp_arena: &'a Arena<IR>,
        expr_arena: &'a Arena<AExpr>,
    ) -> HashableEqLP<'a> {
        HashableEqLP {
            node: *self,
            lp_arena,
            expr_arena,
            ignore_cache: false,
        }
    }
}

pub(crate) struct HashableEqLP<'a> {
    node: IRNode,
    lp_arena: &'a Arena<IR>,
    expr_arena: &'a Arena<AExpr>,
    ignore_cache: bool,
}

impl HashableEqLP<'_> {
    /// When encountering a Cache node, ignore it and take the input.
    #[cfg(feature = "cse")]
    pub(crate) fn ignore_caches(mut self) -> Self {
        self.ignore_cache = true;
        self
    }
}

fn hash_option_expr<H: Hasher>(expr: &Option<ExprIR>, expr_arena: &Arena<AExpr>, state: &mut H) {
    if let Some(e) = expr {
        e.traverse_and_hash(expr_arena, state)
    }
}

fn hash_exprs<H: Hasher>(exprs: &[ExprIR], expr_arena: &Arena<AExpr>, state: &mut H) {
    for e in exprs {
        e.traverse_and_hash(expr_arena, state);
    }
}

impl Hash for HashableEqLP<'_> {
    // This hashes the variant, not the whole plan
    fn hash<H: Hasher>(&self, state: &mut H) {
        let alp = self.node.to_alp(self.lp_arena);
        std::mem::discriminant(alp).hash(state);
        match alp {
            #[cfg(feature = "python")]
            IR::PythonScan { .. } => {},
            IR::Slice {
                offset,
                len,
                input: _,
            } => {
                len.hash(state);
                offset.hash(state);
            },
            IR::Filter {
                input: _,
                predicate,
            } => {
                predicate.traverse_and_hash(self.expr_arena, state);
            },
            IR::Scan {
                paths,
                file_info: _,
                hive_parts: _,
                predicate,
                output_schema: _,
                scan_type,
                file_options,
            } => {
                // We don't have to traverse the schema, hive partitions etc. as they are derivative from the paths.
                scan_type.hash(state);
                paths.hash(state);
                hash_option_expr(predicate, self.expr_arena, state);
                file_options.hash(state);
            },
            IR::DataFrameScan {
                df,
                schema: _,
                output_schema,
                filter: selection,
            } => {
                (Arc::as_ptr(df) as usize).hash(state);
                output_schema.hash(state);
                hash_option_expr(selection, self.expr_arena, state);
            },
            IR::SimpleProjection { columns, input: _ } => {
                columns.hash(state);
            },
            IR::Select {
                input: _,
                expr,
                schema: _,
                options,
            } => {
                hash_exprs(expr, self.expr_arena, state);
                options.hash(state);
            },
            IR::Reduce {
                input: _,
                exprs,
                schema: _,
            } => {
                hash_exprs(exprs, self.expr_arena, state);
            },
            IR::Sort {
                input: _,
                by_column,
                slice,
                sort_options,
            } => {
                hash_exprs(by_column, self.expr_arena, state);
                slice.hash(state);
                sort_options.hash(state);
            },
            IR::GroupBy {
                input: _,
                keys,
                aggs,
                schema: _,
                apply,
                maintain_order,
                options,
            } => {
                hash_exprs(keys, self.expr_arena, state);
                hash_exprs(aggs, self.expr_arena, state);
                apply.is_none().hash(state);
                maintain_order.hash(state);
                options.hash(state);
            },
            IR::Join {
                input_left: _,
                input_right: _,
                schema: _,
                left_on,
                right_on,
                options,
            } => {
                hash_exprs(left_on, self.expr_arena, state);
                hash_exprs(right_on, self.expr_arena, state);
                options.hash(state);
            },
            IR::HStack {
                input: _,
                exprs,
                schema: _,
                options,
            } => {
                hash_exprs(exprs, self.expr_arena, state);
                options.hash(state);
            },
            IR::Distinct { input: _, options } => {
                options.hash(state);
            },
            IR::MapFunction { input: _, function } => {
                function.hash(state);
            },
            IR::Union { inputs: _, options } => options.hash(state),
            IR::HConcat {
                inputs: _,
                schema: _,
                options,
            } => {
                options.hash(state);
            },
            IR::ExtContext {
                input: _,
                contexts,
                schema: _,
            } => {
                for node in contexts {
                    traverse_and_hash_aexpr(*node, self.expr_arena, state);
                }
            },
            IR::Sink { input: _, payload } => {
                payload.hash(state);
            },
            IR::Cache {
                input: _,
                id,
                cache_hits,
            } => {
                id.hash(state);
                cache_hits.hash(state);
            },
            IR::Invalid => unreachable!(),
        }
    }
}

fn expr_irs_eq(l: &[ExprIR], r: &[ExprIR], expr_arena: &Arena<AExpr>) -> bool {
    l.len() == r.len() && l.iter().zip(r).all(|(l, r)| expr_ir_eq(l, r, expr_arena))
}

fn expr_ir_eq(l: &ExprIR, r: &ExprIR, expr_arena: &Arena<AExpr>) -> bool {
    l.get_alias() == r.get_alias() && {
        let l = AexprNode::new(l.node());
        let r = AexprNode::new(r.node());
        l.hashable_and_cmp(expr_arena) == r.hashable_and_cmp(expr_arena)
    }
}

fn opt_expr_ir_eq(l: &Option<ExprIR>, r: &Option<ExprIR>, expr_arena: &Arena<AExpr>) -> bool {
    match (l, r) {
        (None, None) => true,
        (Some(l), Some(r)) => expr_ir_eq(l, r, expr_arena),
        _ => false,
    }
}

impl HashableEqLP<'_> {
    fn is_equal(&self, other: &Self) -> bool {
        let alp_l = self.node.to_alp(self.lp_arena);
        let alp_r = other.node.to_alp(self.lp_arena);
        if std::mem::discriminant(alp_l) != std::mem::discriminant(alp_r) {
            return false;
        }
        match (alp_l, alp_r) {
            (
                IR::Slice {
                    input: _,
                    offset: ol,
                    len: ll,
                },
                IR::Slice {
                    input: _,
                    offset: or,
                    len: lr,
                },
            ) => ol == or && ll == lr,
            (
                IR::Filter {
                    input: _,
                    predicate: l,
                },
                IR::Filter {
                    input: _,
                    predicate: r,
                },
            ) => expr_ir_eq(l, r, self.expr_arena),
            (
                IR::Scan {
                    paths: pl,
                    file_info: _,
                    hive_parts: _,
                    predicate: pred_l,
                    output_schema: _,
                    scan_type: stl,
                    file_options: ol,
                },
                IR::Scan {
                    paths: pr,
                    file_info: _,
                    hive_parts: _,
                    predicate: pred_r,
                    output_schema: _,
                    scan_type: str,
                    file_options: or,
                },
            ) => {
                pl == pr
                    && stl == str
                    && ol == or
                    && opt_expr_ir_eq(pred_l, pred_r, self.expr_arena)
            },
            (
                IR::DataFrameScan {
                    df: dfl,
                    schema: _,
                    output_schema: s_l,
                    filter: sl,
                },
                IR::DataFrameScan {
                    df: dfr,
                    schema: _,
                    output_schema: s_r,
                    filter: sr,
                },
            ) => {
                Arc::as_ptr(dfl) == Arc::as_ptr(dfr)
                    && s_l == s_r
                    && opt_expr_ir_eq(sl, sr, self.expr_arena)
            },
            (
                IR::SimpleProjection {
                    input: _,
                    columns: cl,
                },
                IR::SimpleProjection {
                    input: _,
                    columns: cr,
                },
            ) => cl == cr,
            (
                IR::Select {
                    input: _,
                    expr: el,
                    options: ol,
                    schema: _,
                },
                IR::Select {
                    input: _,
                    expr: er,
                    options: or,
                    schema: _,
                },
            ) => ol == or && expr_irs_eq(el, er, self.expr_arena),
            (
                IR::Sort {
                    input: _,
                    by_column: cl,
                    slice: l_slice,
                    sort_options: l_options,
                },
                IR::Sort {
                    input: _,
                    by_column: cr,
                    slice: r_slice,
                    sort_options: r_options,
                },
            ) => {
                (l_slice == r_slice && l_options == r_options)
                    && expr_irs_eq(cl, cr, self.expr_arena)
            },
            (
                IR::GroupBy {
                    input: _,
                    keys: keys_l,
                    aggs: aggs_l,
                    schema: _,
                    apply: apply_l,
                    maintain_order: maintain_l,
                    options: ol,
                },
                IR::GroupBy {
                    input: _,
                    keys: keys_r,
                    aggs: aggs_r,
                    schema: _,
                    apply: apply_r,
                    maintain_order: maintain_r,
                    options: or,
                },
            ) => {
                apply_l.is_none()
                    && apply_r.is_none()
                    && ol == or
                    && maintain_l == maintain_r
                    && expr_irs_eq(keys_l, keys_r, self.expr_arena)
                    && expr_irs_eq(aggs_l, aggs_r, self.expr_arena)
            },
            (
                IR::Join {
                    input_left: _,
                    input_right: _,
                    schema: _,
                    left_on: ll,
                    right_on: rl,
                    options: ol,
                },
                IR::Join {
                    input_left: _,
                    input_right: _,
                    schema: _,
                    left_on: lr,
                    right_on: rr,
                    options: or,
                },
            ) => {
                ol == or
                    && expr_irs_eq(ll, lr, self.expr_arena)
                    && expr_irs_eq(rl, rr, self.expr_arena)
            },
            (
                IR::HStack {
                    input: _,
                    exprs: el,
                    schema: _,
                    options: ol,
                },
                IR::HStack {
                    input: _,
                    exprs: er,
                    schema: _,
                    options: or,
                },
            ) => ol == or && expr_irs_eq(el, er, self.expr_arena),
            (
                IR::Distinct {
                    input: _,
                    options: ol,
                },
                IR::Distinct {
                    input: _,
                    options: or,
                },
            ) => ol == or,
            (
                IR::MapFunction {
                    input: _,
                    function: l,
                },
                IR::MapFunction {
                    input: _,
                    function: r,
                },
            ) => l == r,
            (
                IR::Union {
                    inputs: _,
                    options: l,
                },
                IR::Union {
                    inputs: _,
                    options: r,
                },
            ) => l == r,
            (
                IR::HConcat {
                    inputs: _,
                    schema: _,
                    options: l,
                },
                IR::HConcat {
                    inputs: _,
                    schema: _,
                    options: r,
                },
            ) => l == r,
            (
                IR::ExtContext {
                    input: _,
                    contexts: l,
                    schema: _,
                },
                IR::ExtContext {
                    input: _,
                    contexts: r,
                    schema: _,
                },
            ) => {
                l.len() == r.len()
                    && l.iter().zip(r.iter()).all(|(l, r)| {
                        let l = AexprNode::new(*l).hashable_and_cmp(self.expr_arena);
                        let r = AexprNode::new(*r).hashable_and_cmp(self.expr_arena);
                        l == r
                    })
            },
            _ => false,
        }
    }
}

impl PartialEq for HashableEqLP<'_> {
    fn eq(&self, other: &Self) -> bool {
        let mut scratch_1 = vec![];
        let mut scratch_2 = vec![];

        scratch_1.push(self.node.node());
        scratch_2.push(other.node.node());

        loop {
            match (scratch_1.pop(), scratch_2.pop()) {
                (Some(l), Some(r)) => {
                    let l = IRNode::new(l);
                    let r = IRNode::new(r);
                    let l_alp = l.to_alp(self.lp_arena);
                    let r_alp = r.to_alp(self.lp_arena);

                    if self.ignore_cache {
                        match (l_alp, r_alp) {
                            (IR::Cache { input: l, .. }, IR::Cache { input: r, .. }) => {
                                scratch_1.push(*l);
                                scratch_2.push(*r);
                                continue;
                            },
                            (IR::Cache { input: l, .. }, _) => {
                                scratch_1.push(*l);
                                scratch_2.push(r.node());
                                continue;
                            },
                            (_, IR::Cache { input: r, .. }) => {
                                scratch_1.push(l.node());
                                scratch_2.push(*r);
                                continue;
                            },
                            _ => {},
                        }
                    }

                    if !l
                        .hashable_and_cmp(self.lp_arena, self.expr_arena)
                        .is_equal(&r.hashable_and_cmp(self.lp_arena, self.expr_arena))
                    {
                        return false;
                    }

                    l_alp.copy_inputs(&mut scratch_1);
                    r_alp.copy_inputs(&mut scratch_2);
                },
                (None, None) => return true,
                _ => return false,
            }
        }
    }
}
