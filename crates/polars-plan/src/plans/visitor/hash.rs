use std::hash::{Hash, Hasher};
use std::sync::Arc;

use polars_utils::arena::Arena;

use super::*;
#[cfg(feature = "python")]
use crate::plans::PythonOptions;
use crate::plans::{AExpr, IR};
use crate::prelude::ExprIR;
use crate::prelude::aexpr::traverse_and_hash_aexpr;

impl IRNode {
    pub(crate) fn hashable_and_cmp<'a>(
        &'a self,
        lp_arena: &'a Arena<IR>,
        expr_arena: &'a Arena<AExpr>,
    ) -> IRHashWrap<'a> {
        IRHashWrap {
            node: *self,
            lp_arena,
            expr_arena,
            hash_as_equality: false,
        }
    }
}

pub(crate) struct IRHashWrap<'a> {
    node: IRNode,
    lp_arena: &'a Arena<IR>,
    expr_arena: &'a Arena<AExpr>,
    hash_as_equality: bool,
}

impl IRHashWrap<'_> {
    pub fn hash_as_equality(mut self) -> Self {
        self.hash_as_equality = true;
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

/// Specialized Hash that dispatches to `ExprIR::traverse_and_hash` instead of just hashing
/// the `Node`.
#[cfg(feature = "python")]
fn hash_python_predicate<H: Hasher>(
    pred: &crate::prelude::PythonPredicate,
    expr_arena: &Arena<AExpr>,
    state: &mut H,
) {
    use crate::prelude::PythonPredicate;
    std::mem::discriminant(pred).hash(state);
    match pred {
        PythonPredicate::None => {},
        PythonPredicate::PyArrow(s) => s.hash(state),
        PythonPredicate::Polars(e) => e.traverse_and_hash(expr_arena, state),
    }
}

impl Hash for IRHashWrap<'_> {
    // This hashes the variant, not the whole plan
    fn hash<H: Hasher>(&self, state: &mut H) {
        let alp = self.node.to_alp(self.lp_arena);
        std::mem::discriminant(alp).hash(state);
        match alp {
            #[cfg(feature = "python")]
            IR::PythonScan {
                options:
                    PythonOptions {
                        scan_fn,
                        schema,
                        output_schema,
                        with_columns,
                        python_source,
                        n_rows,
                        predicate,
                        validate_schema,
                        is_pure,
                    },
            } => {
                // Hash the Python function object using the pointer to the object
                // This should be the same as calling id() in python, but we don't need the GIL

                use std::sync::atomic::AtomicU64;
                static UNIQUE_COUNT: AtomicU64 = AtomicU64::new(0);
                if let Some(scan_fn) = scan_fn {
                    let ptr_addr = scan_fn.0.as_ptr() as usize;
                    ptr_addr.hash(state);
                }
                // Hash the stable fields
                // We include the schema since it can be set by the user
                schema.hash(state);
                output_schema.hash(state);
                with_columns.hash(state);
                python_source.hash(state);
                n_rows.hash(state);
                hash_python_predicate(predicate, self.expr_arena, state);
                validate_schema.hash(state);

                if self.hash_as_equality && !*is_pure {
                    let val = UNIQUE_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    val.hash(state)
                } else {
                    is_pure.hash(state)
                }
            },
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
                sources,
                file_info: _,
                hive_parts: _,
                predicate,
                predicate_file_skip_applied: _,
                output_schema: _,
                scan_type,
                unified_scan_args,
            } => {
                // We don't have to traverse the schema, hive partitions etc. as they are derivative from the paths.
                scan_type.hash(state);
                sources.hash(state);
                hash_option_expr(predicate, self.expr_arena, state);
                unified_scan_args.hash(state);
            },
            IR::DataFrameScan {
                df,
                schema: _,
                output_schema,
                ..
            } => {
                (Arc::as_ptr(df) as usize).hash(state);
                output_schema.hash(state);
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
                payload.traverse_and_hash(self.expr_arena, state);
            },
            IR::SinkMultiple { inputs: _ } => {},
            IR::Cache { input: _, id } => {
                id.hash(state);
            },
            #[cfg(feature = "merge_sorted")]
            IR::MergeSorted {
                input_left: _,
                input_right: _,
                key,
            } => {
                key.hash(state);
            },
            IR::Invalid => unreachable!(),
        }
    }
}
