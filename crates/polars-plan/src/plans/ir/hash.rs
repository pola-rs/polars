use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::plans::{ExprIR, IR};
#[cfg(feature = "python")]
use crate::plans::{PythonOptions, PythonPredicate};
use crate::prelude::{PlanCallback, UnoptimizedOperation};

pub trait ExpressionHasher {
    fn hash_expr<H: Hasher>(&self, expr: &ExprIR, state: &mut H);
}

impl IR {
    /// Hash the contents of the enum, without descending into child IR nodes.
    /// The user can choose how to hash the referenced ExprIR nodes by providing `expr_hash`
    pub(crate) fn hash_excluding_inputs<H: Hasher>(
        &self,
        state: &mut H,
        expr_hash: &impl ExpressionHasher,
    ) {
        let hash_exprs = |exprs: &[ExprIR], state: &mut H| {
            for e in exprs {
                expr_hash.hash_expr(e, state);
            }
        };

        std::mem::discriminant(self).hash(state);
        match self {
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
                // Hash the Python function object using the pointer to the object.
                // This should be the same as calling id() in python, but we don't need the GIL.
                if let Some(scan_fn) = scan_fn {
                    let ptr_addr = scan_fn.0.as_ptr() as usize;
                    ptr_addr.hash(state);
                }
                // Hash the stable fields.
                // We include the schema since it can be set by the user.
                schema.hash(state);
                output_schema.hash(state);
                with_columns.hash(state);
                python_source.hash(state);
                n_rows.hash(state);
                std::mem::discriminant(predicate).hash(state);
                match predicate {
                    PythonPredicate::None => {},
                    // A PyArrow predicate always compares as unequal, so we can hash it however we want
                    PythonPredicate::PyArrow(p) => p.has_residual.hash(state),
                    PythonPredicate::Polars(e) => expr_hash.hash_expr(e, state),
                }
                validate_schema.hash(state);
                is_pure.hash(state);
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
                expr_hash.hash_expr(predicate, state);
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
                // We don't have to traverse the schema, hive partitions etc. as they are derivative
                // from the paths.
                scan_type.hash(state);
                sources.hash(state);
                if let Some(predicate) = predicate {
                    expr_hash.hash_expr(predicate, state);
                }
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
                hash_exprs(expr, state);
                options.hash(state);
            },
            IR::Sort {
                input: _,
                by_column,
                slice,
                sort_options,
            } => {
                hash_exprs(by_column, state);
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
                hash_exprs(keys, state);
                hash_exprs(aggs, state);

                if let Some(function) = apply {
                    true.hash(state);
                    match function {
                        PlanCallback::Rust(f) => {
                            f.hash(state);
                        },
                        #[cfg(feature = "python")]
                        PlanCallback::Python(f) => {
                            f.hash(state);
                        },
                    }
                }

                apply.is_none().hash(state);
                maintain_order.hash(state);
                options.hash(state);
            },
            IR::Join {
                input_left: _,
                input_right: _,
                schema: _,
                options,
            } => {
                options.shallow_hash(state, expr_hash);
            },
            IR::Gather {
                input: _,
                idxs: _,
                null_on_oob,
            } => {
                null_on_oob.hash(state);
            },
            IR::HStack {
                input: _,
                exprs,
                schema: _,
                options,
            } => {
                hash_exprs(exprs, state);
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
                contexts: _,
                schema: _,
            } => {},
            IR::Sink { input: _, payload } => {
                payload.shallow_hash(state, expr_hash);
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
                maintain_order,
            } => {
                key.hash(state);
                maintain_order.hash(state);
            },
            IR::UnoptimizedDispatch {
                inputs: _,
                arg_map: _,
                operation,
            } => match operation {
                UnoptimizedOperation::ColumnarFunction {
                    function,
                    options,
                    output_name,
                } => {
                    function.hash(state);
                    options.hash(state);
                    output_name.hash(state);
                },

                UnoptimizedOperation::AnonymousColumnsUdf {
                    function,
                    options,
                    output_name,
                    fmt_str: _,
                    ctx_schema: _,
                } => {
                    function.hash(state);
                    options.hash(state);
                    output_name.hash(state);
                },

                UnoptimizedOperation::DynamicSlice { output_name } => {
                    output_name.hash(state);
                },
            },
            IR::Invalid => unreachable!(),
        }
    }
}
