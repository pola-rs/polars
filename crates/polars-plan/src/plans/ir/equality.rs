use std::sync::Arc;

use polars_utils::arena::Arena;

use super::IR;
use crate::plans::{AExpr, ExprIR};
#[cfg(feature = "python")]
use crate::plans::{ArrowPredicate, PythonOptions, PythonPredicate};

pub trait ExpressionComparator {
    fn equals(&mut self, lhs: &ExprIR, rhs: &ExprIR, expr_arena: &Arena<AExpr>) -> bool;
}

impl IR {
    fn expr_iter_eq<'a, T>(
        lhs: T,
        rhs: T,
        expr_arena: &Arena<AExpr>,
        cmp: &mut impl ExpressionComparator,
    ) -> bool
    where
        T: IntoIterator<Item = &'a ExprIR>,
        T::IntoIter: ExactSizeIterator,
    {
        let lhs = lhs.into_iter();
        let rhs = rhs.into_iter();
        lhs.len() == rhs.len() && lhs.zip(rhs).all(|(l, r)| cmp.equals(l, r, expr_arena))
    }

    /// Compares two IR nodes at the top level, applying a custom comparator to compare child expressions
    pub fn is_ir_equal_shallow(
        &self,
        other: &Self,
        expr_arena: &Arena<AExpr>,
        expression_cmp: &mut impl ExpressionComparator,
    ) -> bool {
        if std::mem::discriminant(self) != std::mem::discriminant(other) {
            return false;
        }

        macro_rules! expr_eq {
            ($lhs:expr, $rhs:expr) => {
                expression_cmp.equals($lhs, $rhs, expr_arena)
            };
        }

        macro_rules! expr_iter_eq {
            ($lhs:expr, $rhs:expr) => {
                Self::expr_iter_eq($lhs, $rhs, expr_arena, expression_cmp)
            };
        }

        match self {
            #[cfg(feature = "python")]
            IR::PythonScan { options: l_options } => {
                let IR::PythonScan { options: r_options } = other else {
                    return false;
                };
                let PythonOptions {
                    scan_fn: l_scan_fn,
                    schema: l_schema,
                    output_schema: l_output_schema,
                    with_columns: l_with_columns,
                    python_source: l_python_source,
                    n_rows: l_n_rows,
                    predicate: l_predicate,
                    validate_schema: l_validate_schema,
                    is_pure: l_is_pure,
                } = l_options;
                let PythonOptions {
                    scan_fn: r_scan_fn,
                    schema: r_schema,
                    output_schema: r_output_schema,
                    with_columns: r_with_columns,
                    python_source: r_python_source,
                    n_rows: r_n_rows,
                    predicate: r_predicate,
                    validate_schema: r_validate_schema,
                    is_pure: r_is_pure,
                } = r_options;

                let scan_fn_eq = (l_scan_fn.is_some() == r_scan_fn.is_some())
                    && l_scan_fn
                        .as_ref()
                        .map(|l| l.0.as_ptr() == r_scan_fn.as_ref().unwrap().0.as_ptr())
                        .unwrap_or(true);

                use PythonPredicate as PP;
                let predicate_eq = (std::mem::discriminant(l_predicate)
                    == std::mem::discriminant(r_predicate))
                    && match l_predicate {
                        PP::PyArrow(ArrowPredicate {
                            predicate: l_predicate,
                            pyarrow_predicate: _,
                            has_residual: _,
                        }) => {
                            let PP::PyArrow(r_inner) = r_predicate else {
                                return false;
                            };
                            expr_eq!(l_predicate, &r_inner.predicate)
                        },
                        PP::Polars(l_expr) => {
                            let PP::Polars(r_expr) = r_predicate else {
                                return false;
                            };
                            expr_eq!(l_expr, r_expr)
                        },
                        PP::None => true,
                    };

                (*l_is_pure && *r_is_pure)
                    && scan_fn_eq
                    && l_schema == r_schema
                    && l_output_schema == r_output_schema
                    && l_with_columns == r_with_columns
                    && l_python_source == r_python_source
                    && l_n_rows == r_n_rows
                    && predicate_eq
                    && l_validate_schema == r_validate_schema
            },
            IR::Slice {
                offset: l_offset,
                len: l_len,
                input: _,
            } => {
                let IR::Slice {
                    offset: r_offset,
                    len: r_len,
                    input: _,
                } = other
                else {
                    return false;
                };
                l_len == r_len && l_offset == r_offset
            },
            IR::Filter {
                input: _,
                predicate: l_predicate,
            } => {
                let IR::Filter {
                    input: _,
                    predicate: r_predicate,
                } = other
                else {
                    return false;
                };
                expr_eq!(l_predicate, r_predicate)
            },
            IR::Scan {
                sources: l_sources,
                file_info: _,
                hive_parts: _,
                predicate: l_predicate,
                predicate_file_skip_applied: _,
                output_schema: _,
                scan_type: l_scan_type,
                unified_scan_args: l_unified_scan_args,
            } => {
                let IR::Scan {
                    sources: r_sources,
                    file_info: _,
                    hive_parts: _,
                    predicate: r_predicate,
                    predicate_file_skip_applied: _,
                    output_schema: _,
                    scan_type: r_scan_type,
                    unified_scan_args: r_unified_scan_args,
                } = other
                else {
                    return false;
                };
                l_sources == r_sources
                    && expr_iter_eq!(l_predicate, r_predicate)
                    && l_scan_type == r_scan_type
                    && l_unified_scan_args == r_unified_scan_args
            },
            IR::DataFrameScan {
                df: l_df,
                schema: _,
                output_schema: l_output_schema,
            } => {
                let IR::DataFrameScan {
                    df: r_df,
                    schema: _,
                    output_schema: r_output_schema,
                } = other
                else {
                    return false;
                };
                Arc::ptr_eq(l_df, r_df) && l_output_schema == r_output_schema
            },
            IR::SimpleProjection {
                columns: l_columns,
                input: _,
            } => {
                let IR::SimpleProjection {
                    columns: r_columns,
                    input: _,
                } = other
                else {
                    return false;
                };
                l_columns == r_columns
            },
            IR::Select {
                input: _,
                expr: l_expr,
                schema: _,
                options: l_options,
            } => {
                let IR::Select {
                    input: _,
                    expr: r_expr,
                    schema: _,
                    options: r_options,
                } = other
                else {
                    return false;
                };
                expr_iter_eq!(l_expr, r_expr) && l_options == r_options
            },
            IR::Sort {
                input: _,
                by_column: l_by_column,
                slice: l_slice,
                sort_options: l_sort_options,
            } => {
                let IR::Sort {
                    input: _,
                    by_column: r_by_column,
                    slice: r_slice,
                    sort_options: r_sort_options,
                } = other
                else {
                    return false;
                };
                expr_iter_eq!(l_by_column, r_by_column)
                    && l_slice == r_slice
                    && l_sort_options == r_sort_options
            },
            IR::GroupBy {
                input: _,
                keys: l_keys,
                aggs: l_aggs,
                schema: _,
                apply: l_apply,
                maintain_order: l_maintain_order,
                options: l_options,
            } => {
                let IR::GroupBy {
                    input: _,
                    keys: r_keys,
                    aggs: r_aggs,
                    schema: _,
                    apply: r_apply,
                    maintain_order: r_maintain_order,
                    options: r_options,
                } = other
                else {
                    return false;
                };
                expr_iter_eq!(l_keys, r_keys)
                    && expr_iter_eq!(l_aggs, r_aggs)
                    && l_apply == r_apply
                    && l_maintain_order == r_maintain_order
                    && l_options == r_options
            },
            IR::Join {
                input_left: _,
                input_right: _,
                schema: _,
                left_on: l_left_on,
                right_on: l_right_on,
                options: l_options,
            } => {
                let IR::Join {
                    input_left: _,
                    input_right: _,
                    schema: _,
                    left_on: r_left_on,
                    right_on: r_right_on,
                    options: r_options,
                } = other
                else {
                    return false;
                };
                expr_iter_eq!(l_left_on, r_left_on)
                    && expr_iter_eq!(l_right_on, r_right_on)
                    && l_options == r_options
            },
            IR::Gather {
                input: _,
                idxs: _,
                null_on_oob: l_null_on_oob,
            } => {
                let IR::Gather {
                    input: _,
                    idxs: _,
                    null_on_oob: r_null_on_oob,
                } = other
                else {
                    return false;
                };
                l_null_on_oob == r_null_on_oob
            },
            IR::HStack {
                input: _,
                exprs: l_exprs,
                schema: _,
                options: l_options,
            } => {
                let IR::HStack {
                    input: _,
                    exprs: r_exprs,
                    schema: _,
                    options: r_options,
                } = other
                else {
                    return false;
                };
                expr_iter_eq!(l_exprs, r_exprs) && l_options == r_options
            },
            IR::Distinct {
                input: _,
                options: l_options,
            } => {
                let IR::Distinct {
                    input: _,
                    options: r_options,
                } = other
                else {
                    return false;
                };
                l_options == r_options
            },
            IR::MapFunction {
                input: _,
                function: l_function,
            } => {
                let IR::MapFunction {
                    input: _,
                    function: r_function,
                } = other
                else {
                    return false;
                };
                l_function == r_function
            },
            IR::Union {
                inputs: _,
                options: l_options,
            } => {
                let IR::Union {
                    inputs: _,
                    options: r_options,
                } = other
                else {
                    return false;
                };
                l_options == r_options
            },
            IR::HConcat {
                inputs: _,
                schema: _,
                options: l_options,
            } => {
                let IR::HConcat {
                    inputs: _,
                    schema: _,
                    options: r_options,
                } = other
                else {
                    return false;
                };
                l_options == r_options
            },
            IR::ExtContext {
                input: _,
                contexts: _,
                schema: _,
            } => {
                // `input` and `contexts` are both traversal inputs (see `IR::inputs`), so they
                // are compared via child ids. `schema` is derivative. Nothing left to compare.
                true
            },
            IR::Sink {
                input: _,
                payload: l_payload,
            } => {
                let IR::Sink {
                    input: _,
                    payload: r_payload,
                } = other
                else {
                    return false;
                };
                // Note that SinkTypeIR -> PartitionedSinkOptionsIR -> PartitionStrategyIR
                // contains a Vec<ExprIR> that we do not compare deeply. In this case we might not
                // detect that two `Sink` nodes are actually equivalent.
                l_payload == r_payload
            },
            IR::SinkMultiple { inputs: _ } => {
                // `inputs` are traversal inputs, compared via child ids. Nothing else here.
                true
            },
            IR::Cache { input: _, id: l_id } => {
                let IR::Cache { input: _, id: r_id } = other else {
                    return false;
                };
                l_id == r_id
            },
            #[cfg(feature = "merge_sorted")]
            IR::MergeSorted {
                input_left: _,
                input_right: _,
                key: l_key,
                maintain_order: l_maintain_order,
            } => {
                let IR::MergeSorted {
                    input_left: _,
                    input_right: _,
                    key: r_key,
                    maintain_order: r_maintain_order,
                } = other
                else {
                    return false;
                };
                l_key == r_key && l_maintain_order == r_maintain_order
            },
            IR::UnoptimizedDispatch {
                inputs: _,
                arg_map: _,
                operation: _,
            } => {
                todo!("Implement PartialEq for UnoptimizedOperation and ArgMap");
            },
            IR::Invalid => unreachable!("cannot compare `IR::Invalid`"),
        }
    }
}
