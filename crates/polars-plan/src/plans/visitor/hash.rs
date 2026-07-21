use std::hash::{Hash, Hasher};
use std::sync::Arc;

use polars_utils::arena::{Arena, Node};

use super::*;
#[cfg(feature = "python")]
use crate::plans::PythonOptions;
use crate::plans::{AExpr, FunctionIR, IR, UnoptimizedOperation};
use crate::prelude::aexpr::traverse_and_hash_aexpr;
use crate::prelude::{ExprIR, PlanCallback};

impl IRNode {
    pub(crate) fn hashable_and_cmp<'a>(
        &'a self,
        lp_arena: &'a Arena<IR>,
        expr_arena: &'a Arena<AExpr>,
    ) -> IRHashWrap<'a> {
        IRHashWrap {
            node: self.node(),
            lp_arena,
            expr_arena,
            hash_as_equality: false,
        }
    }
}

pub(crate) struct IRHashWrap<'a> {
    node: Node,
    lp_arena: &'a Arena<IR>,
    expr_arena: &'a Arena<AExpr>,
    hash_as_equality: bool,
}

impl<'a> IRHashWrap<'a> {
    pub(crate) fn new(
        node: Node,
        lp_arena: &'a Arena<IR>,
        expr_arena: &'a Arena<AExpr>,
        hash_as_equality: bool,
    ) -> Self {
        Self {
            node,
            lp_arena,
            expr_arena,
            hash_as_equality,
        }
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

fn expr_ir_eq(left: &ExprIR, right: &ExprIR, expr_arena: &Arena<AExpr>) -> bool {
    left.get_alias() == right.get_alias()
        && AexprNode::new(left.node()).hashable_and_cmp(expr_arena)
            == AexprNode::new(right.node()).hashable_and_cmp(expr_arena)
}

fn expr_irs_eq(left: &[ExprIR], right: &[ExprIR], expr_arena: &Arena<AExpr>) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| expr_ir_eq(left, right, expr_arena))
}

fn opt_expr_ir_eq(
    left: &Option<ExprIR>,
    right: &Option<ExprIR>,
    expr_arena: &Arena<AExpr>,
) -> bool {
    match (left, right) {
        (Some(left), Some(right)) => expr_ir_eq(left, right, expr_arena),
        (None, None) => true,
        _ => false,
    }
}

fn function_ir_eq(left: &FunctionIR, right: &FunctionIR) -> bool {
    use FunctionIR::*;

    match (left, right) {
        (
            RowIndex {
                name: left_name,
                offset: left_offset,
                ..
            },
            RowIndex {
                name: right_name,
                offset: right_offset,
                ..
            },
        ) => left_name == right_name && left_offset == right_offset,
        #[cfg(feature = "python")]
        (OpaquePython(left), OpaquePython(right)) => {
            left.function.0.as_ptr() == right.function.0.as_ptr()
                && left.schema == right.schema
                && left.predicate_pd == right.predicate_pd
                && left.projection_pd == right.projection_pd
                && left.streamable == right.streamable
                && left.validate_output == right.validate_output
        },
        (
            FastCount {
                sources: left_sources,
                scan_type: left_scan_type,
                alias: left_alias,
                cloud_options: left_cloud_options,
            },
            FastCount {
                sources: right_sources,
                scan_type: right_scan_type,
                alias: right_alias,
                cloud_options: right_cloud_options,
            },
        ) => {
            left_sources == right_sources
                && left_scan_type == right_scan_type
                && left_alias == right_alias
                && left_cloud_options == right_cloud_options
        },
        (
            Unnest {
                columns: left_columns,
                separator: left_separator,
            },
            Unnest {
                columns: right_columns,
                separator: right_separator,
            },
        ) => left_columns == right_columns && left_separator == right_separator,
        (Rechunk, Rechunk) => true,
        (
            Explode {
                columns: left_columns,
                options: left_options,
                ..
            },
            Explode {
                columns: right_columns,
                options: right_options,
                ..
            },
        ) => left_columns == right_columns && left_options == right_options,
        #[cfg(feature = "pivot")]
        (Unpivot { args: left, .. }, Unpivot { args: right, .. }) => left == right,
        (
            Opaque {
                function: left_function,
                schema: left_schema,
                predicate_pd: left_predicate_pd,
                projection_pd: left_projection_pd,
                streamable: left_streamable,
                fmt_str: left_fmt_str,
            },
            Opaque {
                function: right_function,
                schema: right_schema,
                predicate_pd: right_predicate_pd,
                projection_pd: right_projection_pd,
                streamable: right_streamable,
                fmt_str: right_fmt_str,
            },
        ) => {
            Arc::ptr_eq(left_function, right_function)
                && match (left_schema, right_schema) {
                    (Some(left), Some(right)) => Arc::ptr_eq(left, right),
                    (None, None) => true,
                    _ => false,
                }
                && left_predicate_pd == right_predicate_pd
                && left_projection_pd == right_projection_pd
                && left_streamable == right_streamable
                && left_fmt_str == right_fmt_str
        },
        (Hint(left), Hint(right)) => match (left, right) {
            (
                crate::plans::functions::HintIR::Sorted(left),
                crate::plans::functions::HintIR::Sorted(right),
            ) => left == right,
        },
        _ => false,
    }
}

fn plan_callback_eq<Args, Out>(
    left: &Option<PlanCallback<Args, Out>>,
    right: &Option<PlanCallback<Args, Out>>,
) -> bool {
    match (left, right) {
        (None, None) => true,
        (Some(PlanCallback::Rust(left)), Some(PlanCallback::Rust(right))) => left == right,
        #[cfg(feature = "python")]
        (Some(PlanCallback::Python(left)), Some(PlanCallback::Python(right))) => left == right,
        _ => false,
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
        PythonPredicate::PyArrow(p) => {
            format!("{:?}", p).hash(state);
            p.has_residual.hash(state);
        },
        PythonPredicate::Polars(e) => e.traverse_and_hash(expr_arena, state),
    }
}

impl Hash for IRHashWrap<'_> {
    // This hashes the variant, not the whole plan
    fn hash<H: Hasher>(&self, state: &mut H) {
        let alp = self.lp_arena.get(self.node);
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
                left_on,
                right_on,
                options,
            } => {
                hash_exprs(left_on, self.expr_arena, state);
                hash_exprs(right_on, self.expr_arena, state);
                options.hash(state);
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

impl PartialEq for IRHashWrap<'_> {
    /// Compare one plan node while deliberately ignoring its input node indices.
    ///
    /// CSPE compares the inputs by their bottom-up equivalence-class identifiers. This
    /// comparison validates the remaining semantic attributes after the strong hash has
    /// selected a small set of candidates.
    fn eq(&self, other: &Self) -> bool {
        let left = self.lp_arena.get(self.node);
        let right = other.lp_arena.get(other.node);

        match (left, right) {
            #[cfg(feature = "python")]
            (
                IR::PythonScan {
                    options: left_options,
                },
                IR::PythonScan {
                    options: right_options,
                },
            ) => {
                use crate::prelude::PythonPredicate;

                if !left_options.is_pure || !right_options.is_pure {
                    return false;
                }

                let predicates_equal = match (&left_options.predicate, &right_options.predicate) {
                    (PythonPredicate::None, PythonPredicate::None) => true,
                    (PythonPredicate::PyArrow(left), PythonPredicate::PyArrow(right)) => {
                        left.has_residual == right.has_residual
                            && left.pyarrow_predicate.0.as_ptr()
                                == right.pyarrow_predicate.0.as_ptr()
                            && expr_ir_eq(&left.predicate, &right.predicate, self.expr_arena)
                    },
                    (PythonPredicate::Polars(left), PythonPredicate::Polars(right)) => {
                        expr_ir_eq(left, right, self.expr_arena)
                    },
                    _ => false,
                };

                (match (&left_options.scan_fn, &right_options.scan_fn) {
                    (Some(left), Some(right)) => left.0.as_ptr() == right.0.as_ptr(),
                    _ => false,
                }) && left_options.schema == right_options.schema
                    && left_options.output_schema == right_options.output_schema
                    && left_options.with_columns == right_options.with_columns
                    && left_options.python_source == right_options.python_source
                    && left_options.n_rows == right_options.n_rows
                    && predicates_equal
                    && left_options.validate_schema == right_options.validate_schema
            },
            (
                IR::Slice {
                    offset: left_offset,
                    len: left_len,
                    ..
                },
                IR::Slice {
                    offset: right_offset,
                    len: right_len,
                    ..
                },
            ) => left_offset == right_offset && left_len == right_len,
            (
                IR::Filter {
                    predicate: left, ..
                },
                IR::Filter {
                    predicate: right, ..
                },
            ) => expr_ir_eq(left, right, self.expr_arena),
            (
                IR::Scan {
                    sources: left_sources,
                    predicate: left_predicate,
                    scan_type: left_scan_type,
                    unified_scan_args: left_args,
                    ..
                },
                IR::Scan {
                    sources: right_sources,
                    predicate: right_predicate,
                    scan_type: right_scan_type,
                    unified_scan_args: right_args,
                    ..
                },
            ) => {
                left_sources == right_sources
                    && left_scan_type == right_scan_type
                    && left_args == right_args
                    && opt_expr_ir_eq(left_predicate, right_predicate, self.expr_arena)
            },
            (
                IR::DataFrameScan {
                    df: left_df,
                    output_schema: left_schema,
                    ..
                },
                IR::DataFrameScan {
                    df: right_df,
                    output_schema: right_schema,
                    ..
                },
            ) => Arc::ptr_eq(left_df, right_df) && left_schema == right_schema,
            (
                IR::SimpleProjection { columns: left, .. },
                IR::SimpleProjection { columns: right, .. },
            ) => left == right,
            (
                IR::Select {
                    expr: left_expr,
                    options: left_options,
                    ..
                },
                IR::Select {
                    expr: right_expr,
                    options: right_options,
                    ..
                },
            ) => {
                left_options == right_options && expr_irs_eq(left_expr, right_expr, self.expr_arena)
            },
            (
                IR::Sort {
                    by_column: left_by,
                    slice: left_slice,
                    sort_options: left_options,
                    ..
                },
                IR::Sort {
                    by_column: right_by,
                    slice: right_slice,
                    sort_options: right_options,
                    ..
                },
            ) => {
                left_slice == right_slice
                    && left_options == right_options
                    && expr_irs_eq(left_by, right_by, self.expr_arena)
            },
            (IR::Cache { id: left, .. }, IR::Cache { id: right, .. }) => left == right,
            (
                IR::GroupBy {
                    keys: left_keys,
                    aggs: left_aggs,
                    apply: left_apply,
                    maintain_order: left_maintain_order,
                    options: left_options,
                    ..
                },
                IR::GroupBy {
                    keys: right_keys,
                    aggs: right_aggs,
                    apply: right_apply,
                    maintain_order: right_maintain_order,
                    options: right_options,
                    ..
                },
            ) => {
                plan_callback_eq(left_apply, right_apply)
                    && left_maintain_order == right_maintain_order
                    && left_options == right_options
                    && expr_irs_eq(left_keys, right_keys, self.expr_arena)
                    && expr_irs_eq(left_aggs, right_aggs, self.expr_arena)
            },
            (
                IR::Join {
                    left_on: left_left_on,
                    right_on: left_right_on,
                    options: left_options,
                    ..
                },
                IR::Join {
                    left_on: right_left_on,
                    right_on: right_right_on,
                    options: right_options,
                    ..
                },
            ) => {
                left_options == right_options
                    && expr_irs_eq(left_left_on, right_left_on, self.expr_arena)
                    && expr_irs_eq(left_right_on, right_right_on, self.expr_arena)
            },
            (
                IR::Gather {
                    null_on_oob: left, ..
                },
                IR::Gather {
                    null_on_oob: right, ..
                },
            ) => left == right,
            (
                IR::HStack {
                    exprs: left_exprs,
                    options: left_options,
                    ..
                },
                IR::HStack {
                    exprs: right_exprs,
                    options: right_options,
                    ..
                },
            ) => {
                left_options == right_options
                    && expr_irs_eq(left_exprs, right_exprs, self.expr_arena)
            },
            (IR::Distinct { options: left, .. }, IR::Distinct { options: right, .. }) => {
                left == right
            },
            (
                IR::MapFunction { function: left, .. },
                IR::MapFunction {
                    function: right, ..
                },
            ) => function_ir_eq(left, right),
            (IR::Union { options: left, .. }, IR::Union { options: right, .. }) => left == right,
            (IR::HConcat { options: left, .. }, IR::HConcat { options: right, .. }) => {
                left == right
            },
            (IR::ExtContext { .. }, IR::ExtContext { .. }) => true,
            // Sink nodes are execution boundaries, and unoptimized dispatch nodes are not meant
            // to be optimized across. Keeping them unique is both cheap and conservative.
            (IR::Sink { .. }, IR::Sink { .. })
            | (IR::UnoptimizedDispatch { .. }, IR::UnoptimizedDispatch { .. }) => false,
            (IR::SinkMultiple { .. }, IR::SinkMultiple { .. }) => true,
            #[cfg(feature = "merge_sorted")]
            (
                IR::MergeSorted {
                    key: left_key,
                    maintain_order: left_maintain_order,
                    ..
                },
                IR::MergeSorted {
                    key: right_key,
                    maintain_order: right_maintain_order,
                    ..
                },
            ) => left_key == right_key && left_maintain_order == right_maintain_order,
            (IR::Invalid, IR::Invalid) => unreachable!(),
            _ => false,
        }
    }
}

impl Eq for IRHashWrap<'_> {}
