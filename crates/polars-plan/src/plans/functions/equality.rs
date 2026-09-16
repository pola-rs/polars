use std::sync::Arc;

#[cfg(feature = "python")]
use crate::plans::OpaquePythonUdf;
use crate::plans::{FunctionIR, UdfSchema};

fn opt_udf_schema_ptr_eq(a: &Option<Arc<dyn UdfSchema>>, b: &Option<Arc<dyn UdfSchema>>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => Arc::ptr_eq(a, b),
        _ => false,
    }
}

impl PartialEq for FunctionIR {
    // Mirrors `impl Hash for FunctionIR`
    fn eq(&self, other: &Self) -> bool {
        if std::mem::discriminant(self) != std::mem::discriminant(other) {
            return false;
        }

        use FunctionIR as F;

        match self {
            F::RowIndex {
                name: l_name,
                offset: l_offset,
                schema: _,
            } => {
                let F::RowIndex {
                    name: r_name,
                    offset: r_offset,
                    schema: _,
                } = other
                else {
                    return false;
                };
                l_name == r_name && l_offset == r_offset
            },
            #[cfg(feature = "python")]
            F::OpaquePython(OpaquePythonUdf {
                function: l_function,
                schema: l_schema,
                predicate_pd: l_predicate_pd,
                projection_pd: l_projection_pd,
                streamable: l_streamable,
                validate_output: l_validate_output,
            }) => {
                let F::OpaquePython(OpaquePythonUdf {
                    function: r_function,
                    schema: r_schema,
                    predicate_pd: r_predicate_pd,
                    projection_pd: r_projection_pd,
                    streamable: r_streamable,
                    validate_output: r_validate_output,
                }) = other
                else {
                    return false;
                };

                l_function.0.as_ptr() == r_function.0.as_ptr()
                    && l_schema == r_schema
                    && l_predicate_pd == r_predicate_pd
                    && l_projection_pd == r_projection_pd
                    && l_streamable == r_streamable
                    && l_validate_output == r_validate_output
            },
            F::Opaque {
                function: l_function,
                schema: l_schema,
                predicate_pd: l_predicate_pd,
                projection_pd: l_projection_pd,
                streamable: l_streamable,
                fmt_str: _,
            } => {
                let F::Opaque {
                    function: r_function,
                    schema: r_schema,
                    predicate_pd: r_predicate_pd,
                    projection_pd: r_projection_pd,
                    streamable: r_streamable,
                    fmt_str: _,
                } = other
                else {
                    return false;
                };
                Arc::ptr_eq(l_function, r_function)
                    && opt_udf_schema_ptr_eq(l_schema, r_schema)
                    && l_predicate_pd == r_predicate_pd
                    && l_projection_pd == r_projection_pd
                    && l_streamable == r_streamable
            },
            F::FastCount {
                sources: l_sources,
                scan_type: l_scan_type,
                alias: l_alias,
                cloud_options: _,
            } => {
                let F::FastCount {
                    sources: r_sources,
                    scan_type: r_scan_type,
                    alias: r_alias,
                    cloud_options: _,
                } = other
                else {
                    return false;
                };
                l_sources == r_sources && l_scan_type == r_scan_type && l_alias == r_alias
            },
            F::Unnest {
                columns: l_columns,
                separator: l_separator,
            } => {
                let F::Unnest {
                    columns: r_columns,
                    separator: r_separator,
                } = other
                else {
                    return false;
                };
                l_columns == r_columns && l_separator == r_separator
            },
            F::Rechunk => true,
            F::Explode {
                columns: l_columns,
                options: l_options,
                schema: _,
            } => {
                let F::Explode {
                    columns: r_columns,
                    options: r_options,
                    schema: _,
                } = other
                else {
                    return false;
                };
                l_columns == r_columns && l_options == r_options
            },
            #[cfg(feature = "pivot")]
            F::Unpivot {
                args: l_args,
                schema: _,
            } => {
                let F::Unpivot {
                    args: r_args,
                    schema: _,
                } = other
                else {
                    return false;
                };
                l_args == r_args
            },
            F::Hint(l_hint) => {
                let F::Hint(r_hint) = other else {
                    return false;
                };
                l_hint == r_hint
            },
        }
    }
}
