use std::sync::Arc;

use polars_plan::prelude::*;
use pyo3::prelude::*;

use crate::conversion::Wrap;
use crate::expr::ToExprs;
use crate::prelude::DataType;
use crate::PyExpr;

#[pyfunction]
pub fn dtype_str_repr(dtype: Wrap<DataType>) -> PyResult<String> {
    let dtype = dtype.0;
    Ok(dtype.to_string())
}

#[cfg(feature = "ffi_plugin")]
#[pyfunction]
pub fn register_plugin_function(
    plugin_path: &str,
    function_name: &str,
    args: Vec<PyExpr>,
    kwargs: Vec<u8>,
    is_elementwise: bool,
    input_wildcard_expansion: bool,
    returns_scalar: bool,
    cast_to_supertype: bool,
    pass_name_to_apply: bool,
    changes_length: bool,
) -> PyResult<PyExpr> {
    let collect_groups = if is_elementwise {
        ApplyOptions::ElementWise
    } else {
        ApplyOptions::GroupWise
    };

    Ok(Expr::Function {
        input: args.to_exprs(),
        function: FunctionExpr::FfiPlugin {
            lib: Arc::from(plugin_path),
            symbol: Arc::from(function_name),
            kwargs: Arc::from(kwargs),
        },
        options: FunctionOptions {
            collect_groups,
            input_wildcard_expansion,
            returns_scalar,
            cast_to_supertypes: cast_to_supertype,
            pass_name_to_apply,
            changes_length,
            ..Default::default()
        },
    }
    .into())
}
