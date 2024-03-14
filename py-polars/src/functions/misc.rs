use std::sync::Arc;

use pyo3::prelude::*;

use crate::conversion::Wrap;
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
    lib: &str,
    symbol: &str,
    args: Vec<PyExpr>,
    kwargs: Vec<u8>,
    is_elementwise: bool,
    input_wildcard_expansion: bool,
    returns_scalar: bool,
    cast_to_supertypes: bool,
    pass_name_to_apply: bool,
    changes_length: bool,
) -> PyResult<PyExpr> {
    use polars_plan::prelude::*;

    let collect_groups = if is_elementwise {
        ApplyOptions::ElementWise
    } else {
        ApplyOptions::GroupWise
    };
    let mut input = Vec::with_capacity(args.len());
    for a in args {
        input.push(a.inner)
    }

    Ok(Expr::Function {
        input,
        function: FunctionExpr::FfiPlugin {
            lib: Arc::from(lib),
            symbol: Arc::from(symbol),
            kwargs: Arc::from(kwargs),
        },
        options: FunctionOptions {
            collect_groups,
            input_wildcard_expansion,
            returns_scalar,
            cast_to_supertypes,
            pass_name_to_apply,
            changes_length,
            ..Default::default()
        },
    }
    .into())
}
