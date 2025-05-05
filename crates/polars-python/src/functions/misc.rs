use polars_plan::prelude::*;
use pyo3::prelude::*;

use crate::PyExpr;
use crate::conversion::Wrap;
use crate::expr::ToExprs;
use crate::prelude::DataType;

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
    let cast_to_supertypes = if cast_to_supertype {
        Some(CastingRules::cast_to_supertypes())
    } else {
        None
    };

    let mut flags = FunctionFlags::default();
    if is_elementwise {
        flags.set_elementwise();
    }
    flags.set(FunctionFlags::LENGTH_PRESERVING, !changes_length);
    flags.set(FunctionFlags::PASS_NAME_TO_APPLY, pass_name_to_apply);
    flags.set(FunctionFlags::RETURNS_SCALAR, returns_scalar);
    flags.set(
        FunctionFlags::INPUT_WILDCARD_EXPANSION,
        input_wildcard_expansion,
    );

    let options = FunctionOptions {
        cast_options: cast_to_supertypes,
        flags,
        ..Default::default()
    };

    Ok(Expr::Function {
        input: args.to_exprs(),
        function: FunctionExpr::FfiPlugin {
            flags: options,
            lib: plugin_path.into(),
            symbol: function_name.into(),
            kwargs: kwargs.into(),
        },
        options,
    }
    .into())
}

#[pyfunction]
pub fn __register_startup_deps() {
    #[cfg(feature = "object")]
    unsafe {
        crate::on_startup::register_startup_deps(true)
    }
}
