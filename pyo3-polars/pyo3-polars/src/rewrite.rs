//! Plugin rewrites: decide during query planning what an expression becomes.
//!
//! A function annotated with [`polars_rewrite`] receives the resolved input fields and returns an
//! [`Expr`]. In that expression, [`rewrite_input(i)`](rewrite_input) refers to the `i`-th argument
//! of the call. Register it from Python with `polars.plugins.register_plugin_rewrite`.
//!
//! Polars does not check the data type of the returned expression. Test it in the plugin's own
//! test suite, and end the expression with `.ext().to(..)` or `.cast(..)` to pin the type.
use std::borrow::Cow;

use polars_arrow::ffi::{import_field_from_c, ArrowSchema};
use polars_core::prelude::{DataType, Field, PolarsResult};
use polars_ffi::dsl_rewrite::BytesExport;
pub use polars_plan::dsl::functions::*;
pub use polars_plan::dsl::*;
pub use polars_plan::prelude::{FunctionFlags, FunctionOptions};
pub use pyo3_polars_derive::polars_rewrite;
use serde::Serialize;

use crate::derive::_update_last_error;

/// Extra information passed to a rewrite.
#[derive(Clone, Copy, Debug)]
pub struct RewriteContext<'a> {
    lib: &'a str,
}

impl<'a> RewriteContext<'a> {
    #[doc(hidden)]
    pub fn _new(lib: &'a str) -> Self {
        Self { lib }
    }

    /// Path of the plugin library, as known by Polars.
    pub fn plugin_path(&self) -> &'a str {
        self.lib
    }

    /// Call the `#[polars_expr]` function `symbol` of this plugin.
    ///
    /// `options` has the same role as the flags of `register_plugin_function`, e.g.
    /// [`FunctionOptions::elementwise`] or [`FunctionOptions::aggregation`].
    pub fn plugin_function(&self, symbol: &str, args: Vec<Expr>, options: FunctionOptions) -> Expr {
        self.plugin_function_impl(symbol, args, options, Vec::new())
    }

    /// Like [`Self::plugin_function`], passing `kwargs` to the function.
    pub fn plugin_function_with_kwargs<K: Serialize>(
        &self,
        symbol: &str,
        args: Vec<Expr>,
        options: FunctionOptions,
        kwargs: &K,
    ) -> PolarsResult<Expr> {
        let kwargs = serde_pickle::to_vec(kwargs, Default::default())
            .map_err(polars_core::error::to_compute_err)?;
        Ok(self.plugin_function_impl(symbol, args, options, kwargs))
    }

    fn plugin_function_impl(
        &self,
        symbol: &str,
        args: Vec<Expr>,
        options: FunctionOptions,
        kwargs: Vec<u8>,
    ) -> Expr {
        Expr::Function {
            input: args,
            function: FunctionExpr::FfiPlugin {
                flags: options,
                lib: self.lib.into(),
                symbol: symbol.into(),
                kwargs: kwargs.into(),
            },
        }
    }
}

/// Extension type information of a [`Field`].
pub trait FieldExtension {
    /// Name of the extension type, if the field has one.
    fn extension_name(&self) -> Option<Cow<'_, str>>;

    /// Metadata of the extension type, if the field has an extension type with metadata.
    fn extension_metadata(&self) -> Option<Cow<'_, str>>;
}

impl FieldExtension for Field {
    fn extension_name(&self) -> Option<Cow<'_, str>> {
        match self.dtype() {
            DataType::Extension(ext, _) => Some(ext.name()),
            _ => None,
        }
    }

    fn extension_metadata(&self) -> Option<Cow<'_, str>> {
        match self.dtype() {
            DataType::Extension(ext, _) => ext.serialize_metadata(),
            _ => None,
        }
    }
}

/// # Safety
/// `fields` must point to `n_fields` valid `ArrowSchema`s.
#[doc(hidden)]
pub unsafe fn _import_fields(fields: *const ArrowSchema, n_fields: usize) -> Vec<Field> {
    std::slice::from_raw_parts(fields, n_fields)
        .iter()
        .map(|f| Field::from(&import_field_from_c(f).unwrap()))
        .collect()
}

/// # Safety
/// `lib` must point to `lib_len` bytes of utf8.
#[doc(hidden)]
pub unsafe fn _import_lib<'a>(lib: *const u8, lib_len: usize) -> &'a str {
    std::str::from_utf8(std::slice::from_raw_parts(lib, lib_len)).unwrap()
}

/// # Safety
/// `out` must be a valid pointer.
#[doc(hidden)]
pub unsafe fn _export_rewrite(result: PolarsResult<Expr>, out: *mut BytesExport) {
    let result = result.and_then(|expr| {
        let mut buf = Vec::new();
        expr.serialize_versioned(&mut buf)?;
        Ok(buf)
    });
    match result {
        Ok(buf) => *out = BytesExport::from(buf),
        // Set latest error, but leave return value in empty state.
        Err(err) => _update_last_error(err),
    }
}
