#[macro_export]
macro_rules! polars_plugin_expr_info {
    (
        $name:literal, $data:expr, $data_ty:ty
    ) => {{
        #[unsafe(export_name = concat!("_PL_PLUGIN_V2::", $name))]
        pub static VTABLE: $crate::export::polars_ffi::version_1::PluginSymbol =
            $crate::export::polars_ffi::version_1::VTable::new::<$data_ty>().into_symbol();

        let data = ::std::boxed::Box::new($data);
        let data = ::std::boxed::Box::into_raw(data);
        $crate::v1::PolarsPluginExprInfo::_new($name, data as *const u8)
    }};
}

pub struct PolarsPluginExprInfo {
    symbol: &'static str,
    data_ptr: *const u8,
}

impl PolarsPluginExprInfo {
    #[doc(hidden)]
    pub fn _new(symbol: &'static str, data_ptr: *const u8) -> Self {
        Self { symbol, data_ptr }
    }
}

impl<'py> pyo3::IntoPyObject<'py> for PolarsPluginExprInfo {
    type Target = pyo3::types::PyTuple;
    type Output = pyo3::Bound<'py, Self::Target>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: pyo3::Python<'py>) -> Result<Self::Output, Self::Error> {
        use pyo3::IntoPyObjectExt;
        pyo3::types::PyTuple::new(
            py,
            [
                self.symbol.into_py_any(py)?,
                (self.data_ptr as usize).into_py_any(py)?,
            ],
        )
    }
}
