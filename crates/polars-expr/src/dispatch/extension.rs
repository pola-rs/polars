use std::sync::Arc;

use polars_core::error::PolarsResult;
use polars_core::prelude::*;
use polars_plan::dsl::{ColumnsUdf, SpecialEq};
use polars_plan::plans::IRExtensionFunction;

pub fn function_expr_to_udf(func: IRExtensionFunction) -> SpecialEq<Arc<dyn ColumnsUdf>> {
    use IRExtensionFunction::*;
    match func {
        To(dtype) => map!(ext_to, dtype.clone()),
        Storage => map!(ext_storage),
    }
}

fn ext_to(s: &Column, dtype: DataType) -> PolarsResult<Column> {
    let DataType::Extension(typ, storage) = &dtype else {
        polars_bail!(ComputeError: "ext.to() requires an Extension dtype")
    };

    if let DataType::Extension(_, _) = s.dtype() {
        polars_bail!(
            InvalidOperation:
            "cannot call `.ext.to` on a column that is already an Extension type ({}); \
            extension-to-extension conversion is not defined — if you want to pass the underlying \
            storage into a new extension, do so explicitly with `.ext.storage().ext.to(...)`",
            s.dtype()
        )
    };

    if s.dtype() != &**storage {
        polars_bail!(
            SchemaMismatch:
            "cannot convert column of type {} to extension {} with storage {}; \
             column dtype must match the extension's storage type",
            s.dtype(), typ.name(), **storage
        )
    }

    Ok(s.apply_unary_elementwise(|s| s.clone().into_extension(typ.clone())))
}

fn ext_storage(s: &Column) -> PolarsResult<Column> {
    Ok(s.apply_unary_elementwise(|s| s.to_storage().clone()))
}
