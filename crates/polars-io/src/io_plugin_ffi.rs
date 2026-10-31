//! Host side of the generic I/O plugin contracts of `polars_io_ext_ffi`:
//! `polars.io_plugin.file_reader.v1` (see [`crate::external_reader::ffi`]) and
//! `polars.io_plugin.lazyframe_resolver.v1` (the `Ffi` DSL resolver of `polars-plan`).
//!
//! Plugins are found by Python package name through `polars_io_ext_ffi::registry`, whose locator
//! the Python bindings install.
use polars_arrow::datatypes::ArrowDataType;
use polars_arrow::ffi;
use polars_core::prelude::*;
use polars_error::{PolarsError, PolarsResult, polars_bail, polars_ensure, polars_err};
pub use polars_io_ext_ffi::common::{ArrowSchema, FfiError, FfiErrorKind};
use polars_io_ext_ffi::{file_reader_v1, lazyframe_resolver_v1, registry};

/// The `file_reader.v1` contract of `package`.
pub fn file_reader_v1(package: &str) -> PolarsResult<&'static file_reader_v1::Plugin> {
    // SAFETY: the plugin ID fixes the struct type.
    unsafe { registry::lookup(package, file_reader_v1::ID) }
        .map_err(|e| polars_err!(ComputeError: "{e}"))
}

/// The `lazyframe_resolver.v1` contract of `package`.
pub fn lazyframe_resolver_v1(
    package: &str,
) -> PolarsResult<&'static lazyframe_resolver_v1::Plugin> {
    // SAFETY: the plugin ID fixes the struct type.
    unsafe { registry::lookup(package, lazyframe_resolver_v1::ID) }
        .map_err(|e| polars_err!(ComputeError: "{e}"))
}

/// A plugin error as a Polars error of the same kind. Plugin errors of other kinds are compute
/// errors that name the plugin.
pub fn ffi_err_to_polars(e: FfiError, package: &str) -> PolarsError {
    let msg = e.message();
    let kind = e.kind();
    match kind {
        FfiErrorKind::COLUMN_NOT_FOUND => PolarsError::ColumnNotFound(msg.into()),
        FfiErrorKind::COMPUTE => PolarsError::ComputeError(msg.into()),
        FfiErrorKind::INVALID_OPERATION | FfiErrorKind::INVALID_INPUT => {
            PolarsError::InvalidOperation(msg.into())
        },
        FfiErrorKind::SCHEMA_MISMATCH => PolarsError::SchemaMismatch(msg.into()),
        FfiErrorKind::SHAPE_MISMATCH => PolarsError::ShapeMismatch(msg.into()),
        FfiErrorKind::NO_DATA => PolarsError::NoData(msg.into()),
        FfiErrorKind::OUT_OF_BOUNDS => PolarsError::OutOfBounds(msg.into()),
        FfiErrorKind::DUPLICATE => PolarsError::Duplicate(msg.into()),
        FfiErrorKind::SQL_INTERFACE => PolarsError::SQLInterface(msg.into()),
        FfiErrorKind::SQL_SYNTAX => PolarsError::SQLSyntax(msg.into()),
        FfiErrorKind::IO => PolarsError::IO {
            error: std::sync::Arc::new(std::io::Error::other(msg)),
            msg: None,
        },
        kind => polars_err!(ComputeError: "{package} ({}): {msg}", kind.name()),
    }
}

/// Imports an Arrow struct schema from a plugin as a schema.
pub fn import_struct_schema(schema: ArrowSchema, package: &str) -> PolarsResult<Schema> {
    polars_ensure!(
        !schema.is_released(),
        ComputeError: "{package} returned a released schema"
    );
    // SAFETY: the contract's struct has the C data interface layout.
    let schema: ffi::ArrowSchema = unsafe { schema.transmute_into() };
    let field = unsafe { ffi::import_field_from_c(&schema)? };
    let ArrowDataType::Struct(fields) = field.dtype else {
        polars_bail!(
            ComputeError:
            "{package} returned a non-struct schema: {:?}", field.dtype
        )
    };
    Ok(fields.iter().map(Field::from).collect())
}
