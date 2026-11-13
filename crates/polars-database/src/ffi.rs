//! Helpers for the plugin side of the contracts: handles, errors, panics and Arrow schemas.
use std::any::Any;
use std::ffi::c_void;
use std::panic::AssertUnwindSafe;
use std::sync::Arc;

use polars_arrow::datatypes::{ArrowDataType, Field as ArrowField};
use polars_arrow::ffi as arrow_ffi;
use polars_core::prelude::*;
use polars_error::{PolarsError, PolarsResult, polars_bail, polars_err};
use polars_io_ext_ffi::common::{ArrowSchema, FfiError, FfiErrorKind, FfiResult};

use crate::PACKAGE_NAME;

/// A handle is an `Arc<T>` turned into a raw pointer: one reference per handle.
pub fn into_handle<T>(value: Arc<T>) -> *const c_void {
    Arc::into_raw(value) as *const c_void
}

/// # Safety
/// `ptr` must be a live handle of type `T`.
pub unsafe fn borrow_handle<'a, T>(ptr: *const c_void) -> &'a T {
    unsafe { &*(ptr as *const T) }
}

/// A new reference.
///
/// # Safety
/// `ptr` must be a live handle of type `T`.
pub unsafe fn clone_handle<T>(ptr: *const c_void) -> *const c_void {
    unsafe { Arc::increment_strong_count(ptr as *const T) };
    ptr
}

/// # Safety
/// `ptr` must be a live handle of type `T`, which is not used afterwards.
pub unsafe fn release_handle<T>(ptr: *const c_void) {
    drop(unsafe { Arc::from_raw(ptr as *const T) });
}

/// The kind and message of a Polars error. The message has no kind prefix: the host raises the
/// error as the same kind, which adds it.
fn flatten_error(e: &PolarsError) -> (FfiErrorKind, String) {
    use PolarsError as E;
    match e {
        E::Context { error, msg } => {
            let (kind, inner) = flatten_error(error);
            (kind, format!("{inner}: {msg}"))
        },
        E::ColumnNotFound(m) => (FfiErrorKind::COLUMN_NOT_FOUND, m.to_string()),
        E::ComputeError(m) => (FfiErrorKind::COMPUTE, m.to_string()),
        E::InvalidOperation(m) => (FfiErrorKind::INVALID_OPERATION, m.to_string()),
        E::SchemaMismatch(m) => (FfiErrorKind::SCHEMA_MISMATCH, m.to_string()),
        E::ShapeMismatch(m) => (FfiErrorKind::SHAPE_MISMATCH, m.to_string()),
        E::NoData(m) => (FfiErrorKind::NO_DATA, m.to_string()),
        E::OutOfBounds(m) => (FfiErrorKind::OUT_OF_BOUNDS, m.to_string()),
        E::Duplicate(m) => (FfiErrorKind::DUPLICATE, m.to_string()),
        E::SQLInterface(m) => (FfiErrorKind::SQL_INTERFACE, m.to_string()),
        E::SQLSyntax(m) => (FfiErrorKind::SQL_SYNTAX, m.to_string()),
        E::IO { .. } => (FfiErrorKind::IO, e.to_string()),
        e => (FfiErrorKind::OTHER, e.to_string()),
    }
}

pub fn to_ffi_error(e: &PolarsError) -> FfiError {
    let (kind, msg) = flatten_error(e);
    FfiError::new(kind, msg)
}

/// The plugin's Polars has a config of its own: follow the host's settings, which Polars'
/// Python `Config` keeps in environment variables.
pub fn sync_config() {
    polars_config::config().reload_env_var("POLARS_VERBOSE");
}

/// Runs a contract function: errors and panics become an [`FfiError`]. Unwinding across
/// `extern "C"` is undefined behavior.
pub fn ffi_call<T>(f: impl FnOnce() -> PolarsResult<T>) -> FfiResult<T> {
    match std::panic::catch_unwind(AssertUnwindSafe(|| {
        sync_config();
        f()
    })) {
        Ok(Ok(v)) => FfiResult::Ok(v),
        Ok(Err(e)) => FfiResult::Err(to_ffi_error(&e)),
        Err(payload) => FfiResult::Err(FfiError::new(
            FfiErrorKind::PANIC,
            format!("{PACKAGE_NAME} panicked: {}", panic_message(&*payload)),
        )),
    }
}

/// [`ffi_call`] for functions that can't fail: a panic aborts the process.
pub fn ffi_call_infallible<T>(f: impl FnOnce() -> T) -> T {
    match std::panic::catch_unwind(AssertUnwindSafe(f)) {
        Ok(v) => v,
        Err(payload) => {
            eprintln!("{PACKAGE_NAME} panicked: {}", panic_message(&*payload));
            std::process::abort()
        },
    }
}

pub fn panic_message(payload: &(dyn Any + Send)) -> &str {
    if let Some(s) = payload.downcast_ref::<&str>() {
        s
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s
    } else {
        "<non-string panic payload>"
    }
}

/// The Arrow fields of a schema.
pub fn arrow_fields(schema: &Schema) -> Vec<ArrowField> {
    schema
        .iter_fields()
        .map(|f| f.to_arrow(CompatLevel::newest()))
        .collect()
}

/// A schema as an Arrow struct schema.
pub fn export_schema(schema: &Schema) -> ArrowSchema {
    let field = ArrowField::new(
        PlSmallStr::EMPTY,
        ArrowDataType::Struct(arrow_fields(schema)),
        false,
    );
    // SAFETY: the contract's struct has the C data interface layout.
    unsafe { ArrowSchema::transmute_from(arrow_ffi::export_field_to_c(&field)) }
}

/// The fields of an Arrow struct field as a schema.
pub fn struct_field_to_schema(field: &ArrowField) -> PolarsResult<Schema> {
    let ArrowDataType::Struct(fields) = field.dtype() else {
        polars_bail!(ComputeError: "expected an Arrow struct schema, got {:?}", field.dtype());
    };
    // Polars panics on an Arrow type it has no dtype for
    fields
        .iter()
        .map(|f| {
            std::panic::catch_unwind(|| Field::from(f)).map_err(|payload| {
                polars_err!(
                    ComputeError: "column '{}' has a type Polars can't read: {}",
                    f.name, panic_message(&*payload)
                )
            })
        })
        .collect()
}
