pub mod version_0;

use std::mem::ManuallyDrop;

use arrow::array::ArrayRef;
use arrow::ffi;
use arrow::ffi::{ArrowArray, ArrowSchema};
use polars_core::error::PolarsResult;
use polars_core::prelude::{ArrowField, Series};

pub const MAJOR: u16 = 0;
pub const MINOR: u16 = 1;

pub const fn get_version() -> (u16, u16) {
    (MAJOR, MINOR)
}

// A utility that helps releasing/owning memory.
#[allow(dead_code)]
struct PrivateData {
    schema: Box<ArrowSchema>,
    arrays: Box<[*mut ArrowArray]>,
}

/// # Safety
/// `ArrowArray` and `ArrowSchema` must be valid
unsafe fn import_array(
    array: ffi::ArrowArray,
    schema: &ffi::ArrowSchema,
) -> PolarsResult<ArrayRef> {
    let field = ffi::import_field_from_c(schema)?;
    let out = ffi::import_array_from_c(array, field.data_type)?;
    Ok(out)
}
