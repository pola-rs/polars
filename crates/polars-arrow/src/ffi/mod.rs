//! contains FFI bindings to import and export [`Array`](crate::array::Array) via
//! Arrow's [C Data Interface](https://arrow.apache.org/docs/format/CDataInterface.html)
mod array;
mod bridge;
mod generated;
pub mod mmap;
mod schema;
mod stream;

pub(crate) use array::{try_from, ArrowArrayRef, InternalArrowArray};
pub use generated::{ArrowArray, ArrowArrayStream, ArrowSchema};
use polars_error::PolarsResult;
pub use stream::{export_iterator, ArrowArrayStreamReader};

use self::schema::to_field;
use crate::array::Array;
use crate::datatypes::{ArrowDataType, Field};

/// Exports an [`Box<dyn Array>`] to the C data interface.
pub fn export_array_to_c(array: Box<dyn Array>) -> ArrowArray {
    ArrowArray::new(bridge::align_to_c_data_interface(array))
}

/// Exports a [`Field`] to the C data interface.
pub fn export_field_to_c(field: &Field) -> ArrowSchema {
    ArrowSchema::new(field)
}

/// Imports a [`Field`] from the C data interface.
/// # Safety
/// This function is intrinsically `unsafe` and relies on a [`ArrowSchema`]
/// being valid according to the [C data interface](https://arrow.apache.org/docs/format/CDataInterface.html) (FFI).
pub unsafe fn import_field_from_c(field: &ArrowSchema) -> PolarsResult<Field> {
    to_field(field)
}

/// Imports an [`Array`] from the C data interface.
/// # Safety
/// This function is intrinsically `unsafe` and relies on a [`ArrowArray`]
/// being valid according to the [C data interface](https://arrow.apache.org/docs/format/CDataInterface.html) (FFI).
pub unsafe fn import_array_from_c(
    array: ArrowArray,
    data_type: ArrowDataType,
) -> PolarsResult<Box<dyn Array>> {
    try_from(InternalArrowArray::new(array, data_type))
}
