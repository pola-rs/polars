use crate::array::{ArrayRef, PrimitiveArray, StructArray};
use crate::datatypes::{ArrowDataType, Field};
use crate::record_batch::RecordBatchT;
use crate::types::NativeType;

pub fn chunk_to_struct(chunk: RecordBatchT<ArrayRef>, fields: Vec<Field>) -> StructArray {
    let dtype = ArrowDataType::Struct(fields);
    StructArray::new(dtype, chunk.into_arrays(), None)
}

/// Returns its underlying [`Vec`], if possible.
///
/// This operation returns [`Some`] iff this [`PrimitiveArray`]:
/// * has not been sliced with an offset
/// * has not been cloned (i.e. [`Arc::get_mut`][Arc::get_mut] yields [`Some`])
/// * has not been imported from the c data interface (FFI)
///
/// [Arc::get_mut]: std::sync::Arc::get_mut
pub fn primitive_to_vec<T: NativeType>(arr: ArrayRef) -> Option<Vec<T>> {
    let arr_ref = arr.as_any().downcast_ref::<PrimitiveArray<T>>().unwrap();
    let buffer = arr_ref.values().clone();
    drop(arr); // Drop original reference so refcount becomes 1 if possible.
    buffer.into_mut().right()
}
