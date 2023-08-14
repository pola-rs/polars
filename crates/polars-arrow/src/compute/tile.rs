use arrow::array::{Array, PrimitiveArray};
use arrow::bitmap::MutableBitmap;
use arrow::types::NativeType;

pub fn tile_primitive<T: NativeType>(arr: &PrimitiveArray<T>, n: usize) -> PrimitiveArray<T> {
    let slice = arr.values().as_slice();
    let mut out = Vec::with_capacity(slice.len() * n);

    for _ in 0..n {
        out.extend_from_slice(slice);
    }
    let validity = if arr.null_count() > 0 {
        let mut new_validity = MutableBitmap::with_capacity(slice.len() * n);
        let (slice, offset, len) = arr.validity().unwrap().as_slice();

        for _ in 0..n {
            new_validity.extend_from_slice(slice, offset, len)
        }

        Some(new_validity.into())
    } else {
        None
    };
    PrimitiveArray::new(arr.data_type().clone(), out.into(), validity)
}
