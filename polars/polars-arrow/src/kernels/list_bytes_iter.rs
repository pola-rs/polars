use arrow::array::{ListArray, PrimitiveArray};
use arrow::bitmap::Bitmap;
use arrow::datatypes::PhysicalType::Primitive;
use arrow::types::NativeType;
use polars_error::{polars_bail, polars_ensure, PolarsResult};

use crate::utils::with_match_primitive_type;

unsafe fn bytes_iter<'a, T: NativeType>(
    values: &'a [T],
    offsets: &'a [i64],
    validity: Option<&'a Bitmap>,
) -> impl ExactSizeIterator<Item = Option<&'a [u8]>> {
    let mut start = offsets[0] as usize;
    offsets[1..].iter().enumerate().map(move |(i, end)| {
        let end = *end as usize;
        let out = values.get_unchecked(start..end);
        start = end;

        let data = out.as_ptr() as *const u8;
        let out = std::slice::from_raw_parts(data, std::mem::size_of_val(out));
        match validity {
            None => Some(out),
            Some(validity) => {
                if validity.get_bit_unchecked(i) {
                    Some(out)
                } else {
                    None
                }
            }
        }
    })
}

pub fn numeric_list_bytes_iter(
    arr: &ListArray<i64>,
) -> PolarsResult<Box<dyn ExactSizeIterator<Item = Option<&[u8]>> + '_>> {
    let values = arr.values();
    polars_ensure!(
        values.null_count() == 0,
        ComputeError: "only allowed for child arrays without nulls"
    );
    let offsets = arr.offsets().as_slice();
    let validity = arr.validity();

    if let Primitive(primitive) = values.data_type().to_physical_type() {
        with_match_primitive_type!(primitive, |$T| {
            let arr: &PrimitiveArray<$T> = values.as_any().downcast_ref().unwrap();
            let values = arr.values();
            let iter = unsafe { bytes_iter(values.as_slice(), offsets, validity) };
            Ok(Box::new(iter))
        })
    } else {
        polars_bail!(ComputeError: "only allowed for numeric child arrays");
    }
}
