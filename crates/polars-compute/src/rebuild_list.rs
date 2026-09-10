use arrow::array::{Array, ListArray};
use arrow::datatypes::ArrowDataType;
use arrow::offset::OffsetsBuffer;
use arrow::types::Offset;

/// Rebuild `arr` around replacement `values` for the elements its offsets span.
///
/// `values` must hold exactly those elements, in the same order, so that rebasing the offsets
/// onto them keeps every row length. The outer validity and the row count are preserved, and
/// `dtype` becomes the dtype of the result, so that callers can either keep the original field
/// metadata or change the child dtype.
///
/// Only the outer layer is rebuilt. Descendants of `values` keep their own offsets and backing
/// buffers, so the result is only as normalized as `values` is.
///
/// Rebasing allocates only when the offsets do not already start at zero.
pub fn rebuild_list_shallow<O: Offset>(
    arr: &ListArray<O>,
    dtype: ArrowDataType,
    values: Box<dyn Array>,
) -> ListArray<O> {
    let offsets = arr.offsets();
    assert_eq!(
        values.len(),
        offsets.range().to_usize(),
        "replacement values must cover exactly the offsets of the list"
    );

    let first = *offsets.first();
    let offsets = if first.to_usize() == 0 {
        offsets.clone()
    } else {
        let rebased: Vec<O> = offsets.iter().map(|offset| *offset - first).collect();
        // SAFETY: subtracting the first offset keeps the offsets monotonic and starts them
        // at zero.
        unsafe { OffsetsBuffer::new_unchecked(rebased.into()) }
    };

    ListArray::new(dtype, offsets, values, arr.validity().cloned())
}

#[cfg(test)]
mod tests {
    use arrow::array::{PrimitiveArray, Utf8ViewArray};
    use arrow::bitmap::Bitmap;
    use arrow::datatypes::Field;
    use polars_utils::pl_str::PlSmallStr;

    use super::*;

    fn i32s(len: i32) -> Box<dyn Array> {
        PrimitiveArray::<i32>::from_vec((0..len).collect()).boxed()
    }

    fn list<O: Offset>(
        offsets: &[O],
        values: Box<dyn Array>,
        validity: Option<&[bool]>,
    ) -> ListArray<O> {
        let dtype = ListArray::<O>::default_datatype(values.dtype().clone());
        // SAFETY: the tests pass monotonic offsets within the child.
        let offsets = unsafe { OffsetsBuffer::new_unchecked(offsets.to_vec().into()) };
        ListArray::new(dtype, offsets, values, validity.map(Bitmap::from))
    }

    fn rebases_a_nonzero_window<O: Offset>(offsets: &[O], expected: &[O]) {
        let arr = list(offsets, i32s(8), Some(&[true, false]));
        let out = rebuild_list_shallow(&arr, arr.dtype().clone(), i32s(3));

        assert_eq!(out.offsets().as_slice(), expected);
        assert_eq!(out.len(), 2);
        assert_eq!(out.validity(), arr.validity());
        assert_eq!(out.values().len(), 3);
    }

    #[test]
    fn rebases_a_nonzero_window_for_both_offset_widths() {
        rebases_a_nonzero_window::<i32>(&[2, 4, 5], &[0, 2, 3]);
        rebases_a_nonzero_window::<i64>(&[2, 4, 5], &[0, 2, 3]);
    }

    #[test]
    fn reuses_zero_based_offsets() {
        let arr = list::<i64>(&[0, 2, 3], i32s(3), None);
        let out = rebuild_list_shallow(&arr, arr.dtype().clone(), i32s(3));

        assert_eq!(
            out.offsets().as_slice().as_ptr(),
            arr.offsets().as_slice().as_ptr(),
            "zero-based offsets must not be reallocated"
        );
    }

    #[test]
    fn rebuilds_an_empty_window() {
        let arr = list::<i64>(&[3, 3, 3], i32s(5), None);
        let out = rebuild_list_shallow(&arr, arr.dtype().clone(), i32s(0));

        assert_eq!(out.offsets().as_slice(), &[0, 0, 0]);
        assert_eq!(out.len(), 2);
        assert!(out.values().is_empty());
    }

    #[test]
    fn takes_the_dtype_from_the_caller() {
        let field = Field::new(
            PlSmallStr::from_static("entry"),
            ArrowDataType::Int32,
            false,
        );
        let arr = ListArray::<i64>::new(
            ArrowDataType::LargeList(Box::new(field)),
            unsafe { OffsetsBuffer::new_unchecked(vec![1, 3].into()) },
            i32s(4),
            None,
        );

        // The original field name and nullability survive a same-dtype rebuild.
        let out = rebuild_list_shallow(&arr, arr.dtype().clone(), i32s(2));
        assert_eq!(out.dtype(), arr.dtype());

        // A cast of the child needs a new dtype instead.
        let values = Utf8ViewArray::from_slice([Some("a"), Some("b")]).boxed();
        let dtype = ListArray::<i64>::default_datatype(values.dtype().clone());
        let out = rebuild_list_shallow(&arr, dtype.clone(), values);
        assert_eq!(out.dtype(), &dtype);
    }

    #[test]
    fn leaves_descendants_untouched() {
        // An inner list that keeps its own nonzero offsets into its own values.
        let inner = list::<i64>(&[4, 5, 7], i32s(8), None);
        let arr = list::<i64>(&[1, 2], inner.boxed(), None);

        let values = arr.values().sliced(1, 1);
        let out = rebuild_list_shallow(&arr, arr.dtype().clone(), values);

        assert_eq!(out.offsets().as_slice(), &[0, 1]);
        let inner = out.values();
        let inner = inner.as_any().downcast_ref::<ListArray<i64>>().unwrap();
        assert_eq!(inner.offsets().as_slice(), &[5, 7]);
        assert_eq!(inner.values().len(), 8);
    }

    #[test]
    #[should_panic(expected = "must cover exactly")]
    fn rejects_values_that_do_not_cover_the_window() {
        let arr = list::<i64>(&[0, 2], i32s(4), None);
        rebuild_list_shallow(&arr, arr.dtype().clone(), i32s(4));
    }
}
