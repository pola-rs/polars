use arrow::array::{Array, ArrayRef, FixedSizeListArray, PrimitiveArray, ArrayCollectIterExt, StaticArray};
use arrow::datatypes::ArrowDataType;
use arrow::datatypes::PhysicalType;
use arrow::legacy::prelude::*;
use arrow::legacy::utils::CustomIterTools;
use polars_error::{PolarsResult, polars_bail};
use polars_utils::IdxSize;

use crate::gather::take_unchecked;

use arrow::with_match_primitive_type;
use num_traits::ToPrimitive;
use polars_utils::pl_str::PlSmallStr;

/// Converts an index value to an array index, handling negative indexing.
trait ToArrayIndex: Copy {
    fn to_array_index(self, len: usize) -> Option<usize>;
}

macro_rules! impl_to_array_index {
    (signed: $($t:ty),*) => {
        $(
            impl ToArrayIndex for $t {
                #[inline]
                fn to_array_index(self, len: usize) -> Option<usize> {
                    (self as i64).negative_to_usize(len)
                }
            }
        )*
    };
    (unsigned: $($t:ty),*) => {
        $(
            impl ToArrayIndex for $t {
                #[inline]
                fn to_array_index(self, len: usize) -> Option<usize> {
                    let idx = self as usize;
                    (idx < len).then_some(idx)
                }
            }
        )*
    };
}

impl_to_array_index!(signed: i8, i16, i32, i64);
impl_to_array_index!(unsigned: u8, u16, u32, u64);

fn sub_fixed_size_list_get_indexes_literal(width: usize, len: usize, index: i64) -> IdxArr {
    (0..len)
        .map(|i| {
            if index >= width as i64 {
                return None;
            }
            index
                .negative_to_usize(width)
                .map(|idx| (idx + i * width) as IdxSize)
        })
        .collect_trusted()
}

fn sub_fixed_size_list_get_indexes(width: usize, index: &PrimitiveArray<i64>) -> IdxArr {
    index
        .iter()
        .enumerate()
        .map(|(i, idx)| {
            idx.and_then(|&idx| {
                if idx >= width as i64 {
                    return None;
                }
                idx.negative_to_usize(width)
                    .map(|idx| (idx + i * width) as IdxSize)
            })
        })
        .collect_trusted()
}

pub fn sub_fixed_size_list_get_literal(
    arr: &FixedSizeListArray,
    index: i64,
    null_on_oob: bool,
) -> PolarsResult<ArrayRef> {
    let take_by = sub_fixed_size_list_get_indexes_literal(arr.size(), arr.len(), index);
    if !null_on_oob && take_by.has_nulls() {
        polars_bail!(ComputeError: "get index is out of bounds");
    }
    // SAFETY: the indices we generate are in bounds
    unsafe { Ok(take_unchecked(&**arr.values(), &take_by)) }
}

pub fn sub_fixed_size_list_get(
    arr: &FixedSizeListArray,
    index: &PrimitiveArray<i64>,
    null_on_oob: bool,
) -> PolarsResult<ArrayRef> {
    let take_by = sub_fixed_size_list_get_indexes(arr.size(), index);
    if !null_on_oob && take_by.has_nulls() {
        polars_bail!(ComputeError: "get index is out of bounds");
    }
    // SAFETY: the indices we generate are in bounds
    unsafe { Ok(take_unchecked(&**arr.values(), &take_by)) }
}

/// Single-pass gather for primitive inner types.
///
/// - Uses `get_unchecked` which returns `Option<T>` and handles validity
/// - Outer validity is handled by caller, not here
///
/// # Safety
/// Caller must ensure indices are within bounds or `null_on_oob` is true.
unsafe fn gather_inner_primitive<V, I>(
    values_inner: &PrimitiveArray<V>,
    indices_inner: &PrimitiveArray<I>,
    value_width: usize,
    output_width: usize,
    num_rows: usize,
    null_on_oob: bool,
    dtype: &ArrowDataType,
) -> PolarsResult<PrimitiveArray<V>>
where
    V: arrow::types::NativeType,
    I: arrow::types::NativeType + ToPrimitive,
{
    let out_len = num_rows * output_width;
    let mut any_oob = false;

    // Reference: horizontal_flatten uses (0..out_len).map(...).collect_arr_trusted_with_dtype()
    let result: PrimitiveArray<V> = (0..out_len)
        .map(|flat_idx| {
            let row = flat_idx / output_width;
            let elem = flat_idx % output_width;
            let idx_pos = row * output_width + elem;

            // get_unchecked returns Option<T>, handles inner validity automatically
            // Reference: horizontal_flatten/mod.rs:172
            let opt_raw_idx = indices_inner.get_unchecked(idx_pos);

            match opt_raw_idx {
                None => None,
                Some(raw_idx) => {
                    let raw_idx_i64 = raw_idx.to_i64().unwrap_or(i64::MAX);

                    match raw_idx_i64.negative_to_usize(value_width) {
                        Some(local_idx) => {
                            let abs_pos = row * value_width + local_idx;
                            // get_unchecked handles value validity automatically
                            values_inner.get_unchecked(abs_pos)
                        },
                        None => {
                            any_oob = true;
                            None
                        },
                    }
                },
            }
        })
        .collect_arr_trusted_with_dtype(dtype.clone());

    if any_oob && !null_on_oob {
        polars_bail!(ComputeError: "gather index is out of bounds");
    }

    Ok(result)
}

/// Dispatch gather based on index type.
///
/// # Safety
/// Caller must ensure indices are within bounds or `null_on_oob` is true.
unsafe fn gather_inner_primitive_idx_dispatch<V>(
    values_inner: &PrimitiveArray<V>,
    indices_inner: &dyn Array,
    value_width: usize,
    output_width: usize,
    num_rows: usize,
    null_on_oob: bool,
    dtype: &ArrowDataType,
) -> PolarsResult<PrimitiveArray<V>>
where
    V: arrow::types::NativeType,
{
    use ArrowDataType::*;
    match indices_inner.dtype() {
        Int8 => {
            let idx = indices_inner.as_any().downcast_ref::<PrimitiveArray<i8>>().unwrap();
            gather_inner_primitive(values_inner, idx, value_width, output_width, num_rows, null_on_oob, dtype)
        },
        Int16 => {
            let idx = indices_inner.as_any().downcast_ref::<PrimitiveArray<i16>>().unwrap();
            gather_inner_primitive(values_inner, idx, value_width, output_width, num_rows, null_on_oob, dtype)
        },
        Int32 => {
            let idx = indices_inner.as_any().downcast_ref::<PrimitiveArray<i32>>().unwrap();
            gather_inner_primitive(values_inner, idx, value_width, output_width, num_rows, null_on_oob, dtype)
        },
        Int64 => {
            let idx = indices_inner.as_any().downcast_ref::<PrimitiveArray<i64>>().unwrap();
            gather_inner_primitive(values_inner, idx, value_width, output_width, num_rows, null_on_oob, dtype)
        },
        UInt8 => {
            let idx = indices_inner.as_any().downcast_ref::<PrimitiveArray<u8>>().unwrap();
            gather_inner_primitive(values_inner, idx, value_width, output_width, num_rows, null_on_oob, dtype)
        },
        UInt16 => {
            let idx = indices_inner.as_any().downcast_ref::<PrimitiveArray<u16>>().unwrap();
            gather_inner_primitive(values_inner, idx, value_width, output_width, num_rows, null_on_oob, dtype)
        },
        UInt32 => {
            let idx = indices_inner.as_any().downcast_ref::<PrimitiveArray<u32>>().unwrap();
            gather_inner_primitive(values_inner, idx, value_width, output_width, num_rows, null_on_oob, dtype)
        },
        UInt64 => {
            let idx = indices_inner.as_any().downcast_ref::<PrimitiveArray<u64>>().unwrap();
            gather_inner_primitive(values_inner, idx, value_width, output_width, num_rows, null_on_oob, dtype)
        },
        dt => polars_bail!(ComputeError: "cannot use dtype `{:?}` as gather index", dt),
    }
}

/// Gather elements from a FixedSizeListArray, returning inner values.
///
/// Reference: horizontal_flatten/mod.rs:23-127 for type dispatch pattern
///
/// # Safety
/// Caller must ensure indices are within bounds or `null_on_oob` is true.
unsafe fn gather_fixed_size_list_inner(
    arr: &FixedSizeListArray,
    indices: &FixedSizeListArray,
    null_on_oob: bool,
) -> PolarsResult<Box<dyn Array>> {
    let value_width = arr.size();
    let output_width = indices.size();
    let num_rows = arr.len();

    let values_inner = arr.values();
    let indices_inner = indices.values();
    let dtype = values_inner.dtype();

    // Reference: horizontal_flatten/mod.rs:30-127 for type dispatch
    match dtype.to_physical_type() {
        PhysicalType::Primitive(primitive) => {
            with_match_primitive_type!(primitive, |$T| {
                let values_prim = values_inner
                    .as_any()
                    .downcast_ref::<PrimitiveArray<$T>>()
                    .unwrap();

                let result = gather_inner_primitive_idx_dispatch(
                    values_prim,
                    indices_inner.as_ref(),
                    value_width,
                    output_width,
                    num_rows,
                    null_on_oob,
                    dtype,
                )?;

                Ok(Box::new(result) as Box<dyn Array>)
            })
        },
        t => polars_bail!(ComputeError: "arr.gather not yet supported for {:?}", t),
    }
}

/// Gather elements from each sub-array by index.
///
/// Returns FixedSizeListArray (Array dtype), following arr.eval semantics.
///
/// # Arguments
/// * `arr` - Source FixedSizeListArray
/// * `indices` - FixedSizeListArray of indices (must have integer inner type)
/// * `null_on_oob` - If true, out-of-bounds indices produce null; otherwise error
pub fn sub_fixed_size_list_gather(
    arr: &FixedSizeListArray,
    indices: &FixedSizeListArray,
    null_on_oob: bool,
) -> PolarsResult<FixedSizeListArray> {
    let output_width = indices.size();
    let num_rows = arr.len();
    let inner_dtype = arr.values().dtype().clone();

    // Gather the inner values
    let gathered_inner = unsafe { gather_fixed_size_list_inner(arr, indices, null_on_oob)? };

    // Combine outer validity (Reference: horizontal_flatten/struct_.rs:47-56)
    let validity = match (arr.validity(), indices.validity()) {
        (None, None) => None,
        (Some(v), None) | (None, Some(v)) => Some(v.clone()),
        (Some(a), Some(b)) => Some(a & b),
    };

    // Return as FixedSizeListArray (Array dtype)
    let dtype = ArrowDataType::FixedSizeList(
        Box::new(arrow::datatypes::Field::new(
            PlSmallStr::from_static("item"),
            inner_dtype,
            true,
        )),
        output_width,
    );

    Ok(FixedSizeListArray::new(
        dtype,
        num_rows,
        gathered_inner,
        validity,
    ))
}

#[cfg(test)]
mod test {
    use super::*;

    fn fixture_array() -> FixedSizeListArray {
        // [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        let values = PrimitiveArray::from_vec(vec![1i64, 2, 3, 4, 5, 6, 7, 8, 9]);
        FixedSizeListArray::new(
            ArrowDataType::FixedSizeList(
                Box::new(arrow::datatypes::Field::new(
                    PlSmallStr::from_static("item"),
                    ArrowDataType::Int64,
                    true,
                )),
                3,
            ),
            3,
            Box::new(values),
            None,
        )
    }

    fn make_indices_array(values: Vec<i64>, width: usize, len: usize) -> FixedSizeListArray {
        let arr = PrimitiveArray::<i64>::from_vec(values);
        FixedSizeListArray::new(
            ArrowDataType::FixedSizeList(
                Box::new(arrow::datatypes::Field::new(
                    PlSmallStr::from_static("item"),
                    ArrowDataType::Int64,
                    true,
                )),
                width,
            ),
            len,
            Box::new(arr),
            None,
        )
    }

    #[test]
    fn test_gather_returns_fixed_size_list_by_default() {
        let arr = fixture_array();
        let indices = make_indices_array(vec![0, 2, 0, 2, 0, 2], 2, 3);

        let result = sub_fixed_size_list_gather(&arr, &indices, false).unwrap();

        assert!(result.as_any().downcast_ref::<FixedSizeListArray>().is_some());

        let result = result.as_any().downcast_ref::<FixedSizeListArray>().unwrap();
        assert_eq!(result.size(), 2);
        assert_eq!(result.len(), 3);

        let inner = result.values().as_any().downcast_ref::<PrimitiveArray<i64>>().unwrap();
        assert_eq!(inner.values().as_slice(), &[1, 3, 4, 6, 7, 9]);
    }

    #[test]
    fn test_gather_negative_index() {
        let arr = fixture_array();
        // [[-1], [-1], [-1]] -> last element of each
        let indices = make_indices_array(vec![-1, -1, -1], 1, 3);

        let result = sub_fixed_size_list_gather(&arr, &indices, false).unwrap();
        let result = result.as_any().downcast_ref::<FixedSizeListArray>().unwrap();

        let inner = result.values().as_any().downcast_ref::<PrimitiveArray<i64>>().unwrap();
        assert_eq!(inner.values().as_slice(), &[3, 6, 9]);
    }

    #[test]
    fn test_gather_oob_error() {
        let arr = fixture_array();
        let indices = make_indices_array(vec![10, 0, 0], 1, 3);

        let result = sub_fixed_size_list_gather(&arr, &indices, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_gather_oob_null() {
        let arr = fixture_array();
        let indices = make_indices_array(vec![10, 0, 0], 1, 3);

        let result = sub_fixed_size_list_gather(&arr, &indices, true).unwrap();
        let result = result.as_any().downcast_ref::<FixedSizeListArray>().unwrap();

        let inner = result.values().as_any().downcast_ref::<PrimitiveArray<i64>>().unwrap();
        assert!(!inner.is_valid(0)); // OOB -> null
        assert!(inner.is_valid(1));
        assert!(inner.is_valid(2));
    }

    #[test]
    fn test_gather_with_null_index() {
        use arrow::bitmap::Bitmap;

        let arr = fixture_array();
        // Indices with a null in position 1
        let idx_values = PrimitiveArray::<i64>::from_vec(vec![0, 0, 0]).with_validity(Some(Bitmap::from_iter([true, false, true])));
        let indices = FixedSizeListArray::new(
            ArrowDataType::FixedSizeList(
                Box::new(arrow::datatypes::Field::new(
                    PlSmallStr::from_static("item"),
                    ArrowDataType::Int64,
                    true,
                )),
                1,
            ),
            3,
            Box::new(idx_values),
            None,
        );

        let result = sub_fixed_size_list_gather(&arr, &indices, false).unwrap();
        let result = result.as_any().downcast_ref::<FixedSizeListArray>().unwrap();

        let inner = result.values().as_any().downcast_ref::<PrimitiveArray<i64>>().unwrap();
        assert!(inner.is_valid(0));
        assert!(!inner.is_valid(1)); // null index -> null value
        assert!(inner.is_valid(2));
    }

    #[test]
    fn test_gather_with_outer_validity() {
        use arrow::bitmap::Bitmap;

        // Source array with row 1 null
        let values = PrimitiveArray::from_vec(vec![1i64, 2, 3, 4, 5, 6, 7, 8, 9]);
        let arr = FixedSizeListArray::new(
            ArrowDataType::FixedSizeList(
                Box::new(arrow::datatypes::Field::new(
                    PlSmallStr::from_static("item"),
                    ArrowDataType::Int64,
                    true,
                )),
                3,
            ),
            3,
            Box::new(values),
            Some(Bitmap::from_iter([true, false, true])), // row 1 is null
        );

        let indices = make_indices_array(vec![0, 0, 0], 1, 3);

        let result = sub_fixed_size_list_gather(&arr, &indices, false).unwrap();
        let result = result.as_any().downcast_ref::<FixedSizeListArray>().unwrap();

        // Outer validity should propagate
        assert!(result.validity().is_some());
        let validity = result.validity().unwrap();
        assert!(validity.get_bit(0));
        assert!(!validity.get_bit(1)); // row 1 null
        assert!(validity.get_bit(2));
    }
}
