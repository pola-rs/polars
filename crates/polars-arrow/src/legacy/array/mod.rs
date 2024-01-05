use crate::array::{
    new_null_array, Array, BinaryArray, BooleanArray, FixedSizeListArray, ListArray,
    PrimitiveArray, StructArray, Utf8Array,
};
use crate::bitmap::MutableBitmap;
use crate::datatypes::ArrowDataType;
use crate::legacy::utils::CustomIterTools;
use crate::offset::Offsets;
use crate::types::NativeType;

pub mod default_arrays;
#[cfg(feature = "dtype-array")]
pub mod fixed_size_list;
pub mod list;
pub mod null;
pub mod slice;
pub mod utf8;

pub use slice::*;

use crate::legacy::prelude::LargeListArray;

macro_rules! iter_to_values {
    ($iterator:expr, $validity:expr, $offsets:expr, $length_so_far:expr) => {{
        $iterator
            .filter_map(|opt_iter| match opt_iter {
                Some(x) => {
                    let it = x.into_iter();
                    $length_so_far += it.size_hint().0 as i64;
                    $validity.push(true);
                    $offsets.push($length_so_far);
                    Some(it)
                },
                None => {
                    $validity.push(false);
                    $offsets.push($length_so_far);
                    None
                },
            })
            .flatten()
            .collect()
    }};
}

pub trait ListFromIter {
    /// Create a list-array from an iterator.
    /// Used in group_by agg-list
    ///
    /// # Safety
    /// Will produce incorrect arrays if size hint is incorrect.
    unsafe fn from_iter_primitive_trusted_len<T, P, I>(
        iter: I,
        data_type: ArrowDataType,
    ) -> ListArray<i64>
    where
        T: NativeType,
        P: IntoIterator<Item = Option<T>>,
        I: IntoIterator<Item = Option<P>>,
    {
        let iterator = iter.into_iter();
        let (lower, _) = iterator.size_hint();

        let mut validity = MutableBitmap::with_capacity(lower);
        let mut offsets = Vec::<i64>::with_capacity(lower + 1);
        let mut length_so_far = 0i64;
        offsets.push(length_so_far);

        let values: PrimitiveArray<T> = iter_to_values!(iterator, validity, offsets, length_so_far);

        // Safety:
        // offsets are monotonically increasing
        ListArray::new(
            ListArray::<i64>::default_datatype(data_type.clone()),
            Offsets::new_unchecked(offsets).into(),
            Box::new(values.to(data_type)),
            Some(validity.into()),
        )
    }

    /// Create a list-array from an iterator.
    /// Used in group_by agg-list
    ///
    /// # Safety
    /// Will produce incorrect arrays if size hint is incorrect.
    unsafe fn from_iter_bool_trusted_len<I, P>(iter: I) -> ListArray<i64>
    where
        I: IntoIterator<Item = Option<P>>,
        P: IntoIterator<Item = Option<bool>>,
    {
        let iterator = iter.into_iter();
        let (lower, _) = iterator.size_hint();

        let mut validity = Vec::with_capacity(lower);
        let mut offsets = Vec::<i64>::with_capacity(lower + 1);
        let mut length_so_far = 0i64;
        offsets.push(length_so_far);

        let values: BooleanArray = iter_to_values!(iterator, validity, offsets, length_so_far);

        // Safety:
        // Offsets are monotonically increasing.
        ListArray::new(
            ListArray::<i64>::default_datatype(ArrowDataType::Boolean),
            Offsets::new_unchecked(offsets).into(),
            Box::new(values),
            Some(validity.into()),
        )
    }

    /// Create a list-array from an iterator.
    /// Used in group_by agg-list
    ///
    /// # Safety
    /// Will produce incorrect arrays if size hint is incorrect.
    unsafe fn from_iter_utf8_trusted_len<I, P, Ref>(iter: I, n_elements: usize) -> ListArray<i64>
    where
        I: IntoIterator<Item = Option<P>>,
        P: IntoIterator<Item = Option<Ref>>,
        Ref: AsRef<str>,
    {
        let iterator = iter.into_iter();
        let (lower, _) = iterator.size_hint();

        let mut validity = MutableBitmap::with_capacity(lower);
        let mut offsets = Vec::<i64>::with_capacity(lower + 1);
        let mut length_so_far = 0i64;
        offsets.push(length_so_far);
        let values: Utf8Array<i64> = iterator
            .filter_map(|opt_iter| match opt_iter {
                Some(x) => {
                    let it = x.into_iter();
                    length_so_far += it.size_hint().0 as i64;
                    validity.push(true);
                    offsets.push(length_so_far);
                    Some(it)
                },
                None => {
                    validity.push(false);
                    offsets.push(length_so_far);
                    None
                },
            })
            .flatten()
            .trust_my_length(n_elements)
            .collect();

        // Safety:
        // offsets are monotonically increasing
        ListArray::new(
            ListArray::<i64>::default_datatype(ArrowDataType::LargeUtf8),
            Offsets::new_unchecked(offsets).into(),
            Box::new(values),
            Some(validity.into()),
        )
    }

    /// Create a list-array from an iterator.
    /// Used in group_by agg-list
    ///
    /// # Safety
    /// Will produce incorrect arrays if size hint is incorrect.
    unsafe fn from_iter_binary_trusted_len<I, P, Ref>(iter: I, n_elements: usize) -> ListArray<i64>
    where
        I: IntoIterator<Item = Option<P>>,
        P: IntoIterator<Item = Option<Ref>>,
        Ref: AsRef<[u8]>,
    {
        let iterator = iter.into_iter();
        let (lower, _) = iterator.size_hint();

        let mut validity = MutableBitmap::with_capacity(lower);
        let mut offsets = Vec::<i64>::with_capacity(lower + 1);
        let mut length_so_far = 0i64;
        offsets.push(length_so_far);
        let values: BinaryArray<i64> = iterator
            .filter_map(|opt_iter| match opt_iter {
                Some(x) => {
                    let it = x.into_iter();
                    length_so_far += it.size_hint().0 as i64;
                    validity.push(true);
                    offsets.push(length_so_far);
                    Some(it)
                },
                None => {
                    validity.push(false);
                    offsets.push(length_so_far);
                    None
                },
            })
            .flatten()
            .trust_my_length(n_elements)
            .collect();

        // Safety:
        // offsets are monotonically increasing
        ListArray::new(
            ListArray::<i64>::default_datatype(ArrowDataType::LargeBinary),
            Offsets::new_unchecked(offsets).into(),
            Box::new(values),
            Some(validity.into()),
        )
    }
}
impl ListFromIter for ListArray<i64> {}

pub trait PolarsArray: Array {
    fn has_validity(&self) -> bool {
        self.validity().is_some()
    }
}

impl<A: Array + ?Sized> PolarsArray for A {}

fn is_nested_null(data_type: &ArrowDataType) -> bool {
    match data_type {
        ArrowDataType::Null => true,
        ArrowDataType::LargeList(field) => is_nested_null(field.data_type()),
        ArrowDataType::FixedSizeList(field, _) => is_nested_null(field.data_type()),
        ArrowDataType::Struct(fields) => {
            fields.iter().all(|field| is_nested_null(field.data_type()))
        },
        _ => false,
    }
}

/// Cast null arrays to inner type and ensure that all offsets remain correct
pub fn convert_inner_type(array: &dyn Array, dtype: &ArrowDataType) -> Box<dyn Array> {
    match dtype {
        ArrowDataType::LargeList(field) => {
            let array = array.as_any().downcast_ref::<LargeListArray>().unwrap();
            let inner = array.values();
            let new_values = convert_inner_type(inner.as_ref(), field.data_type());
            let dtype = LargeListArray::default_datatype(new_values.data_type().clone());
            LargeListArray::new(
                dtype,
                array.offsets().clone(),
                new_values,
                array.validity().cloned(),
            )
            .boxed()
        },
        ArrowDataType::FixedSizeList(field, width) => {
            let array = array.as_any().downcast_ref::<FixedSizeListArray>().unwrap();
            let inner = array.values();
            let new_values = convert_inner_type(inner.as_ref(), field.data_type());
            let dtype =
                FixedSizeListArray::default_datatype(new_values.data_type().clone(), *width);
            FixedSizeListArray::new(dtype, new_values, array.validity().cloned()).boxed()
        },
        ArrowDataType::Struct(fields) => {
            let array = array.as_any().downcast_ref::<StructArray>().unwrap();
            let inner = array.values();
            let new_values = inner
                .iter()
                .zip(fields)
                .map(|(arr, field)| convert_inner_type(arr.as_ref(), field.data_type()))
                .collect::<Vec<_>>();
            StructArray::new(dtype.clone(), new_values, array.validity().cloned()).boxed()
        },
        _ => new_null_array(dtype.clone(), array.len()),
    }
}
