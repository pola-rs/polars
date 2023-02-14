use arrow::array::{Array, BinaryArray, BooleanArray, ListArray, PrimitiveArray, Utf8Array};
use arrow::bitmap::MutableBitmap;
use arrow::datatypes::DataType;
use arrow::offset::Offsets;
use arrow::types::NativeType;

use crate::prelude::*;
use crate::utils::CustomIterTools;

pub mod default_arrays;
mod get;
pub mod list;
pub mod slice;

pub use get::ArrowGetItem;
pub use slice::*;

pub trait ValueSize {
    /// Useful for a Utf8 or a List to get underlying value size.
    /// During a rechunk this is handy
    fn get_values_size(&self) -> usize;
}

impl ValueSize for ListArray<i64> {
    fn get_values_size(&self) -> usize {
        self.values().len()
    }
}

impl ValueSize for Utf8Array<i64> {
    fn get_values_size(&self) -> usize {
        self.values().len()
    }
}

impl ValueSize for BinaryArray<i64> {
    fn get_values_size(&self) -> usize {
        self.values().len()
    }
}

impl ValueSize for ArrayRef {
    fn get_values_size(&self) -> usize {
        match self.data_type() {
            DataType::LargeUtf8 => self
                .as_any()
                .downcast_ref::<Utf8Array<i64>>()
                .unwrap()
                .get_values_size(),
            DataType::LargeList(_) => self
                .as_any()
                .downcast_ref::<ListArray<i64>>()
                .unwrap()
                .get_values_size(),
            _ => unimplemented!(),
        }
    }
}

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
                }
                None => {
                    $validity.push(false);
                    None
                }
            })
            .flatten()
            .collect()
    }};
}

pub trait ListFromIter {
    /// Create a list-array from an iterator.
    /// Used in groupby agg-list
    ///
    /// # Safety
    /// Will produce incorrect arrays if size hint is incorrect.
    unsafe fn from_iter_primitive_trusted_len<T, P, I>(
        iter: I,
        data_type: DataType,
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
    /// Used in groupby agg-list
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
            ListArray::<i64>::default_datatype(DataType::Boolean),
            Offsets::new_unchecked(offsets).into(),
            Box::new(values),
            Some(validity.into()),
        )
    }

    /// Create a list-array from an iterator.
    /// Used in groupby agg-list
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
                }
                None => {
                    validity.push(false);
                    None
                }
            })
            .flatten()
            .trust_my_length(n_elements)
            .collect();

        // Safety:
        // offsets are monotonically increasing
        ListArray::new(
            ListArray::<i64>::default_datatype(DataType::LargeUtf8),
            Offsets::new_unchecked(offsets).into(),
            Box::new(values),
            Some(validity.into()),
        )
    }

    /// Create a list-array from an iterator.
    /// Used in groupby agg-list
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
                }
                None => {
                    validity.push(false);
                    None
                }
            })
            .flatten()
            .trust_my_length(n_elements)
            .collect();

        // Safety:
        // offsets are monotonically increasing
        ListArray::new(
            ListArray::<i64>::default_datatype(DataType::LargeBinary),
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
