use std::sync::Arc;

use polars_error::{polars_bail, PolarsResult};

use super::{MutableUtf8Array, StrAsBytes, Utf8Array};
use crate::array::physical_binary::*;
use crate::array::specification::{try_check_offsets_bounds, try_check_utf8};
use crate::array::{Array, ArrayValuesIter, MutableArray, TryExtend, TryExtendFromSelf, TryPush};
use crate::bitmap::MutableBitmap;
use crate::datatypes::ArrowDataType;
use crate::offset::{Offset, Offsets};
use crate::trusted_len::TrustedLen;

/// A [`MutableArray`] that builds a [`Utf8Array`]. It differs
/// from [`MutableUtf8Array`] in that it builds non-null [`Utf8Array`].
#[derive(Debug, Clone)]
pub struct MutableUtf8ValuesArray<O: Offset> {
    data_type: ArrowDataType,
    offsets: Offsets<O>,
    values: Vec<u8>,
}

impl<O: Offset> From<MutableUtf8ValuesArray<O>> for Utf8Array<O> {
    fn from(other: MutableUtf8ValuesArray<O>) -> Self {
        // SAFETY:
        // `MutableUtf8ValuesArray` has the same invariants as `Utf8Array` and thus
        // `Utf8Array` can be safely created from `MutableUtf8ValuesArray` without checks.
        unsafe {
            Utf8Array::<O>::new_unchecked(
                other.data_type,
                other.offsets.into(),
                other.values.into(),
                None,
            )
        }
    }
}

impl<O: Offset> From<MutableUtf8ValuesArray<O>> for MutableUtf8Array<O> {
    fn from(other: MutableUtf8ValuesArray<O>) -> Self {
        // SAFETY:
        // `MutableUtf8ValuesArray` has the same invariants as `MutableUtf8Array`
        unsafe {
            MutableUtf8Array::<O>::new_unchecked(other.data_type, other.offsets, other.values, None)
        }
    }
}

impl<O: Offset> Default for MutableUtf8ValuesArray<O> {
    fn default() -> Self {
        Self::new()
    }
}

impl<O: Offset> MutableUtf8ValuesArray<O> {
    /// Returns an empty [`MutableUtf8ValuesArray`].
    pub fn new() -> Self {
        Self {
            data_type: Self::default_data_type(),
            offsets: Offsets::new(),
            values: Vec::<u8>::new(),
        }
    }

    /// Returns a [`MutableUtf8ValuesArray`] created from its internal representation.
    ///
    /// # Errors
    /// This function returns an error iff:
    /// * The last offset is not equal to the values' length.
    /// * The `data_type`'s [`crate::datatypes::PhysicalType`] is not equal to either `Utf8` or `LargeUtf8`.
    /// * The `values` between two consecutive `offsets` are not valid utf8
    /// # Implementation
    /// This function is `O(N)` - checking utf8 is `O(N)`
    pub fn try_new(
        data_type: ArrowDataType,
        offsets: Offsets<O>,
        values: Vec<u8>,
    ) -> PolarsResult<Self> {
        try_check_utf8(&offsets, &values)?;
        if data_type.to_physical_type() != Self::default_data_type().to_physical_type() {
            polars_bail!(ComputeError: "MutableUtf8ValuesArray can only be initialized with DataType::Utf8 or DataType::LargeUtf8")
        }

        Ok(Self {
            data_type,
            offsets,
            values,
        })
    }

    /// Returns a [`MutableUtf8ValuesArray`] created from its internal representation.
    ///
    /// # Panic
    /// This function does not panic iff:
    /// * The last offset is equal to the values' length.
    /// * The `data_type`'s [`crate::datatypes::PhysicalType`] is equal to either `Utf8` or `LargeUtf8`.
    ///
    /// # Safety
    /// This function is safe iff:
    /// * the offsets are monotonically increasing
    /// * The `values` between two consecutive `offsets` are not valid utf8
    /// # Implementation
    /// This function is `O(1)`
    pub unsafe fn new_unchecked(
        data_type: ArrowDataType,
        offsets: Offsets<O>,
        values: Vec<u8>,
    ) -> Self {
        try_check_offsets_bounds(&offsets, values.len())
            .expect("The length of the values must be equal to the last offset value");

        if data_type.to_physical_type() != Self::default_data_type().to_physical_type() {
            panic!("MutableUtf8ValuesArray can only be initialized with DataType::Utf8 or DataType::LargeUtf8")
        }

        Self {
            data_type,
            offsets,
            values,
        }
    }

    /// Returns the default [`ArrowDataType`] of this container: [`ArrowDataType::Utf8`] or [`ArrowDataType::LargeUtf8`]
    /// depending on the generic [`Offset`].
    pub fn default_data_type() -> ArrowDataType {
        Utf8Array::<O>::default_data_type()
    }

    /// Initializes a new [`MutableUtf8ValuesArray`] with a pre-allocated capacity of items.
    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacities(capacity, 0)
    }

    /// Initializes a new [`MutableUtf8ValuesArray`] with a pre-allocated capacity of items and values.
    pub fn with_capacities(capacity: usize, values: usize) -> Self {
        Self {
            data_type: Self::default_data_type(),
            offsets: Offsets::<O>::with_capacity(capacity),
            values: Vec::<u8>::with_capacity(values),
        }
    }

    /// returns its values.
    #[inline]
    pub fn values(&self) -> &Vec<u8> {
        &self.values
    }

    /// returns its offsets.
    #[inline]
    pub fn offsets(&self) -> &Offsets<O> {
        &self.offsets
    }

    /// Reserves `additional` elements and `additional_values` on the values.
    #[inline]
    pub fn reserve(&mut self, additional: usize, additional_values: usize) {
        self.offsets.reserve(additional + 1);
        self.values.reserve(additional_values);
    }

    /// Returns the capacity in number of items
    pub fn capacity(&self) -> usize {
        self.offsets.capacity()
    }

    /// Returns the length of this array
    #[inline]
    pub fn len(&self) -> usize {
        self.offsets.len_proxy()
    }

    /// Pushes a new item to the array.
    /// # Panic
    /// This operation panics iff the length of all values (in bytes) exceeds `O` maximum value.
    #[inline]
    pub fn push<T: AsRef<str>>(&mut self, value: T) {
        self.try_push(value).unwrap()
    }

    /// Pop the last entry from [`MutableUtf8ValuesArray`].
    /// This function returns `None` iff this array is empty.
    pub fn pop(&mut self) -> Option<String> {
        if self.len() == 0 {
            return None;
        }
        self.offsets.pop()?;
        let start = self.offsets.last().to_usize();
        let value = self.values.split_off(start);
        // SAFETY: utf8 is validated on initialization
        Some(unsafe { String::from_utf8_unchecked(value) })
    }

    /// Returns the value of the element at index `i`.
    /// # Panic
    /// This function panics iff `i >= self.len`.
    #[inline]
    pub fn value(&self, i: usize) -> &str {
        assert!(i < self.len());
        unsafe { self.value_unchecked(i) }
    }

    /// Returns the value of the element at index `i`.
    ///
    /// # Safety
    /// This function is safe iff `i < self.len`.
    #[inline]
    pub unsafe fn value_unchecked(&self, i: usize) -> &str {
        // soundness: the invariant of the function
        let (start, end) = self.offsets.start_end(i);

        // soundness: the invariant of the struct
        let slice = self.values.get_unchecked(start..end);

        // soundness: the invariant of the struct
        std::str::from_utf8_unchecked(slice)
    }

    /// Returns an iterator of `&str`
    pub fn iter(&self) -> ArrayValuesIter<Self> {
        ArrayValuesIter::new(self)
    }

    /// Shrinks the capacity of the [`MutableUtf8ValuesArray`] to fit its current length.
    pub fn shrink_to_fit(&mut self) {
        self.values.shrink_to_fit();
        self.offsets.shrink_to_fit();
    }

    /// Extract the low-end APIs from the [`MutableUtf8ValuesArray`].
    pub fn into_inner(self) -> (ArrowDataType, Offsets<O>, Vec<u8>) {
        (self.data_type, self.offsets, self.values)
    }
}

impl<O: Offset> MutableArray for MutableUtf8ValuesArray<O> {
    fn len(&self) -> usize {
        self.len()
    }

    fn validity(&self) -> Option<&MutableBitmap> {
        None
    }

    fn as_box(&mut self) -> Box<dyn Array> {
        let array: Utf8Array<O> = std::mem::take(self).into();
        array.boxed()
    }

    fn as_arc(&mut self) -> Arc<dyn Array> {
        let array: Utf8Array<O> = std::mem::take(self).into();
        array.arced()
    }

    fn data_type(&self) -> &ArrowDataType {
        &self.data_type
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_mut_any(&mut self) -> &mut dyn std::any::Any {
        self
    }

    #[inline]
    fn push_null(&mut self) {
        self.push::<&str>("")
    }

    fn reserve(&mut self, additional: usize) {
        self.reserve(additional, 0)
    }

    fn shrink_to_fit(&mut self) {
        self.shrink_to_fit()
    }
}

impl<O: Offset, P: AsRef<str>> FromIterator<P> for MutableUtf8ValuesArray<O> {
    fn from_iter<I: IntoIterator<Item = P>>(iter: I) -> Self {
        let (offsets, values) = values_iter(iter.into_iter().map(StrAsBytes));
        // soundness: T: AsRef<str> and offsets are monotonically increasing
        unsafe { Self::new_unchecked(Self::default_data_type(), offsets, values) }
    }
}

impl<O: Offset> MutableUtf8ValuesArray<O> {
    pub(crate) unsafe fn extend_from_trusted_len_iter<I, P>(
        &mut self,
        validity: &mut MutableBitmap,
        iterator: I,
    ) where
        P: AsRef<str>,
        I: Iterator<Item = Option<P>>,
    {
        let iterator = iterator.map(|x| x.map(StrAsBytes));
        extend_from_trusted_len_iter(&mut self.offsets, &mut self.values, validity, iterator);
    }

    /// Extends the [`MutableUtf8ValuesArray`] from a [`TrustedLen`]
    #[inline]
    pub fn extend_trusted_len<I, P>(&mut self, iterator: I)
    where
        P: AsRef<str>,
        I: TrustedLen<Item = P>,
    {
        unsafe { self.extend_trusted_len_unchecked(iterator) }
    }

    /// Extends [`MutableUtf8ValuesArray`] from an iterator of trusted len.
    ///
    /// # Safety
    /// The iterator must be trusted len.
    #[inline]
    pub unsafe fn extend_trusted_len_unchecked<I, P>(&mut self, iterator: I)
    where
        P: AsRef<str>,
        I: Iterator<Item = P>,
    {
        let iterator = iterator.map(StrAsBytes);
        extend_from_trusted_len_values_iter(&mut self.offsets, &mut self.values, iterator);
    }

    /// Creates a [`MutableUtf8ValuesArray`] from a [`TrustedLen`]
    #[inline]
    pub fn from_trusted_len_iter<I, P>(iterator: I) -> Self
    where
        P: AsRef<str>,
        I: TrustedLen<Item = P>,
    {
        // soundness: I is `TrustedLen`
        unsafe { Self::from_trusted_len_iter_unchecked(iterator) }
    }

    /// Returns a new [`MutableUtf8ValuesArray`] from an iterator of trusted length.
    ///
    /// # Safety
    /// The iterator must be [`TrustedLen`](https://doc.rust-lang.org/std/iter/trait.TrustedLen.html).
    /// I.e. that `size_hint().1` correctly reports its length.
    #[inline]
    pub unsafe fn from_trusted_len_iter_unchecked<I, P>(iterator: I) -> Self
    where
        P: AsRef<str>,
        I: Iterator<Item = P>,
    {
        let iterator = iterator.map(StrAsBytes);
        let (offsets, values) = trusted_len_values_iter(iterator);

        // soundness: P is `str` and offsets are monotonically increasing
        Self::new_unchecked(Self::default_data_type(), offsets, values)
    }

    /// Returns a new [`MutableUtf8ValuesArray`] from an iterator.
    /// # Error
    /// This operation errors iff the total length in bytes on the iterator exceeds `O`'s maximum value.
    /// (`i32::MAX` or `i64::MAX` respectively).
    pub fn try_from_iter<P: AsRef<str>, I: IntoIterator<Item = P>>(iter: I) -> PolarsResult<Self> {
        let iterator = iter.into_iter();
        let (lower, _) = iterator.size_hint();
        let mut array = Self::with_capacity(lower);
        for item in iterator {
            array.try_push(item)?;
        }
        Ok(array)
    }

    /// Extend with a fallible iterator
    pub fn extend_fallible<T, I, E>(&mut self, iter: I) -> std::result::Result<(), E>
    where
        E: std::error::Error,
        I: IntoIterator<Item = std::result::Result<T, E>>,
        T: AsRef<str>,
    {
        let mut iter = iter.into_iter();
        self.reserve(iter.size_hint().0, 0);
        iter.try_for_each(|x| {
            self.push(x?);
            Ok(())
        })
    }
}

impl<O: Offset, T: AsRef<str>> Extend<T> for MutableUtf8ValuesArray<O> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        extend_from_values_iter(
            &mut self.offsets,
            &mut self.values,
            iter.into_iter().map(StrAsBytes),
        );
    }
}

impl<O: Offset, T: AsRef<str>> TryExtend<T> for MutableUtf8ValuesArray<O> {
    fn try_extend<I: IntoIterator<Item = T>>(&mut self, iter: I) -> PolarsResult<()> {
        let mut iter = iter.into_iter();
        self.reserve(iter.size_hint().0, 0);
        iter.try_for_each(|x| self.try_push(x))
    }
}

impl<O: Offset, T: AsRef<str>> TryPush<T> for MutableUtf8ValuesArray<O> {
    #[inline]
    fn try_push(&mut self, value: T) -> PolarsResult<()> {
        let bytes = value.as_ref().as_bytes();
        self.values.extend_from_slice(bytes);
        self.offsets.try_push(bytes.len())
    }
}

impl<O: Offset> TryExtendFromSelf for MutableUtf8ValuesArray<O> {
    fn try_extend_from_self(&mut self, other: &Self) -> PolarsResult<()> {
        self.values.extend_from_slice(&other.values);
        self.offsets.try_extend_from_self(&other.offsets)
    }
}
