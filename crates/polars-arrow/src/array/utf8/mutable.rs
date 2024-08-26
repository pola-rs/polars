use std::sync::Arc;

use polars_error::{polars_bail, PolarsResult};

use super::{MutableUtf8ValuesArray, MutableUtf8ValuesIter, StrAsBytes, Utf8Array};
use crate::array::physical_binary::*;
use crate::array::{Array, MutableArray, TryExtend, TryExtendFromSelf, TryPush};
use crate::bitmap::utils::{BitmapIter, ZipValidity};
use crate::bitmap::{Bitmap, MutableBitmap};
use crate::datatypes::ArrowDataType;
use crate::offset::{Offset, Offsets};
use crate::trusted_len::TrustedLen;

/// A [`MutableArray`] that builds a [`Utf8Array`]. It differs
/// from [`MutableUtf8ValuesArray`] in that it can build nullable [`Utf8Array`]s.
#[derive(Debug, Clone)]
pub struct MutableUtf8Array<O: Offset> {
    values: MutableUtf8ValuesArray<O>,
    validity: Option<MutableBitmap>,
}

impl<O: Offset> From<MutableUtf8Array<O>> for Utf8Array<O> {
    fn from(other: MutableUtf8Array<O>) -> Self {
        let validity = other.validity.and_then(|x| {
            let validity: Option<Bitmap> = x.into();
            validity
        });
        let array: Utf8Array<O> = other.values.into();
        array.with_validity(validity)
    }
}

impl<O: Offset> Default for MutableUtf8Array<O> {
    fn default() -> Self {
        Self::new()
    }
}

impl<O: Offset> MutableUtf8Array<O> {
    /// Initializes a new empty [`MutableUtf8Array`].
    pub fn new() -> Self {
        Self {
            values: Default::default(),
            validity: None,
        }
    }

    /// Returns a [`MutableUtf8Array`] created from its internal representation.
    ///
    /// # Errors
    /// This function returns an error iff:
    /// * The last offset is not equal to the values' length.
    /// * the validity's length is not equal to `offsets.len()`.
    /// * The `data_type`'s [`crate::datatypes::PhysicalType`] is not equal to either `Utf8` or `LargeUtf8`.
    /// * The `values` between two consecutive `offsets` are not valid utf8
    /// # Implementation
    /// This function is `O(N)` - checking utf8 is `O(N)`
    pub fn try_new(
        data_type: ArrowDataType,
        offsets: Offsets<O>,
        values: Vec<u8>,
        validity: Option<MutableBitmap>,
    ) -> PolarsResult<Self> {
        let values = MutableUtf8ValuesArray::try_new(data_type, offsets, values)?;

        if validity
            .as_ref()
            .map_or(false, |validity| validity.len() != values.len())
        {
            polars_bail!(ComputeError: "validity's length must be equal to the number of values")
        }

        Ok(Self { values, validity })
    }

    /// Create a [`MutableUtf8Array`] out of low-end APIs.
    ///
    /// # Safety
    /// The caller must ensure that every value between offsets is a valid utf8.
    /// # Panics
    /// This function panics iff:
    /// * The `offsets` and `values` are inconsistent
    /// * The validity is not `None` and its length is different from `offsets`'s length minus one.
    pub unsafe fn new_unchecked(
        data_type: ArrowDataType,
        offsets: Offsets<O>,
        values: Vec<u8>,
        validity: Option<MutableBitmap>,
    ) -> Self {
        let values = MutableUtf8ValuesArray::new_unchecked(data_type, offsets, values);
        if let Some(ref validity) = validity {
            assert_eq!(values.len(), validity.len());
        }
        Self { values, validity }
    }

    /// Creates a new [`MutableUtf8Array`] from a slice of optional `&[u8]`.
    // Note: this can't be `impl From` because Rust does not allow double `AsRef` on it.
    pub fn from<T: AsRef<str>, P: AsRef<[Option<T>]>>(slice: P) -> Self {
        Self::from_trusted_len_iter(slice.as_ref().iter().map(|x| x.as_ref()))
    }

    fn default_data_type() -> ArrowDataType {
        Utf8Array::<O>::default_data_type()
    }

    /// Initializes a new [`MutableUtf8Array`] with a pre-allocated capacity of slots.
    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacities(capacity, 0)
    }

    /// Initializes a new [`MutableUtf8Array`] with a pre-allocated capacity of slots and values.
    pub fn with_capacities(capacity: usize, values: usize) -> Self {
        Self {
            values: MutableUtf8ValuesArray::with_capacities(capacity, values),
            validity: None,
        }
    }

    /// Reserves `additional` elements and `additional_values` on the values buffer.
    pub fn reserve(&mut self, additional: usize, additional_values: usize) {
        self.values.reserve(additional, additional_values);
        if let Some(x) = self.validity.as_mut() {
            x.reserve(additional)
        }
    }

    /// Reserves `additional` elements and `additional_values` on the values buffer.
    pub fn capacity(&self) -> usize {
        self.values.capacity()
    }

    /// Returns the length of this array
    #[inline]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Pushes a new element to the array.
    /// # Panic
    /// This operation panics iff the length of all values (in bytes) exceeds `O` maximum value.
    #[inline]
    pub fn push<T: AsRef<str>>(&mut self, value: Option<T>) {
        self.try_push(value).unwrap()
    }

    /// Returns the value of the element at index `i`, ignoring the array's validity.
    #[inline]
    pub fn value(&self, i: usize) -> &str {
        self.values.value(i)
    }

    /// Returns the value of the element at index `i`, ignoring the array's validity.
    ///
    /// # Safety
    /// This function is safe iff `i < self.len`.
    #[inline]
    pub unsafe fn value_unchecked(&self, i: usize) -> &str {
        self.values.value_unchecked(i)
    }

    /// Pop the last entry from [`MutableUtf8Array`].
    /// This function returns `None` iff this array is empty.
    pub fn pop(&mut self) -> Option<String> {
        let value = self.values.pop()?;
        self.validity
            .as_mut()
            .map(|x| x.pop()?.then(|| ()))
            .unwrap_or_else(|| Some(()))
            .map(|_| value)
    }

    fn init_validity(&mut self) {
        let mut validity = MutableBitmap::with_capacity(self.values.capacity());
        validity.extend_constant(self.len(), true);
        validity.set(self.len() - 1, false);
        self.validity = Some(validity);
    }

    /// Returns an iterator of `Option<&str>`
    pub fn iter(&self) -> ZipValidity<&str, MutableUtf8ValuesIter<O>, BitmapIter> {
        ZipValidity::new(self.values_iter(), self.validity.as_ref().map(|x| x.iter()))
    }

    /// Converts itself into an [`Array`].
    pub fn into_arc(self) -> Arc<dyn Array> {
        let a: Utf8Array<O> = self.into();
        Arc::new(a)
    }

    /// Shrinks the capacity of the [`MutableUtf8Array`] to fit its current length.
    pub fn shrink_to_fit(&mut self) {
        self.values.shrink_to_fit();
        if let Some(validity) = &mut self.validity {
            validity.shrink_to_fit()
        }
    }

    /// Extract the low-end APIs from the [`MutableUtf8Array`].
    pub fn into_data(self) -> (ArrowDataType, Offsets<O>, Vec<u8>, Option<MutableBitmap>) {
        let (data_type, offsets, values) = self.values.into_inner();
        (data_type, offsets, values, self.validity)
    }

    /// Returns an iterator of `&str`
    pub fn values_iter(&self) -> MutableUtf8ValuesIter<O> {
        self.values.iter()
    }

    /// Sets the validity.
    /// # Panic
    /// Panics iff the validity's len is not equal to the existing values' length.
    pub fn set_validity(&mut self, validity: Option<MutableBitmap>) {
        if let Some(validity) = &validity {
            assert_eq!(self.values.len(), validity.len())
        }
        self.validity = validity;
    }

    /// Applies a function `f` to the validity of this array.
    ///
    /// This is an API to leverage clone-on-write
    /// # Panics
    /// This function panics if the function `f` modifies the length of the [`Bitmap`].
    pub fn apply_validity<F: FnOnce(MutableBitmap) -> MutableBitmap>(&mut self, f: F) {
        if let Some(validity) = std::mem::take(&mut self.validity) {
            self.set_validity(Some(f(validity)))
        }
    }
}

impl<O: Offset> MutableUtf8Array<O> {
    /// returns its values.
    pub fn values(&self) -> &Vec<u8> {
        self.values.values()
    }

    /// returns its offsets.
    pub fn offsets(&self) -> &Offsets<O> {
        self.values.offsets()
    }
}

impl<O: Offset> MutableArray for MutableUtf8Array<O> {
    fn len(&self) -> usize {
        self.len()
    }

    fn validity(&self) -> Option<&MutableBitmap> {
        self.validity.as_ref()
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
        if O::IS_LARGE {
            &ArrowDataType::LargeUtf8
        } else {
            &ArrowDataType::Utf8
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_mut_any(&mut self) -> &mut dyn std::any::Any {
        self
    }

    #[inline]
    fn push_null(&mut self) {
        self.push::<&str>(None)
    }

    fn reserve(&mut self, additional: usize) {
        self.reserve(additional, 0)
    }

    fn shrink_to_fit(&mut self) {
        self.shrink_to_fit()
    }
}

impl<O: Offset, P: AsRef<str>> FromIterator<Option<P>> for MutableUtf8Array<O> {
    fn from_iter<I: IntoIterator<Item = Option<P>>>(iter: I) -> Self {
        Self::try_from_iter(iter).unwrap()
    }
}

impl<O: Offset> MutableUtf8Array<O> {
    /// Extends the [`MutableUtf8Array`] from an iterator of values of trusted len.
    /// This differs from `extended_trusted_len` which accepts iterator of optional values.
    #[inline]
    pub fn extend_trusted_len_values<I, P>(&mut self, iterator: I)
    where
        P: AsRef<str>,
        I: TrustedLen<Item = P>,
    {
        unsafe { self.extend_trusted_len_values_unchecked(iterator) }
    }

    /// Extends the [`MutableUtf8Array`] from an iterator of values.
    /// This differs from `extended_trusted_len` which accepts iterator of optional values.
    #[inline]
    pub fn extend_values<I, P>(&mut self, iterator: I)
    where
        P: AsRef<str>,
        I: Iterator<Item = P>,
    {
        let length = self.values.len();
        self.values.extend(iterator);
        let additional = self.values.len() - length;

        if let Some(validity) = self.validity.as_mut() {
            validity.extend_constant(additional, true);
        }
    }

    /// Extends the [`MutableUtf8Array`] from an iterator of values of trusted len.
    /// This differs from `extended_trusted_len_unchecked` which accepts iterator of optional
    /// values.
    ///
    /// # Safety
    /// The iterator must be trusted len.
    #[inline]
    pub unsafe fn extend_trusted_len_values_unchecked<I, P>(&mut self, iterator: I)
    where
        P: AsRef<str>,
        I: Iterator<Item = P>,
    {
        let length = self.values.len();
        self.values.extend_trusted_len_unchecked(iterator);
        let additional = self.values.len() - length;

        if let Some(validity) = self.validity.as_mut() {
            validity.extend_constant(additional, true);
        }
    }

    /// Extends the [`MutableUtf8Array`] from an iterator of trusted len.
    #[inline]
    pub fn extend_trusted_len<I, P>(&mut self, iterator: I)
    where
        P: AsRef<str>,
        I: TrustedLen<Item = Option<P>>,
    {
        unsafe { self.extend_trusted_len_unchecked(iterator) }
    }

    /// Extends [`MutableUtf8Array`] from an iterator of trusted len.
    ///
    /// # Safety
    /// The iterator must be trusted len.
    #[inline]
    pub unsafe fn extend_trusted_len_unchecked<I, P>(&mut self, iterator: I)
    where
        P: AsRef<str>,
        I: Iterator<Item = Option<P>>,
    {
        if self.validity.is_none() {
            let mut validity = MutableBitmap::new();
            validity.extend_constant(self.len(), true);
            self.validity = Some(validity);
        }

        self.values
            .extend_from_trusted_len_iter(self.validity.as_mut().unwrap(), iterator);
    }

    /// Creates a [`MutableUtf8Array`] from an iterator of trusted length.
    ///
    /// # Safety
    /// The iterator must be [`TrustedLen`](https://doc.rust-lang.org/std/iter/trait.TrustedLen.html).
    /// I.e. that `size_hint().1` correctly reports its length.
    #[inline]
    pub unsafe fn from_trusted_len_iter_unchecked<I, P>(iterator: I) -> Self
    where
        P: AsRef<str>,
        I: Iterator<Item = Option<P>>,
    {
        let iterator = iterator.map(|x| x.map(StrAsBytes));
        let (validity, offsets, values) = trusted_len_unzip(iterator);

        // soundness: P is `str`
        Self::new_unchecked(Self::default_data_type(), offsets, values, validity)
    }

    /// Creates a [`MutableUtf8Array`] from an iterator of trusted length.
    #[inline]
    pub fn from_trusted_len_iter<I, P>(iterator: I) -> Self
    where
        P: AsRef<str>,
        I: TrustedLen<Item = Option<P>>,
    {
        // soundness: I is `TrustedLen`
        unsafe { Self::from_trusted_len_iter_unchecked(iterator) }
    }

    /// Creates a [`MutableUtf8Array`] from an iterator of trusted length of `&str`.
    ///
    /// # Safety
    /// The iterator must be [`TrustedLen`](https://doc.rust-lang.org/std/iter/trait.TrustedLen.html).
    /// I.e. that `size_hint().1` correctly reports its length.
    #[inline]
    pub unsafe fn from_trusted_len_values_iter_unchecked<T: AsRef<str>, I: Iterator<Item = T>>(
        iterator: I,
    ) -> Self {
        MutableUtf8ValuesArray::from_trusted_len_iter_unchecked(iterator).into()
    }

    /// Creates a new [`MutableUtf8Array`] from a [`TrustedLen`] of `&str`.
    #[inline]
    pub fn from_trusted_len_values_iter<T: AsRef<str>, I: TrustedLen<Item = T>>(
        iterator: I,
    ) -> Self {
        // soundness: I is `TrustedLen`
        unsafe { Self::from_trusted_len_values_iter_unchecked(iterator) }
    }

    /// Creates a new [`MutableUtf8Array`] from an iterator.
    /// # Error
    /// This operation errors iff the total length in bytes on the iterator exceeds `O`'s maximum value.
    /// (`i32::MAX` or `i64::MAX` respectively).
    fn try_from_iter<P: AsRef<str>, I: IntoIterator<Item = Option<P>>>(
        iter: I,
    ) -> PolarsResult<Self> {
        let iterator = iter.into_iter();
        let (lower, _) = iterator.size_hint();
        let mut array = Self::with_capacity(lower);
        for item in iterator {
            array.try_push(item)?;
        }
        Ok(array)
    }

    /// Creates a [`MutableUtf8Array`] from an falible iterator of trusted length.
    ///
    /// # Safety
    /// The iterator must be [`TrustedLen`](https://doc.rust-lang.org/std/iter/trait.TrustedLen.html).
    /// I.e. that `size_hint().1` correctly reports its length.
    #[inline]
    pub unsafe fn try_from_trusted_len_iter_unchecked<E, I, P>(
        iterator: I,
    ) -> std::result::Result<Self, E>
    where
        P: AsRef<str>,
        I: IntoIterator<Item = std::result::Result<Option<P>, E>>,
    {
        let iterator = iterator.into_iter();

        let iterator = iterator.map(|x| x.map(|x| x.map(StrAsBytes)));
        let (validity, offsets, values) = try_trusted_len_unzip(iterator)?;

        // soundness: P is `str`
        Ok(Self::new_unchecked(
            Self::default_data_type(),
            offsets,
            values,
            validity,
        ))
    }

    /// Creates a [`MutableUtf8Array`] from an falible iterator of trusted length.
    #[inline]
    pub fn try_from_trusted_len_iter<E, I, P>(iterator: I) -> std::result::Result<Self, E>
    where
        P: AsRef<str>,
        I: TrustedLen<Item = std::result::Result<Option<P>, E>>,
    {
        // soundness: I: TrustedLen
        unsafe { Self::try_from_trusted_len_iter_unchecked(iterator) }
    }

    /// Creates a new [`MutableUtf8Array`] from a [`Iterator`] of `&str`.
    pub fn from_iter_values<T: AsRef<str>, I: Iterator<Item = T>>(iterator: I) -> Self {
        MutableUtf8ValuesArray::from_iter(iterator).into()
    }

    /// Extend with a fallible iterator
    pub fn extend_fallible<T, I, E>(&mut self, iter: I) -> std::result::Result<(), E>
    where
        E: std::error::Error,
        I: IntoIterator<Item = std::result::Result<Option<T>, E>>,
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

impl<O: Offset, T: AsRef<str>> Extend<Option<T>> for MutableUtf8Array<O> {
    fn extend<I: IntoIterator<Item = Option<T>>>(&mut self, iter: I) {
        self.try_extend(iter).unwrap();
    }
}

impl<O: Offset, T: AsRef<str>> TryExtend<Option<T>> for MutableUtf8Array<O> {
    fn try_extend<I: IntoIterator<Item = Option<T>>>(&mut self, iter: I) -> PolarsResult<()> {
        let mut iter = iter.into_iter();
        self.reserve(iter.size_hint().0, 0);
        iter.try_for_each(|x| self.try_push(x))
    }
}

impl<O: Offset, T: AsRef<str>> TryPush<Option<T>> for MutableUtf8Array<O> {
    #[inline]
    fn try_push(&mut self, value: Option<T>) -> PolarsResult<()> {
        match value {
            Some(value) => {
                self.values.try_push(value.as_ref())?;

                if let Some(validity) = &mut self.validity {
                    validity.push(true)
                }
            },
            None => {
                self.values.push("");
                match &mut self.validity {
                    Some(validity) => validity.push(false),
                    None => self.init_validity(),
                }
            },
        }
        Ok(())
    }
}

impl<O: Offset> PartialEq for MutableUtf8Array<O> {
    fn eq(&self, other: &Self) -> bool {
        self.iter().eq(other.iter())
    }
}

impl<O: Offset> TryExtendFromSelf for MutableUtf8Array<O> {
    fn try_extend_from_self(&mut self, other: &Self) -> PolarsResult<()> {
        extend_validity(self.len(), &mut self.validity, &other.validity);

        self.values.try_extend_from_self(&other.values)
    }
}
