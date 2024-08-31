use std::sync::Arc;

use polars_error::{polars_bail, PolarsResult};

use super::{BinaryArray, MutableBinaryValuesArray, MutableBinaryValuesIter};
use crate::array::physical_binary::*;
use crate::array::{Array, MutableArray, TryExtend, TryExtendFromSelf, TryPush};
use crate::bitmap::utils::{BitmapIter, ZipValidity};
use crate::bitmap::{Bitmap, MutableBitmap};
use crate::datatypes::ArrowDataType;
use crate::offset::{Offset, Offsets};
use crate::trusted_len::TrustedLen;

/// The Arrow's equivalent to `Vec<Option<Vec<u8>>>`.
/// Converting a [`MutableBinaryArray`] into a [`BinaryArray`] is `O(1)`.
/// # Implementation
/// This struct does not allocate a validity until one is required (i.e. push a null to it).
#[derive(Debug, Clone)]
pub struct MutableBinaryArray<O: Offset> {
    values: MutableBinaryValuesArray<O>,
    validity: Option<MutableBitmap>,
}

impl<O: Offset> From<MutableBinaryArray<O>> for BinaryArray<O> {
    fn from(other: MutableBinaryArray<O>) -> Self {
        let validity = other.validity.and_then(|x| {
            let validity: Option<Bitmap> = x.into();
            validity
        });
        let array: BinaryArray<O> = other.values.into();
        array.with_validity(validity)
    }
}

impl<O: Offset> Default for MutableBinaryArray<O> {
    fn default() -> Self {
        Self::new()
    }
}

impl<O: Offset> MutableBinaryArray<O> {
    /// Creates a new empty [`MutableBinaryArray`].
    /// # Implementation
    /// This allocates a [`Vec`] of one element
    pub fn new() -> Self {
        Self::with_capacity(0)
    }

    /// Returns a [`MutableBinaryArray`] created from its internal representation.
    ///
    /// # Errors
    /// This function returns an error iff:
    /// * The last offset is not equal to the values' length.
    /// * the validity's length is not equal to `offsets.len()`.
    /// * The `data_type`'s [`crate::datatypes::PhysicalType`] is not equal to either `Binary` or `LargeBinary`.
    /// # Implementation
    /// This function is `O(1)`
    pub fn try_new(
        data_type: ArrowDataType,
        offsets: Offsets<O>,
        values: Vec<u8>,
        validity: Option<MutableBitmap>,
    ) -> PolarsResult<Self> {
        let values = MutableBinaryValuesArray::try_new(data_type, offsets, values)?;

        if validity
            .as_ref()
            .map_or(false, |validity| validity.len() != values.len())
        {
            polars_bail!(ComputeError: "validity's length must be equal to the number of values")
        }

        Ok(Self { values, validity })
    }

    /// Creates a new [`MutableBinaryArray`] from a slice of optional `&[u8]`.
    // Note: this can't be `impl From` because Rust does not allow double `AsRef` on it.
    pub fn from<T: AsRef<[u8]>, P: AsRef<[Option<T>]>>(slice: P) -> Self {
        Self::from_trusted_len_iter(slice.as_ref().iter().map(|x| x.as_ref()))
    }

    fn default_data_type() -> ArrowDataType {
        BinaryArray::<O>::default_data_type()
    }

    /// Initializes a new [`MutableBinaryArray`] with a pre-allocated capacity of slots.
    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacities(capacity, 0)
    }

    /// Initializes a new [`MutableBinaryArray`] with a pre-allocated capacity of slots and values.
    /// # Implementation
    /// This does not allocate the validity.
    pub fn with_capacities(capacity: usize, values: usize) -> Self {
        Self {
            values: MutableBinaryValuesArray::with_capacities(capacity, values),
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

    /// Pushes a new element to the array.
    /// # Panic
    /// This operation panics iff the length of all values (in bytes) exceeds `O` maximum value.
    pub fn push<T: AsRef<[u8]>>(&mut self, value: Option<T>) {
        self.try_push(value).unwrap()
    }

    /// Pop the last entry from [`MutableBinaryArray`].
    /// This function returns `None` iff this array is empty
    pub fn pop(&mut self) -> Option<Vec<u8>> {
        let value = self.values.pop()?;
        self.validity
            .as_mut()
            .map(|x| x.pop()?.then(|| ()))
            .unwrap_or_else(|| Some(()))
            .map(|_| value)
    }

    fn try_from_iter<P: AsRef<[u8]>, I: IntoIterator<Item = Option<P>>>(
        iter: I,
    ) -> PolarsResult<Self> {
        let iterator = iter.into_iter();
        let (lower, _) = iterator.size_hint();
        let mut primitive = Self::with_capacity(lower);
        for item in iterator {
            primitive.try_push(item.as_ref())?
        }
        Ok(primitive)
    }

    fn init_validity(&mut self) {
        let mut validity = MutableBitmap::with_capacity(self.values.capacity());
        validity.extend_constant(self.len(), true);
        validity.set(self.len() - 1, false);
        self.validity = Some(validity);
    }

    /// Converts itself into an [`Array`].
    pub fn into_arc(self) -> Arc<dyn Array> {
        let a: BinaryArray<O> = self.into();
        Arc::new(a)
    }

    /// Shrinks the capacity of the [`MutableBinaryArray`] to fit its current length.
    pub fn shrink_to_fit(&mut self) {
        self.values.shrink_to_fit();
        if let Some(validity) = &mut self.validity {
            validity.shrink_to_fit()
        }
    }

    impl_mutable_array_mut_validity!();
}

impl<O: Offset> MutableBinaryArray<O> {
    /// returns its values.
    pub fn values(&self) -> &Vec<u8> {
        self.values.values()
    }

    /// returns its offsets.
    pub fn offsets(&self) -> &Offsets<O> {
        self.values.offsets()
    }

    /// Returns an iterator of `Option<&[u8]>`
    pub fn iter(&self) -> ZipValidity<&[u8], MutableBinaryValuesIter<O>, BitmapIter> {
        ZipValidity::new(self.values_iter(), self.validity.as_ref().map(|x| x.iter()))
    }

    /// Returns an iterator over the values of this array
    pub fn values_iter(&self) -> MutableBinaryValuesIter<O> {
        self.values.iter()
    }
}

impl<O: Offset> MutableArray for MutableBinaryArray<O> {
    fn len(&self) -> usize {
        self.values.len()
    }

    fn validity(&self) -> Option<&MutableBitmap> {
        self.validity.as_ref()
    }

    fn as_box(&mut self) -> Box<dyn Array> {
        let array: BinaryArray<O> = std::mem::take(self).into();
        array.boxed()
    }

    fn as_arc(&mut self) -> Arc<dyn Array> {
        let array: BinaryArray<O> = std::mem::take(self).into();
        array.arced()
    }

    fn data_type(&self) -> &ArrowDataType {
        self.values.data_type()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_mut_any(&mut self) -> &mut dyn std::any::Any {
        self
    }

    #[inline]
    fn push_null(&mut self) {
        self.push::<&[u8]>(None)
    }

    fn reserve(&mut self, additional: usize) {
        self.reserve(additional, 0)
    }

    fn shrink_to_fit(&mut self) {
        self.shrink_to_fit()
    }
}

impl<O: Offset, P: AsRef<[u8]>> FromIterator<Option<P>> for MutableBinaryArray<O> {
    fn from_iter<I: IntoIterator<Item = Option<P>>>(iter: I) -> Self {
        Self::try_from_iter(iter).unwrap()
    }
}

impl<O: Offset> MutableBinaryArray<O> {
    /// Creates a [`MutableBinaryArray`] from an iterator of trusted length.
    ///
    /// # Safety
    /// The iterator must be [`TrustedLen`](https://doc.rust-lang.org/std/iter/trait.TrustedLen.html).
    /// I.e. that `size_hint().1` correctly reports its length.
    #[inline]
    pub unsafe fn from_trusted_len_iter_unchecked<I, P>(iterator: I) -> Self
    where
        P: AsRef<[u8]>,
        I: Iterator<Item = Option<P>>,
    {
        let (validity, offsets, values) = trusted_len_unzip(iterator);

        Self::try_new(Self::default_data_type(), offsets, values, validity).unwrap()
    }

    /// Creates a [`MutableBinaryArray`] from an iterator of trusted length.
    #[inline]
    pub fn from_trusted_len_iter<I, P>(iterator: I) -> Self
    where
        P: AsRef<[u8]>,
        I: TrustedLen<Item = Option<P>>,
    {
        // soundness: I is `TrustedLen`
        unsafe { Self::from_trusted_len_iter_unchecked(iterator) }
    }

    /// Creates a new [`BinaryArray`] from a [`TrustedLen`] of `&[u8]`.
    ///
    /// # Safety
    /// The iterator must be [`TrustedLen`](https://doc.rust-lang.org/std/iter/trait.TrustedLen.html).
    /// I.e. that `size_hint().1` correctly reports its length.
    #[inline]
    pub unsafe fn from_trusted_len_values_iter_unchecked<T: AsRef<[u8]>, I: Iterator<Item = T>>(
        iterator: I,
    ) -> Self {
        let (offsets, values) = trusted_len_values_iter(iterator);
        Self::try_new(Self::default_data_type(), offsets, values, None).unwrap()
    }

    /// Creates a new [`BinaryArray`] from a [`TrustedLen`] of `&[u8]`.
    #[inline]
    pub fn from_trusted_len_values_iter<T: AsRef<[u8]>, I: TrustedLen<Item = T>>(
        iterator: I,
    ) -> Self {
        // soundness: I is `TrustedLen`
        unsafe { Self::from_trusted_len_values_iter_unchecked(iterator) }
    }

    /// Creates a [`MutableBinaryArray`] from an falible iterator of trusted length.
    ///
    /// # Safety
    /// The iterator must be [`TrustedLen`](https://doc.rust-lang.org/std/iter/trait.TrustedLen.html).
    /// I.e. that `size_hint().1` correctly reports its length.
    #[inline]
    pub unsafe fn try_from_trusted_len_iter_unchecked<E, I, P>(
        iterator: I,
    ) -> std::result::Result<Self, E>
    where
        P: AsRef<[u8]>,
        I: IntoIterator<Item = std::result::Result<Option<P>, E>>,
    {
        let iterator = iterator.into_iter();

        // soundness: assumed trusted len
        let (mut validity, offsets, values) = try_trusted_len_unzip(iterator)?;

        if validity.as_mut().unwrap().unset_bits() == 0 {
            validity = None;
        }

        Ok(Self::try_new(Self::default_data_type(), offsets, values, validity).unwrap())
    }

    /// Creates a [`MutableBinaryArray`] from an falible iterator of trusted length.
    #[inline]
    pub fn try_from_trusted_len_iter<E, I, P>(iterator: I) -> std::result::Result<Self, E>
    where
        P: AsRef<[u8]>,
        I: TrustedLen<Item = std::result::Result<Option<P>, E>>,
    {
        // soundness: I: TrustedLen
        unsafe { Self::try_from_trusted_len_iter_unchecked(iterator) }
    }

    /// Extends the [`MutableBinaryArray`] from an iterator of trusted length.
    /// This differs from `extend_trusted_len` which accepts iterator of optional values.
    #[inline]
    pub fn extend_trusted_len_values<I, P>(&mut self, iterator: I)
    where
        P: AsRef<[u8]>,
        I: TrustedLen<Item = P>,
    {
        // SAFETY: The iterator is `TrustedLen`
        unsafe { self.extend_trusted_len_values_unchecked(iterator) }
    }

    /// Extends the [`MutableBinaryArray`] from an iterator of values.
    /// This differs from `extended_trusted_len` which accepts iterator of optional values.
    #[inline]
    pub fn extend_values<I, P>(&mut self, iterator: I)
    where
        P: AsRef<[u8]>,
        I: Iterator<Item = P>,
    {
        let length = self.values.len();
        self.values.extend(iterator);
        let additional = self.values.len() - length;

        if let Some(validity) = self.validity.as_mut() {
            validity.extend_constant(additional, true);
        }
    }

    /// Extends the [`MutableBinaryArray`] from an `iterator` of values of trusted length.
    /// This differs from `extend_trusted_len_unchecked` which accepts iterator of optional
    /// values.
    ///
    /// # Safety
    /// The `iterator` must be [`TrustedLen`]
    #[inline]
    pub unsafe fn extend_trusted_len_values_unchecked<I, P>(&mut self, iterator: I)
    where
        P: AsRef<[u8]>,
        I: Iterator<Item = P>,
    {
        let length = self.values.len();
        self.values.extend_trusted_len_unchecked(iterator);
        let additional = self.values.len() - length;

        if let Some(validity) = self.validity.as_mut() {
            validity.extend_constant(additional, true);
        }
    }

    /// Extends the [`MutableBinaryArray`] from an iterator of [`TrustedLen`]
    #[inline]
    pub fn extend_trusted_len<I, P>(&mut self, iterator: I)
    where
        P: AsRef<[u8]>,
        I: TrustedLen<Item = Option<P>>,
    {
        // SAFETY: The iterator is `TrustedLen`
        unsafe { self.extend_trusted_len_unchecked(iterator) }
    }

    /// Extends the [`MutableBinaryArray`] from an iterator of [`TrustedLen`]
    ///
    /// # Safety
    /// The `iterator` must be [`TrustedLen`]
    #[inline]
    pub unsafe fn extend_trusted_len_unchecked<I, P>(&mut self, iterator: I)
    where
        P: AsRef<[u8]>,
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

    /// Creates a new [`MutableBinaryArray`] from a [`Iterator`] of `&[u8]`.
    pub fn from_iter_values<T: AsRef<[u8]>, I: Iterator<Item = T>>(iterator: I) -> Self {
        let (offsets, values) = values_iter(iterator);
        Self::try_new(Self::default_data_type(), offsets, values, None).unwrap()
    }

    /// Extend with a fallible iterator
    pub fn extend_fallible<T, I, E>(&mut self, iter: I) -> std::result::Result<(), E>
    where
        E: std::error::Error,
        I: IntoIterator<Item = std::result::Result<Option<T>, E>>,
        T: AsRef<[u8]>,
    {
        let mut iter = iter.into_iter();
        self.reserve(iter.size_hint().0, 0);
        iter.try_for_each(|x| {
            self.push(x?);
            Ok(())
        })
    }
}

impl<O: Offset, T: AsRef<[u8]>> Extend<Option<T>> for MutableBinaryArray<O> {
    fn extend<I: IntoIterator<Item = Option<T>>>(&mut self, iter: I) {
        self.try_extend(iter).unwrap();
    }
}

impl<O: Offset, T: AsRef<[u8]>> TryExtend<Option<T>> for MutableBinaryArray<O> {
    fn try_extend<I: IntoIterator<Item = Option<T>>>(&mut self, iter: I) -> PolarsResult<()> {
        let mut iter = iter.into_iter();
        self.reserve(iter.size_hint().0, 0);
        iter.try_for_each(|x| self.try_push(x))
    }
}

impl<O: Offset, T: AsRef<[u8]>> TryPush<Option<T>> for MutableBinaryArray<O> {
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

impl<O: Offset> PartialEq for MutableBinaryArray<O> {
    fn eq(&self, other: &Self) -> bool {
        self.iter().eq(other.iter())
    }
}

impl<O: Offset> TryExtendFromSelf for MutableBinaryArray<O> {
    fn try_extend_from_self(&mut self, other: &Self) -> PolarsResult<()> {
        extend_validity(self.len(), &mut self.validity, &other.validity);

        self.values.try_extend_from_self(&other.values)
    }
}
