use std::sync::Arc;

use polars_error::PolarsResult;

use super::{check, PrimitiveArray};
use crate::array::physical_binary::extend_validity;
use crate::array::{Array, MutableArray, TryExtend, TryExtendFromSelf, TryPush};
use crate::bitmap::{Bitmap, MutableBitmap};
use crate::datatypes::ArrowDataType;
use crate::trusted_len::TrustedLen;
use crate::types::NativeType;

/// The Arrow's equivalent to `Vec<Option<T>>` where `T` is byte-size (e.g. `i32`).
/// Converting a [`MutablePrimitiveArray`] into a [`PrimitiveArray`] is `O(1)`.
#[derive(Debug, Clone)]
pub struct MutablePrimitiveArray<T: NativeType> {
    data_type: ArrowDataType,
    values: Vec<T>,
    validity: Option<MutableBitmap>,
}

impl<T: NativeType> From<MutablePrimitiveArray<T>> for PrimitiveArray<T> {
    fn from(other: MutablePrimitiveArray<T>) -> Self {
        let validity = other.validity.and_then(|x| {
            let bitmap: Bitmap = x.into();
            if bitmap.unset_bits() == 0 {
                None
            } else {
                Some(bitmap)
            }
        });

        PrimitiveArray::<T>::new(other.data_type, other.values.into(), validity)
    }
}

impl<T: NativeType, P: AsRef<[Option<T>]>> From<P> for MutablePrimitiveArray<T> {
    fn from(slice: P) -> Self {
        Self::from_trusted_len_iter(slice.as_ref().iter().map(|x| x.as_ref()))
    }
}

impl<T: NativeType> MutablePrimitiveArray<T> {
    /// Creates a new empty [`MutablePrimitiveArray`].
    pub fn new() -> Self {
        Self::with_capacity(0)
    }

    /// Creates a new [`MutablePrimitiveArray`] with a capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_from(capacity, T::PRIMITIVE.into())
    }

    /// The canonical method to create a [`MutablePrimitiveArray`] out of its internal components.
    /// # Implementation
    /// This function is `O(1)`.
    ///
    /// # Errors
    /// This function errors iff:
    /// * The validity is not `None` and its length is different from `values`'s length
    /// * The `data_type`'s [`crate::datatypes::PhysicalType`] is not equal to [`crate::datatypes::PhysicalType::Primitive(T::PRIMITIVE)`]
    pub fn try_new(
        data_type: ArrowDataType,
        values: Vec<T>,
        validity: Option<MutableBitmap>,
    ) -> PolarsResult<Self> {
        check(&data_type, &values, validity.as_ref().map(|x| x.len()))?;
        Ok(Self {
            data_type,
            values,
            validity,
        })
    }

    /// Extract the low-end APIs from the [`MutablePrimitiveArray`].
    pub fn into_inner(self) -> (ArrowDataType, Vec<T>, Option<MutableBitmap>) {
        (self.data_type, self.values, self.validity)
    }

    /// Applies a function `f` to the values of this array, cloning the values
    /// iff they are being shared with others
    ///
    /// This is an API to use clone-on-write
    /// # Implementation
    /// This function is `O(f)` if the data is not being shared, and `O(N) + O(f)`
    /// if it is being shared (since it results in a `O(N)` memcopy).
    /// # Panics
    /// This function panics iff `f` panics
    pub fn apply_values<F: Fn(&mut [T])>(&mut self, f: F) {
        f(&mut self.values);
    }
}

impl<T: NativeType> Default for MutablePrimitiveArray<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: NativeType> From<ArrowDataType> for MutablePrimitiveArray<T> {
    fn from(data_type: ArrowDataType) -> Self {
        assert!(data_type.to_physical_type().eq_primitive(T::PRIMITIVE));
        Self {
            data_type,
            values: Vec::<T>::new(),
            validity: None,
        }
    }
}

impl<T: NativeType> MutablePrimitiveArray<T> {
    /// Creates a new [`MutablePrimitiveArray`] from a capacity and [`ArrowDataType`].
    pub fn with_capacity_from(capacity: usize, data_type: ArrowDataType) -> Self {
        assert!(data_type.to_physical_type().eq_primitive(T::PRIMITIVE));
        Self {
            data_type,
            values: Vec::<T>::with_capacity(capacity),
            validity: None,
        }
    }

    /// Reserves `additional` entries.
    pub fn reserve(&mut self, additional: usize) {
        self.values.reserve(additional);
        if let Some(x) = self.validity.as_mut() {
            x.reserve(additional)
        }
    }

    #[inline]
    pub fn push_value(&mut self, value: T) {
        self.values.push(value);
        if let Some(validity) = &mut self.validity {
            validity.push(true)
        }
    }

    /// Adds a new value to the array.
    #[inline]
    pub fn push(&mut self, value: Option<T>) {
        match value {
            Some(value) => self.push_value(value),
            None => {
                self.values.push(T::default());
                match &mut self.validity {
                    Some(validity) => validity.push(false),
                    None => {
                        self.init_validity();
                    },
                }
            },
        }
    }

    /// Pop a value from the array.
    /// Note if the values is empty, this method will return None.
    pub fn pop(&mut self) -> Option<T> {
        let value = self.values.pop()?;
        self.validity
            .as_mut()
            .map(|x| x.pop()?.then(|| value))
            .unwrap_or_else(|| Some(value))
    }

    /// Extends the [`MutablePrimitiveArray`] with a constant
    #[inline]
    pub fn extend_constant(&mut self, additional: usize, value: Option<T>) {
        if let Some(value) = value {
            self.values.resize(self.values.len() + additional, value);
            if let Some(validity) = &mut self.validity {
                validity.extend_constant(additional, true)
            }
        } else {
            if let Some(validity) = &mut self.validity {
                validity.extend_constant(additional, false)
            } else {
                let mut validity = MutableBitmap::with_capacity(self.values.capacity());
                validity.extend_constant(self.len(), true);
                validity.extend_constant(additional, false);
                self.validity = Some(validity)
            }
            self.values
                .resize(self.values.len() + additional, T::default());
        }
    }

    /// Extends the [`MutablePrimitiveArray`] from an iterator of trusted len.
    #[inline]
    pub fn extend_trusted_len<P, I>(&mut self, iterator: I)
    where
        P: std::borrow::Borrow<T>,
        I: TrustedLen<Item = Option<P>>,
    {
        unsafe { self.extend_trusted_len_unchecked(iterator) }
    }

    /// Extends the [`MutablePrimitiveArray`] from an iterator of trusted len.
    ///
    /// # Safety
    /// The iterator must be trusted len.
    #[inline]
    pub unsafe fn extend_trusted_len_unchecked<P, I>(&mut self, iterator: I)
    where
        P: std::borrow::Borrow<T>,
        I: Iterator<Item = Option<P>>,
    {
        if let Some(validity) = self.validity.as_mut() {
            extend_trusted_len_unzip(iterator, validity, &mut self.values)
        } else {
            let mut validity = MutableBitmap::new();
            validity.extend_constant(self.len(), true);
            extend_trusted_len_unzip(iterator, &mut validity, &mut self.values);
            self.validity = Some(validity);
        }
    }
    /// Extends the [`MutablePrimitiveArray`] from an iterator of values of trusted len.
    /// This differs from `extend_trusted_len` which accepts in iterator of optional values.
    #[inline]
    pub fn extend_trusted_len_values<I>(&mut self, iterator: I)
    where
        I: TrustedLen<Item = T>,
    {
        unsafe { self.extend_trusted_len_values_unchecked(iterator) }
    }

    /// Extends the [`MutablePrimitiveArray`] from an iterator of values of trusted len.
    /// This differs from `extend_trusted_len_unchecked` which accepts in iterator of optional values.
    ///
    /// # Safety
    /// The iterator must be trusted len.
    #[inline]
    pub unsafe fn extend_trusted_len_values_unchecked<I>(&mut self, iterator: I)
    where
        I: Iterator<Item = T>,
    {
        self.values.extend(iterator);
        self.update_all_valid();
    }

    #[inline]
    /// Extends the [`MutablePrimitiveArray`] from a slice
    pub fn extend_from_slice(&mut self, items: &[T]) {
        self.values.extend_from_slice(items);
        self.update_all_valid();
    }

    fn update_all_valid(&mut self) {
        // get len before mutable borrow
        let len = self.len();
        if let Some(validity) = self.validity.as_mut() {
            validity.extend_constant(len - validity.len(), true);
        }
    }

    fn init_validity(&mut self) {
        let mut validity = MutableBitmap::with_capacity(self.values.capacity());
        validity.extend_constant(self.len(), true);
        validity.set(self.len() - 1, false);
        self.validity = Some(validity)
    }

    /// Changes the arrays' [`ArrowDataType`], returning a new [`MutablePrimitiveArray`].
    /// Use to change the logical type without changing the corresponding physical Type.
    /// # Implementation
    /// This operation is `O(1)`.
    #[inline]
    pub fn to(self, data_type: ArrowDataType) -> Self {
        Self::try_new(data_type, self.values, self.validity).unwrap()
    }

    /// Converts itself into an [`Array`].
    pub fn into_arc(self) -> Arc<dyn Array> {
        let a: PrimitiveArray<T> = self.into();
        Arc::new(a)
    }

    /// Shrinks the capacity of the [`MutablePrimitiveArray`] to fit its current length.
    pub fn shrink_to_fit(&mut self) {
        self.values.shrink_to_fit();
        if let Some(validity) = &mut self.validity {
            validity.shrink_to_fit()
        }
    }

    /// Returns the capacity of this [`MutablePrimitiveArray`].
    pub fn capacity(&self) -> usize {
        self.values.capacity()
    }

    pub fn freeze(self) -> PrimitiveArray<T> {
        self.into()
    }

    /// Clears the array, removing all values.
    ///
    /// Note that this method has no effect on the allocated capacity
    /// of the array.
    pub fn clear(&mut self) {
        self.values.clear();
        self.validity = None;
    }

    /// Apply a function that temporarily freezes this `MutableArray` into a `PrimitiveArray`.
    pub fn with_freeze<K, F: FnOnce(&PrimitiveArray<T>) -> K>(&mut self, f: F) -> K {
        let mutable = std::mem::take(self);
        let arr = mutable.freeze();
        let out = f(&arr);
        *self = arr.into_mut().right().unwrap();
        out
    }
}

/// Accessors
impl<T: NativeType> MutablePrimitiveArray<T> {
    /// Returns its values.
    pub fn values(&self) -> &Vec<T> {
        &self.values
    }

    /// Returns a mutable slice of values.
    pub fn values_mut_slice(&mut self) -> &mut [T] {
        self.values.as_mut_slice()
    }
}

/// Setters
impl<T: NativeType> MutablePrimitiveArray<T> {
    /// Sets position `index` to `value`.
    /// Note that if it is the first time a null appears in this array,
    /// this initializes the validity bitmap (`O(N)`).
    /// # Panic
    /// Panics iff `index >= self.len()`.
    pub fn set(&mut self, index: usize, value: Option<T>) {
        assert!(index < self.len());
        // SAFETY:
        // we just checked bounds
        unsafe { self.set_unchecked(index, value) }
    }

    /// Sets position `index` to `value`.
    /// Note that if it is the first time a null appears in this array,
    /// this initializes the validity bitmap (`O(N)`).
    ///
    /// # Safety
    /// Caller must ensure `index < self.len()`
    pub unsafe fn set_unchecked(&mut self, index: usize, value: Option<T>) {
        *self.values.get_unchecked_mut(index) = value.unwrap_or_default();

        if value.is_none() && self.validity.is_none() {
            // When the validity is None, all elements so far are valid. When one of the elements is set of null,
            // the validity must be initialized.
            let mut validity = MutableBitmap::new();
            validity.extend_constant(self.len(), true);
            self.validity = Some(validity);
        }
        if let Some(x) = self.validity.as_mut() {
            x.set_unchecked(index, value.is_some())
        }
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

    /// Sets values.
    /// # Panic
    /// Panics iff the values' length is not equal to the existing validity's len.
    pub fn set_values(&mut self, values: Vec<T>) {
        assert_eq!(values.len(), self.values.len());
        self.values = values;
    }
}

impl<T: NativeType> Extend<Option<T>> for MutablePrimitiveArray<T> {
    fn extend<I: IntoIterator<Item = Option<T>>>(&mut self, iter: I) {
        let iter = iter.into_iter();
        self.reserve(iter.size_hint().0);
        iter.for_each(|x| self.push(x))
    }
}

impl<T: NativeType> TryExtend<Option<T>> for MutablePrimitiveArray<T> {
    /// This is infallible and is implemented for consistency with all other types
    fn try_extend<I: IntoIterator<Item = Option<T>>>(&mut self, iter: I) -> PolarsResult<()> {
        self.extend(iter);
        Ok(())
    }
}

impl<T: NativeType> TryPush<Option<T>> for MutablePrimitiveArray<T> {
    /// This is infalible and is implemented for consistency with all other types
    #[inline]
    fn try_push(&mut self, item: Option<T>) -> PolarsResult<()> {
        self.push(item);
        Ok(())
    }
}

impl<T: NativeType> MutableArray for MutablePrimitiveArray<T> {
    fn len(&self) -> usize {
        self.values.len()
    }

    fn validity(&self) -> Option<&MutableBitmap> {
        self.validity.as_ref()
    }

    fn as_box(&mut self) -> Box<dyn Array> {
        PrimitiveArray::new(
            self.data_type.clone(),
            std::mem::take(&mut self.values).into(),
            std::mem::take(&mut self.validity).map(|x| x.into()),
        )
        .boxed()
    }

    fn as_arc(&mut self) -> Arc<dyn Array> {
        PrimitiveArray::new(
            self.data_type.clone(),
            std::mem::take(&mut self.values).into(),
            std::mem::take(&mut self.validity).map(|x| x.into()),
        )
        .arced()
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

    fn push_null(&mut self) {
        self.push(None)
    }

    fn reserve(&mut self, additional: usize) {
        self.reserve(additional)
    }

    fn shrink_to_fit(&mut self) {
        self.shrink_to_fit()
    }
}

impl<T: NativeType> MutablePrimitiveArray<T> {
    /// Creates a [`MutablePrimitiveArray`] from a slice of values.
    pub fn from_slice<P: AsRef<[T]>>(slice: P) -> Self {
        Self::from_trusted_len_values_iter(slice.as_ref().iter().copied())
    }

    /// Creates a [`MutablePrimitiveArray`] from an iterator of trusted length.
    ///
    /// # Safety
    /// The iterator must be [`TrustedLen`](https://doc.rust-lang.org/std/iter/trait.TrustedLen.html).
    /// I.e. `size_hint().1` correctly reports its length.
    #[inline]
    pub unsafe fn from_trusted_len_iter_unchecked<I, P>(iterator: I) -> Self
    where
        P: std::borrow::Borrow<T>,
        I: Iterator<Item = Option<P>>,
    {
        let (validity, values) = trusted_len_unzip(iterator);

        Self {
            data_type: T::PRIMITIVE.into(),
            values,
            validity,
        }
    }

    /// Creates a [`MutablePrimitiveArray`] from a [`TrustedLen`].
    #[inline]
    pub fn from_trusted_len_iter<I, P>(iterator: I) -> Self
    where
        P: std::borrow::Borrow<T>,
        I: TrustedLen<Item = Option<P>>,
    {
        unsafe { Self::from_trusted_len_iter_unchecked(iterator) }
    }

    /// Creates a [`MutablePrimitiveArray`] from an fallible iterator of trusted length.
    ///
    /// # Safety
    /// The iterator must be [`TrustedLen`](https://doc.rust-lang.org/std/iter/trait.TrustedLen.html).
    /// I.e. that `size_hint().1` correctly reports its length.
    #[inline]
    pub unsafe fn try_from_trusted_len_iter_unchecked<E, I, P>(
        iter: I,
    ) -> std::result::Result<Self, E>
    where
        P: std::borrow::Borrow<T>,
        I: IntoIterator<Item = std::result::Result<Option<P>, E>>,
    {
        let iterator = iter.into_iter();

        let (validity, values) = try_trusted_len_unzip(iterator)?;

        Ok(Self {
            data_type: T::PRIMITIVE.into(),
            values,
            validity,
        })
    }

    /// Creates a [`MutablePrimitiveArray`] from an fallible iterator of trusted length.
    #[inline]
    pub fn try_from_trusted_len_iter<E, I, P>(iterator: I) -> std::result::Result<Self, E>
    where
        P: std::borrow::Borrow<T>,
        I: TrustedLen<Item = std::result::Result<Option<P>, E>>,
    {
        unsafe { Self::try_from_trusted_len_iter_unchecked(iterator) }
    }

    /// Creates a new [`MutablePrimitiveArray`] out an iterator over values
    pub fn from_trusted_len_values_iter<I: TrustedLen<Item = T>>(iter: I) -> Self {
        Self {
            data_type: T::PRIMITIVE.into(),
            values: iter.collect(),
            validity: None,
        }
    }

    /// Creates a (non-null) [`MutablePrimitiveArray`] from a vector of values.
    /// This does not have memcopy and is the fastest way to create a [`PrimitiveArray`].
    pub fn from_vec(values: Vec<T>) -> Self {
        Self::try_new(T::PRIMITIVE.into(), values, None).unwrap()
    }

    /// Creates a new [`MutablePrimitiveArray`] from an iterator over values
    ///
    /// # Safety
    /// The iterator must be [`TrustedLen`](https://doc.rust-lang.org/std/iter/trait.TrustedLen.html).
    /// I.e. that `size_hint().1` correctly reports its length.
    pub unsafe fn from_trusted_len_values_iter_unchecked<I: Iterator<Item = T>>(iter: I) -> Self {
        Self {
            data_type: T::PRIMITIVE.into(),
            values: iter.collect(),
            validity: None,
        }
    }
}

impl<T: NativeType, Ptr: std::borrow::Borrow<Option<T>>> FromIterator<Ptr>
    for MutablePrimitiveArray<T>
{
    fn from_iter<I: IntoIterator<Item = Ptr>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();

        let mut validity = MutableBitmap::with_capacity(lower);

        let values: Vec<T> = iter
            .map(|item| {
                if let Some(a) = item.borrow() {
                    validity.push(true);
                    *a
                } else {
                    validity.push(false);
                    T::default()
                }
            })
            .collect();

        let validity = Some(validity);

        Self {
            data_type: T::PRIMITIVE.into(),
            values,
            validity,
        }
    }
}

/// Extends a [`MutableBitmap`] and a [`Vec`] from an iterator of `Option`.
/// The first buffer corresponds to a bitmap buffer, the second one
/// corresponds to a values buffer.
/// # Safety
/// The caller must ensure that `iterator` is `TrustedLen`.
#[inline]
pub(crate) unsafe fn extend_trusted_len_unzip<I, P, T>(
    iterator: I,
    validity: &mut MutableBitmap,
    buffer: &mut Vec<T>,
) where
    T: NativeType,
    P: std::borrow::Borrow<T>,
    I: Iterator<Item = Option<P>>,
{
    let (_, upper) = iterator.size_hint();
    let additional = upper.expect("trusted_len_unzip requires an upper limit");

    validity.reserve(additional);
    let values = iterator.map(|item| {
        if let Some(item) = item {
            validity.push_unchecked(true);
            *item.borrow()
        } else {
            validity.push_unchecked(false);
            T::default()
        }
    });
    buffer.extend(values);
}

/// Creates a [`MutableBitmap`] and a [`Vec`] from an iterator of `Option`.
/// The first buffer corresponds to a bitmap buffer, the second one
/// corresponds to a values buffer.
/// # Safety
/// The caller must ensure that `iterator` is `TrustedLen`.
#[inline]
pub(crate) unsafe fn trusted_len_unzip<I, P, T>(iterator: I) -> (Option<MutableBitmap>, Vec<T>)
where
    T: NativeType,
    P: std::borrow::Borrow<T>,
    I: Iterator<Item = Option<P>>,
{
    let mut validity = MutableBitmap::new();
    let mut buffer = Vec::<T>::new();

    extend_trusted_len_unzip(iterator, &mut validity, &mut buffer);

    let validity = Some(validity);

    (validity, buffer)
}

/// # Safety
/// The caller must ensure that `iterator` is `TrustedLen`.
#[inline]
pub(crate) unsafe fn try_trusted_len_unzip<E, I, P, T>(
    iterator: I,
) -> std::result::Result<(Option<MutableBitmap>, Vec<T>), E>
where
    T: NativeType,
    P: std::borrow::Borrow<T>,
    I: Iterator<Item = std::result::Result<Option<P>, E>>,
{
    let (_, upper) = iterator.size_hint();
    let len = upper.expect("trusted_len_unzip requires an upper limit");

    let mut null = MutableBitmap::with_capacity(len);
    let mut buffer = Vec::<T>::with_capacity(len);

    let mut dst = buffer.as_mut_ptr();
    for item in iterator {
        let item = if let Some(item) = item? {
            null.push(true);
            *item.borrow()
        } else {
            null.push(false);
            T::default()
        };
        std::ptr::write(dst, item);
        dst = dst.add(1);
    }
    assert_eq!(
        dst.offset_from(buffer.as_ptr()) as usize,
        len,
        "Trusted iterator length was not accurately reported"
    );
    buffer.set_len(len);
    null.set_len(len);

    let validity = Some(null);

    Ok((validity, buffer))
}

impl<T: NativeType> PartialEq for MutablePrimitiveArray<T> {
    fn eq(&self, other: &Self) -> bool {
        self.iter().eq(other.iter())
    }
}

impl<T: NativeType> TryExtendFromSelf for MutablePrimitiveArray<T> {
    fn try_extend_from_self(&mut self, other: &Self) -> PolarsResult<()> {
        extend_validity(self.len(), &mut self.validity, &other.validity);

        let slice = other.values.as_slice();
        self.values.extend_from_slice(slice);
        Ok(())
    }
}
