use std::sync::Arc;

use polars_error::{polars_bail, PolarsResult};

use super::BooleanArray;
use crate::array::physical_binary::extend_validity;
use crate::array::{Array, MutableArray, TryExtend, TryExtendFromSelf, TryPush};
use crate::bitmap::MutableBitmap;
use crate::datatypes::{ArrowDataType, PhysicalType};
use crate::trusted_len::TrustedLen;

/// The Arrow's equivalent to `Vec<Option<bool>>`, but with `1/16` of its size.
/// Converting a [`MutableBooleanArray`] into a [`BooleanArray`] is `O(1)`.
/// # Implementation
/// This struct does not allocate a validity until one is required (i.e. push a null to it).
#[derive(Debug, Clone)]
pub struct MutableBooleanArray {
    data_type: ArrowDataType,
    values: MutableBitmap,
    validity: Option<MutableBitmap>,
}

impl From<MutableBooleanArray> for BooleanArray {
    fn from(other: MutableBooleanArray) -> Self {
        BooleanArray::new(
            other.data_type,
            other.values.into(),
            other.validity.map(|x| x.into()),
        )
    }
}

impl<P: AsRef<[Option<bool>]>> From<P> for MutableBooleanArray {
    /// Creates a new [`MutableBooleanArray`] out of a slice of Optional `bool`.
    fn from(slice: P) -> Self {
        Self::from_trusted_len_iter(slice.as_ref().iter().map(|x| x.as_ref()))
    }
}

impl Default for MutableBooleanArray {
    fn default() -> Self {
        Self::new()
    }
}

impl MutableBooleanArray {
    /// Creates an new empty [`MutableBooleanArray`].
    pub fn new() -> Self {
        Self::with_capacity(0)
    }

    /// The canonical method to create a [`MutableBooleanArray`] out of low-end APIs.
    /// # Errors
    /// This function errors iff:
    /// * The validity is not `None` and its length is different from `values`'s length
    /// * The `data_type`'s [`PhysicalType`] is not equal to [`PhysicalType::Boolean`].
    pub fn try_new(
        data_type: ArrowDataType,
        values: MutableBitmap,
        validity: Option<MutableBitmap>,
    ) -> PolarsResult<Self> {
        if validity
            .as_ref()
            .map_or(false, |validity| validity.len() != values.len())
        {
            polars_bail!(ComputeError:
                "validity mask length must match the number of values",
            )
        }

        if data_type.to_physical_type() != PhysicalType::Boolean {
            polars_bail!(oos =
                "MutableBooleanArray can only be initialized with a DataType whose physical type is Boolean",
            )
        }

        Ok(Self {
            data_type,
            values,
            validity,
        })
    }

    /// Creates an new [`MutableBooleanArray`] with a capacity of values.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data_type: ArrowDataType::Boolean,
            values: MutableBitmap::with_capacity(capacity),
            validity: None,
        }
    }

    /// Reserves `additional` slots.
    pub fn reserve(&mut self, additional: usize) {
        self.values.reserve(additional);
        if let Some(x) = self.validity.as_mut() {
            x.reserve(additional)
        }
    }

    #[inline]
    pub fn push_value(&mut self, value: bool) {
        self.values.push(value);
        if let Some(validity) = &mut self.validity {
            validity.push(true)
        }
    }

    #[inline]
    pub fn push_null(&mut self) {
        self.values.push(false);
        match &mut self.validity {
            Some(validity) => validity.push(false),
            None => self.init_validity(),
        }
    }

    /// Pushes a new entry to [`MutableBooleanArray`].
    #[inline]
    pub fn push(&mut self, value: Option<bool>) {
        match value {
            Some(value) => self.push_value(value),
            None => self.push_null(),
        }
    }

    /// Pop an entry from [`MutableBooleanArray`].
    /// Note If the values is empty, this method will return None.
    pub fn pop(&mut self) -> Option<bool> {
        let value = self.values.pop()?;
        self.validity
            .as_mut()
            .map(|x| x.pop()?.then(|| value))
            .unwrap_or_else(|| Some(value))
    }

    /// Extends the [`MutableBooleanArray`] from an iterator of values of trusted len.
    /// This differs from `extend_trusted_len` which accepts in iterator of optional values.
    #[inline]
    pub fn extend_trusted_len_values<I>(&mut self, iterator: I)
    where
        I: TrustedLen<Item = bool>,
    {
        // SAFETY: `I` is `TrustedLen`
        unsafe { self.extend_trusted_len_values_unchecked(iterator) }
    }

    /// Extends the [`MutableBooleanArray`] from an iterator of values of trusted len.
    /// This differs from `extend_trusted_len_unchecked`, which accepts in iterator of optional values.
    ///
    /// # Safety
    /// The iterator must be trusted len.
    #[inline]
    pub unsafe fn extend_trusted_len_values_unchecked<I>(&mut self, iterator: I)
    where
        I: Iterator<Item = bool>,
    {
        let (_, upper) = iterator.size_hint();
        let additional =
            upper.expect("extend_trusted_len_values_unchecked requires an upper limit");

        if let Some(validity) = self.validity.as_mut() {
            validity.extend_constant(additional, true);
        }

        self.values.extend_from_trusted_len_iter_unchecked(iterator)
    }

    /// Extends the [`MutableBooleanArray`] from an iterator of trusted len.
    #[inline]
    pub fn extend_trusted_len<I, P>(&mut self, iterator: I)
    where
        P: std::borrow::Borrow<bool>,
        I: TrustedLen<Item = Option<P>>,
    {
        // SAFETY: `I` is `TrustedLen`
        unsafe { self.extend_trusted_len_unchecked(iterator) }
    }

    /// Extends the [`MutableBooleanArray`] from an iterator of trusted len.
    ///
    /// # Safety
    /// The iterator must be trusted len.
    #[inline]
    pub unsafe fn extend_trusted_len_unchecked<I, P>(&mut self, iterator: I)
    where
        P: std::borrow::Borrow<bool>,
        I: Iterator<Item = Option<P>>,
    {
        if let Some(validity) = self.validity.as_mut() {
            extend_trusted_len_unzip(iterator, validity, &mut self.values);
        } else {
            let mut validity = MutableBitmap::new();
            validity.extend_constant(self.len(), true);

            extend_trusted_len_unzip(iterator, &mut validity, &mut self.values);

            if validity.unset_bits() > 0 {
                self.validity = Some(validity);
            }
        }
    }

    /// Extends `MutableBooleanArray` by additional values of constant value.
    #[inline]
    pub fn extend_constant(&mut self, additional: usize, value: Option<bool>) {
        match value {
            Some(value) => {
                self.values.extend_constant(additional, value);
                if let Some(validity) = self.validity.as_mut() {
                    validity.extend_constant(additional, true);
                }
            },
            None => {
                self.values.extend_constant(additional, false);
                if let Some(validity) = self.validity.as_mut() {
                    validity.extend_constant(additional, false)
                } else {
                    self.init_validity();
                    self.validity
                        .as_mut()
                        .unwrap()
                        .extend_constant(additional, false)
                };
            },
        };
    }

    fn init_validity(&mut self) {
        let mut validity = MutableBitmap::with_capacity(self.values.capacity());
        validity.extend_constant(self.len(), true);
        validity.set(self.len() - 1, false);
        self.validity = Some(validity)
    }

    /// Converts itself into an [`Array`].
    pub fn into_arc(self) -> Arc<dyn Array> {
        let a: BooleanArray = self.into();
        Arc::new(a)
    }

    pub fn freeze(self) -> BooleanArray {
        self.into()
    }
}

/// Getters
impl MutableBooleanArray {
    /// Returns its values.
    pub fn values(&self) -> &MutableBitmap {
        &self.values
    }
}

/// Setters
impl MutableBooleanArray {
    /// Sets position `index` to `value`.
    /// Note that if it is the first time a null appears in this array,
    /// this initializes the validity bitmap (`O(N)`).
    /// # Panic
    /// Panics iff index is larger than `self.len()`.
    pub fn set(&mut self, index: usize, value: Option<bool>) {
        self.values.set(index, value.unwrap_or_default());

        if value.is_none() && self.validity.is_none() {
            // When the validity is None, all elements so far are valid. When one of the elements is set of null,
            // the validity must be initialized.
            self.validity = Some(MutableBitmap::from_trusted_len_iter(
                std::iter::repeat(true).take(self.len()),
            ));
        }
        if let Some(x) = self.validity.as_mut() {
            x.set(index, value.is_some())
        }
    }
}

/// From implementations
impl MutableBooleanArray {
    /// Creates a new [`MutableBooleanArray`] from an [`TrustedLen`] of `bool`.
    #[inline]
    pub fn from_trusted_len_values_iter<I: TrustedLen<Item = bool>>(iterator: I) -> Self {
        Self::try_new(
            ArrowDataType::Boolean,
            MutableBitmap::from_trusted_len_iter(iterator),
            None,
        )
        .unwrap()
    }

    /// Creates a new [`MutableBooleanArray`] from an [`TrustedLen`] of `bool`.
    /// Use this over [`BooleanArray::from_trusted_len_iter`] when the iterator is trusted len
    /// but this crate does not mark it as such.
    ///
    /// # Safety
    /// The iterator must be [`TrustedLen`](https://doc.rust-lang.org/std/iter/trait.TrustedLen.html).
    /// I.e. that `size_hint().1` correctly reports its length.
    #[inline]
    pub unsafe fn from_trusted_len_values_iter_unchecked<I: Iterator<Item = bool>>(
        iterator: I,
    ) -> Self {
        let mut mutable = MutableBitmap::new();
        mutable.extend_from_trusted_len_iter_unchecked(iterator);
        MutableBooleanArray::try_new(ArrowDataType::Boolean, mutable, None).unwrap()
    }

    /// Creates a new [`MutableBooleanArray`] from a slice of `bool`.
    #[inline]
    pub fn from_slice<P: AsRef<[bool]>>(slice: P) -> Self {
        Self::from_trusted_len_values_iter(slice.as_ref().iter().copied())
    }

    /// Creates a [`BooleanArray`] from an iterator of trusted length.
    /// Use this over [`BooleanArray::from_trusted_len_iter`] when the iterator is trusted len
    /// but this crate does not mark it as such.
    ///
    /// # Safety
    /// The iterator must be [`TrustedLen`](https://doc.rust-lang.org/std/iter/trait.TrustedLen.html).
    /// I.e. that `size_hint().1` correctly reports its length.
    #[inline]
    pub unsafe fn from_trusted_len_iter_unchecked<I, P>(iterator: I) -> Self
    where
        P: std::borrow::Borrow<bool>,
        I: Iterator<Item = Option<P>>,
    {
        let (validity, values) = trusted_len_unzip(iterator);

        Self::try_new(ArrowDataType::Boolean, values, validity).unwrap()
    }

    /// Creates a [`BooleanArray`] from a [`TrustedLen`].
    #[inline]
    pub fn from_trusted_len_iter<I, P>(iterator: I) -> Self
    where
        P: std::borrow::Borrow<bool>,
        I: TrustedLen<Item = Option<P>>,
    {
        // SAFETY: `I` is `TrustedLen`
        unsafe { Self::from_trusted_len_iter_unchecked(iterator) }
    }

    /// Creates a [`BooleanArray`] from an falible iterator of trusted length.
    ///
    /// # Safety
    /// The iterator must be [`TrustedLen`](https://doc.rust-lang.org/std/iter/trait.TrustedLen.html).
    /// I.e. that `size_hint().1` correctly reports its length.
    #[inline]
    pub unsafe fn try_from_trusted_len_iter_unchecked<E, I, P>(
        iterator: I,
    ) -> std::result::Result<Self, E>
    where
        P: std::borrow::Borrow<bool>,
        I: Iterator<Item = std::result::Result<Option<P>, E>>,
    {
        let (validity, values) = try_trusted_len_unzip(iterator)?;

        let validity = if validity.unset_bits() > 0 {
            Some(validity)
        } else {
            None
        };

        Ok(Self::try_new(ArrowDataType::Boolean, values, validity).unwrap())
    }

    /// Creates a [`BooleanArray`] from a [`TrustedLen`].
    #[inline]
    pub fn try_from_trusted_len_iter<E, I, P>(iterator: I) -> std::result::Result<Self, E>
    where
        P: std::borrow::Borrow<bool>,
        I: TrustedLen<Item = std::result::Result<Option<P>, E>>,
    {
        // SAFETY: `I` is `TrustedLen`
        unsafe { Self::try_from_trusted_len_iter_unchecked(iterator) }
    }

    /// Shrinks the capacity of the [`MutableBooleanArray`] to fit its current length.
    pub fn shrink_to_fit(&mut self) {
        self.values.shrink_to_fit();
        if let Some(validity) = &mut self.validity {
            validity.shrink_to_fit()
        }
    }
}

/// Creates a Bitmap and an optional [`MutableBitmap`] from an iterator of `Option<bool>`.
/// The first buffer corresponds to a bitmap buffer, the second one
/// corresponds to a values buffer.
/// # Safety
/// The caller must ensure that `iterator` is `TrustedLen`.
#[inline]
pub(crate) unsafe fn trusted_len_unzip<I, P>(iterator: I) -> (Option<MutableBitmap>, MutableBitmap)
where
    P: std::borrow::Borrow<bool>,
    I: Iterator<Item = Option<P>>,
{
    let mut validity = MutableBitmap::new();
    let mut values = MutableBitmap::new();

    extend_trusted_len_unzip(iterator, &mut validity, &mut values);

    let validity = if validity.unset_bits() > 0 {
        Some(validity)
    } else {
        None
    };

    (validity, values)
}

/// Extends validity [`MutableBitmap`] and values [`MutableBitmap`] from an iterator of `Option`.
/// # Safety
/// The caller must ensure that `iterator` is `TrustedLen`.
#[inline]
pub(crate) unsafe fn extend_trusted_len_unzip<I, P>(
    iterator: I,
    validity: &mut MutableBitmap,
    values: &mut MutableBitmap,
) where
    P: std::borrow::Borrow<bool>,
    I: Iterator<Item = Option<P>>,
{
    let (_, upper) = iterator.size_hint();
    let additional = upper.expect("extend_trusted_len_unzip requires an upper limit");

    // Length of the array before new values are pushed,
    // variable created for assertion post operation
    let pre_length = values.len();

    validity.reserve(additional);
    values.reserve(additional);

    for item in iterator {
        let item = if let Some(item) = item {
            validity.push_unchecked(true);
            *item.borrow()
        } else {
            validity.push_unchecked(false);
            bool::default()
        };
        values.push_unchecked(item);
    }

    debug_assert_eq!(
        values.len(),
        pre_length + additional,
        "Trusted iterator length was not accurately reported"
    );
}

/// # Safety
/// The caller must ensure that `iterator` is `TrustedLen`.
#[inline]
pub(crate) unsafe fn try_trusted_len_unzip<E, I, P>(
    iterator: I,
) -> std::result::Result<(MutableBitmap, MutableBitmap), E>
where
    P: std::borrow::Borrow<bool>,
    I: Iterator<Item = std::result::Result<Option<P>, E>>,
{
    let (_, upper) = iterator.size_hint();
    let len = upper.expect("trusted_len_unzip requires an upper limit");

    let mut null = MutableBitmap::with_capacity(len);
    let mut values = MutableBitmap::with_capacity(len);

    for item in iterator {
        let item = if let Some(item) = item? {
            null.push(true);
            *item.borrow()
        } else {
            null.push(false);
            false
        };
        values.push(item);
    }
    assert_eq!(
        values.len(),
        len,
        "Trusted iterator length was not accurately reported"
    );
    values.set_len(len);
    null.set_len(len);

    Ok((null, values))
}

impl<Ptr: std::borrow::Borrow<Option<bool>>> FromIterator<Ptr> for MutableBooleanArray {
    fn from_iter<I: IntoIterator<Item = Ptr>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();

        let mut validity = MutableBitmap::with_capacity(lower);

        let values: MutableBitmap = iter
            .map(|item| {
                if let Some(a) = item.borrow() {
                    validity.push(true);
                    *a
                } else {
                    validity.push(false);
                    false
                }
            })
            .collect();

        let validity = if validity.unset_bits() > 0 {
            Some(validity)
        } else {
            None
        };

        MutableBooleanArray::try_new(ArrowDataType::Boolean, values, validity).unwrap()
    }
}

impl MutableArray for MutableBooleanArray {
    fn len(&self) -> usize {
        self.values.len()
    }

    fn validity(&self) -> Option<&MutableBitmap> {
        self.validity.as_ref()
    }

    fn as_box(&mut self) -> Box<dyn Array> {
        let array: BooleanArray = std::mem::take(self).into();
        array.boxed()
    }

    fn as_arc(&mut self) -> Arc<dyn Array> {
        let array: BooleanArray = std::mem::take(self).into();
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
        self.push(None)
    }

    fn reserve(&mut self, additional: usize) {
        self.reserve(additional)
    }

    fn shrink_to_fit(&mut self) {
        self.shrink_to_fit()
    }
}

impl Extend<Option<bool>> for MutableBooleanArray {
    fn extend<I: IntoIterator<Item = Option<bool>>>(&mut self, iter: I) {
        let iter = iter.into_iter();
        self.reserve(iter.size_hint().0);
        iter.for_each(|x| self.push(x))
    }
}

impl TryExtend<Option<bool>> for MutableBooleanArray {
    /// This is infalible and is implemented for consistency with all other types
    fn try_extend<I: IntoIterator<Item = Option<bool>>>(&mut self, iter: I) -> PolarsResult<()> {
        self.extend(iter);
        Ok(())
    }
}

impl TryPush<Option<bool>> for MutableBooleanArray {
    /// This is infalible and is implemented for consistency with all other types
    fn try_push(&mut self, item: Option<bool>) -> PolarsResult<()> {
        self.push(item);
        Ok(())
    }
}

impl PartialEq for MutableBooleanArray {
    fn eq(&self, other: &Self) -> bool {
        self.iter().eq(other.iter())
    }
}

impl TryExtendFromSelf for MutableBooleanArray {
    fn try_extend_from_self(&mut self, other: &Self) -> PolarsResult<()> {
        extend_validity(self.len(), &mut self.validity, &other.validity);

        let slice = other.values.as_slice();
        // SAFETY: invariant offset + length <= slice.len()
        unsafe {
            self.values
                .extend_from_slice_unchecked(slice, 0, other.values.len());
        }
        Ok(())
    }
}
