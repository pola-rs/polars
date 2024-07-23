use std::hash::Hash;
use std::hint::unreachable_unchecked;

use crate::bitmap::utils::{BitmapIter, ZipValidity};
use crate::bitmap::Bitmap;
use crate::datatypes::{ArrowDataType, IntegerType};
use crate::scalar::{new_scalar, Scalar};
use crate::trusted_len::TrustedLen;
use crate::types::NativeType;

#[cfg(feature = "arrow_rs")]
mod data;
mod ffi;
pub(super) mod fmt;
mod iterator;
mod mutable;
use crate::array::specification::check_indexes_unchecked;
mod typed_iterator;
mod value_map;

pub use iterator::*;
pub use mutable::*;
use polars_error::{polars_bail, PolarsResult};

use super::primitive::PrimitiveArray;
use super::specification::check_indexes;
use super::{new_empty_array, new_null_array, Array, Splitable};
use crate::array::dictionary::typed_iterator::{
    DictValue, DictionaryIterTyped, DictionaryValuesIterTyped,
};

/// Trait denoting [`NativeType`]s that can be used as keys of a dictionary.
/// # Safety
///
/// Any implementation of this trait must ensure that `always_fits_usize` only
/// returns `true` if all values succeeds on `value::try_into::<usize>().unwrap()`.
pub unsafe trait DictionaryKey: NativeType + TryInto<usize> + TryFrom<usize> + Hash {
    /// The corresponding [`IntegerType`] of this key
    const KEY_TYPE: IntegerType;
    const MAX_USIZE_VALUE: usize;

    /// Represents this key as a `usize`.
    ///
    /// # Safety
    /// The caller _must_ have checked that the value can be casted to `usize`.
    #[inline]
    unsafe fn as_usize(self) -> usize {
        match self.try_into() {
            Ok(v) => v,
            Err(_) => unreachable_unchecked(),
        }
    }

    /// Create a key from a `usize` without checking bounds.
    ///
    /// # Safety
    /// The caller _must_ have checked that the value can be created from a `usize`.
    #[inline]
    unsafe fn from_usize_unchecked(x: usize) -> Self {
        debug_assert!(Self::try_from(x).is_ok());
        unsafe { Self::try_from(x).unwrap_unchecked() }
    }

    /// If the key type always can be converted to `usize`.
    fn always_fits_usize() -> bool {
        false
    }
}

unsafe impl DictionaryKey for i8 {
    const KEY_TYPE: IntegerType = IntegerType::Int8;
    const MAX_USIZE_VALUE: usize = i8::MAX as usize;
}
unsafe impl DictionaryKey for i16 {
    const KEY_TYPE: IntegerType = IntegerType::Int16;
    const MAX_USIZE_VALUE: usize = i16::MAX as usize;
}
unsafe impl DictionaryKey for i32 {
    const KEY_TYPE: IntegerType = IntegerType::Int32;
    const MAX_USIZE_VALUE: usize = i32::MAX as usize;
}
unsafe impl DictionaryKey for i64 {
    const KEY_TYPE: IntegerType = IntegerType::Int64;
    const MAX_USIZE_VALUE: usize = i64::MAX as usize;
}
unsafe impl DictionaryKey for u8 {
    const KEY_TYPE: IntegerType = IntegerType::UInt8;
    const MAX_USIZE_VALUE: usize = u8::MAX as usize;

    fn always_fits_usize() -> bool {
        true
    }
}
unsafe impl DictionaryKey for u16 {
    const KEY_TYPE: IntegerType = IntegerType::UInt16;
    const MAX_USIZE_VALUE: usize = u16::MAX as usize;

    fn always_fits_usize() -> bool {
        true
    }
}
unsafe impl DictionaryKey for u32 {
    const KEY_TYPE: IntegerType = IntegerType::UInt32;
    const MAX_USIZE_VALUE: usize = u32::MAX as usize;

    fn always_fits_usize() -> bool {
        true
    }
}
unsafe impl DictionaryKey for u64 {
    const KEY_TYPE: IntegerType = IntegerType::UInt64;
    const MAX_USIZE_VALUE: usize = u64::MAX as usize;

    #[cfg(target_pointer_width = "64")]
    fn always_fits_usize() -> bool {
        true
    }
}

/// An [`Array`] whose values are stored as indices. This [`Array`] is useful when the cardinality of
/// values is low compared to the length of the [`Array`].
///
/// # Safety
/// This struct guarantees that each item of [`DictionaryArray::keys`] is castable to `usize` and
/// its value is smaller than [`DictionaryArray::values`]`.len()`. In other words, you can safely
/// use `unchecked` calls to retrieve the values
#[derive(Clone)]
pub struct DictionaryArray<K: DictionaryKey> {
    data_type: ArrowDataType,
    keys: PrimitiveArray<K>,
    values: Box<dyn Array>,
}

fn check_data_type(
    key_type: IntegerType,
    data_type: &ArrowDataType,
    values_data_type: &ArrowDataType,
) -> PolarsResult<()> {
    if let ArrowDataType::Dictionary(key, value, _) = data_type.to_logical_type() {
        if *key != key_type {
            polars_bail!(ComputeError: "DictionaryArray must be initialized with a DataType::Dictionary whose integer is compatible to its keys")
        }
        if value.as_ref().to_logical_type() != values_data_type.to_logical_type() {
            polars_bail!(ComputeError: "DictionaryArray must be initialized with a DataType::Dictionary whose value is equal to its values")
        }
    } else {
        polars_bail!(ComputeError: "DictionaryArray must be initialized with logical DataType::Dictionary")
    }
    Ok(())
}

impl<K: DictionaryKey> DictionaryArray<K> {
    /// Returns a new [`DictionaryArray`].
    /// # Implementation
    /// This function is `O(N)` where `N` is the length of keys
    /// # Errors
    /// This function errors iff
    /// * the `data_type`'s logical type is not a `DictionaryArray`
    /// * the `data_type`'s keys is not compatible with `keys`
    /// * the `data_type`'s values's data_type is not equal with `values.data_type()`
    /// * any of the keys's values is not represented in `usize` or is `>= values.len()`
    pub fn try_new(
        data_type: ArrowDataType,
        keys: PrimitiveArray<K>,
        values: Box<dyn Array>,
    ) -> PolarsResult<Self> {
        check_data_type(K::KEY_TYPE, &data_type, values.data_type())?;

        if keys.null_count() != keys.len() {
            if K::always_fits_usize() {
                // SAFETY: we just checked that conversion to `usize` always
                // succeeds
                unsafe { check_indexes_unchecked(keys.values(), values.len()) }?;
            } else {
                check_indexes(keys.values(), values.len())?;
            }
        }

        Ok(Self {
            data_type,
            keys,
            values,
        })
    }

    /// Returns a new [`DictionaryArray`].
    /// # Implementation
    /// This function is `O(N)` where `N` is the length of keys
    /// # Errors
    /// This function errors iff
    /// * any of the keys's values is not represented in `usize` or is `>= values.len()`
    pub fn try_from_keys(keys: PrimitiveArray<K>, values: Box<dyn Array>) -> PolarsResult<Self> {
        let data_type = Self::default_data_type(values.data_type().clone());
        Self::try_new(data_type, keys, values)
    }

    /// Returns a new [`DictionaryArray`].
    /// # Errors
    /// This function errors iff
    /// * the `data_type`'s logical type is not a `DictionaryArray`
    /// * the `data_type`'s keys is not compatible with `keys`
    /// * the `data_type`'s values's data_type is not equal with `values.data_type()`
    ///
    /// # Safety
    /// The caller must ensure that every keys's values is represented in `usize` and is `< values.len()`
    pub unsafe fn try_new_unchecked(
        data_type: ArrowDataType,
        keys: PrimitiveArray<K>,
        values: Box<dyn Array>,
    ) -> PolarsResult<Self> {
        check_data_type(K::KEY_TYPE, &data_type, values.data_type())?;

        Ok(Self {
            data_type,
            keys,
            values,
        })
    }

    /// Returns a new empty [`DictionaryArray`].
    pub fn new_empty(data_type: ArrowDataType) -> Self {
        let values = Self::try_get_child(&data_type).unwrap();
        let values = new_empty_array(values.clone());
        Self::try_new(
            data_type,
            PrimitiveArray::<K>::new_empty(K::PRIMITIVE.into()),
            values,
        )
        .unwrap()
    }

    /// Returns an [`DictionaryArray`] whose all elements are null
    #[inline]
    pub fn new_null(data_type: ArrowDataType, length: usize) -> Self {
        let values = Self::try_get_child(&data_type).unwrap();
        let values = new_null_array(values.clone(), 1);
        Self::try_new(
            data_type,
            PrimitiveArray::<K>::new_null(K::PRIMITIVE.into(), length),
            values,
        )
        .unwrap()
    }

    /// Returns an iterator of [`Option<Box<dyn Scalar>>`].
    /// # Implementation
    /// This function will allocate a new [`Scalar`] per item and is usually not performant.
    /// Consider calling `keys_iter` and `values`, downcasting `values`, and iterating over that.
    pub fn iter(&self) -> ZipValidity<Box<dyn Scalar>, DictionaryValuesIter<K>, BitmapIter> {
        ZipValidity::new_with_validity(DictionaryValuesIter::new(self), self.keys.validity())
    }

    /// Returns an iterator of [`Box<dyn Scalar>`]
    /// # Implementation
    /// This function will allocate a new [`Scalar`] per item and is usually not performant.
    /// Consider calling `keys_iter` and `values`, downcasting `values`, and iterating over that.
    pub fn values_iter(&self) -> DictionaryValuesIter<K> {
        DictionaryValuesIter::new(self)
    }

    /// Returns an iterator over the values [`V::IterValue`].
    ///
    /// # Panics
    ///
    /// Panics if the keys of this [`DictionaryArray`] has any nulls.
    /// If they do [`DictionaryArray::iter_typed`] should be used.
    pub fn values_iter_typed<V: DictValue>(&self) -> PolarsResult<DictionaryValuesIterTyped<K, V>> {
        let keys = &self.keys;
        assert_eq!(keys.null_count(), 0);
        let values = self.values.as_ref();
        let values = V::downcast_values(values)?;
        Ok(DictionaryValuesIterTyped::new(keys, values))
    }

    /// Returns an iterator over the optional values of  [`Option<V::IterValue>`].
    pub fn iter_typed<V: DictValue>(&self) -> PolarsResult<DictionaryIterTyped<K, V>> {
        let keys = &self.keys;
        let values = self.values.as_ref();
        let values = V::downcast_values(values)?;
        Ok(DictionaryIterTyped::new(keys, values))
    }

    /// Returns the [`ArrowDataType`] of this [`DictionaryArray`]
    #[inline]
    pub fn data_type(&self) -> &ArrowDataType {
        &self.data_type
    }

    /// Returns whether the values of this [`DictionaryArray`] are ordered
    #[inline]
    pub fn is_ordered(&self) -> bool {
        match self.data_type.to_logical_type() {
            ArrowDataType::Dictionary(_, _, is_ordered) => *is_ordered,
            _ => unreachable!(),
        }
    }

    pub(crate) fn default_data_type(values_datatype: ArrowDataType) -> ArrowDataType {
        ArrowDataType::Dictionary(K::KEY_TYPE, Box::new(values_datatype), false)
    }

    /// Slices this [`DictionaryArray`].
    /// # Panics
    /// iff `offset + length > self.len()`.
    pub fn slice(&mut self, offset: usize, length: usize) {
        self.keys.slice(offset, length);
    }

    /// Slices this [`DictionaryArray`].
    ///
    /// # Safety
    /// Safe iff `offset + length <= self.len()`.
    pub unsafe fn slice_unchecked(&mut self, offset: usize, length: usize) {
        self.keys.slice_unchecked(offset, length);
    }

    impl_sliced!();

    /// Returns this [`DictionaryArray`] with a new validity.
    /// # Panic
    /// This function panics iff `validity.len() != self.len()`.
    #[must_use]
    pub fn with_validity(mut self, validity: Option<Bitmap>) -> Self {
        self.set_validity(validity);
        self
    }

    /// Sets the validity of the keys of this [`DictionaryArray`].
    /// # Panics
    /// This function panics iff `validity.len() != self.len()`.
    pub fn set_validity(&mut self, validity: Option<Bitmap>) {
        self.keys.set_validity(validity);
    }

    impl_into_array!();

    /// Returns the length of this array
    #[inline]
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// The optional validity. Equivalent to `self.keys().validity()`.
    #[inline]
    pub fn validity(&self) -> Option<&Bitmap> {
        self.keys.validity()
    }

    /// Returns the keys of the [`DictionaryArray`]. These keys can be used to fetch values
    /// from `values`.
    #[inline]
    pub fn keys(&self) -> &PrimitiveArray<K> {
        &self.keys
    }

    /// Returns an iterator of the keys' values of the [`DictionaryArray`] as `usize`
    #[inline]
    pub fn keys_values_iter(&self) -> impl TrustedLen<Item = usize> + Clone + '_ {
        // SAFETY: invariant of the struct
        self.keys.values_iter().map(|x| unsafe { x.as_usize() })
    }

    /// Returns an iterator of the keys' of the [`DictionaryArray`] as `usize`
    #[inline]
    pub fn keys_iter(&self) -> impl TrustedLen<Item = Option<usize>> + Clone + '_ {
        // SAFETY: invariant of the struct
        self.keys.iter().map(|x| x.map(|x| unsafe { x.as_usize() }))
    }

    /// Returns the keys' value of the [`DictionaryArray`] as `usize`
    /// # Panics
    /// This function panics iff `index >= self.len()`
    #[inline]
    pub fn key_value(&self, index: usize) -> usize {
        // SAFETY: invariant of the struct
        unsafe { self.keys.values()[index].as_usize() }
    }

    /// Returns the values of the [`DictionaryArray`].
    #[inline]
    pub fn values(&self) -> &Box<dyn Array> {
        &self.values
    }

    /// Returns the value of the [`DictionaryArray`] at position `i`.
    /// # Implementation
    /// This function will allocate a new [`Scalar`] and is usually not performant.
    /// Consider calling `keys` and `values`, downcasting `values`, and iterating over that.
    /// # Panic
    /// This function panics iff `index >= self.len()`
    #[inline]
    pub fn value(&self, index: usize) -> Box<dyn Scalar> {
        // SAFETY: invariant of this struct
        let index = unsafe { self.keys.value(index).as_usize() };
        new_scalar(self.values.as_ref(), index)
    }

    pub(crate) fn try_get_child(data_type: &ArrowDataType) -> PolarsResult<&ArrowDataType> {
        Ok(match data_type.to_logical_type() {
            ArrowDataType::Dictionary(_, values, _) => values.as_ref(),
            _ => {
                polars_bail!(ComputeError: "Dictionaries must be initialized with DataType::Dictionary")
            },
        })
    }
}

impl<K: DictionaryKey> Array for DictionaryArray<K> {
    impl_common_array!();

    fn validity(&self) -> Option<&Bitmap> {
        self.keys.validity()
    }

    #[inline]
    fn with_validity(&self, validity: Option<Bitmap>) -> Box<dyn Array> {
        Box::new(self.clone().with_validity(validity))
    }
}

impl<K: DictionaryKey> Splitable for DictionaryArray<K> {
    fn check_bound(&self, offset: usize) -> bool {
        offset < self.len()
    }

    unsafe fn _split_at_unchecked(&self, offset: usize) -> (Self, Self) {
        let (lhs_keys, rhs_keys) = unsafe { Splitable::split_at_unchecked(&self.keys, offset) };

        (
            Self {
                data_type: self.data_type.clone(),
                keys: lhs_keys,
                values: self.values.clone(),
            },
            Self {
                data_type: self.data_type.clone(),
                keys: rhs_keys,
                values: self.values.clone(),
            },
        )
    }
}
