use std::sync::Arc;

use crate::array::PushUnchecked;
use crate::{
    array::{
        physical_binary::extend_validity, Array, MutableArray, TryExtend, TryExtendFromSelf,
        TryPush,
    },
    bitmap::MutableBitmap,
    datatypes::{DataType, Field},
    error::{Error, Result},
};

use super::FixedSizeListArray;

/// The mutable version of [`FixedSizeListArray`].
#[derive(Debug, Clone)]
pub struct MutableFixedSizeListArray<M: MutableArray> {
    data_type: DataType,
    size: usize,
    values: M,
    validity: Option<MutableBitmap>,
}

impl<M: MutableArray> From<MutableFixedSizeListArray<M>> for FixedSizeListArray {
    fn from(mut other: MutableFixedSizeListArray<M>) -> Self {
        FixedSizeListArray::new(
            other.data_type,
            other.values.as_box(),
            other.validity.map(|x| x.into()),
        )
    }
}

impl<M: MutableArray> MutableFixedSizeListArray<M> {
    /// Creates a new [`MutableFixedSizeListArray`] from a [`MutableArray`] and size.
    pub fn new(values: M, size: usize) -> Self {
        let data_type = FixedSizeListArray::default_datatype(values.data_type().clone(), size);
        Self::new_from(values, data_type, size)
    }

    /// Creates a new [`MutableFixedSizeListArray`] from a [`MutableArray`] and size.
    pub fn new_with_field(values: M, name: &str, nullable: bool, size: usize) -> Self {
        let data_type = DataType::FixedSizeList(
            Box::new(Field::new(name, values.data_type().clone(), nullable)),
            size,
        );
        Self::new_from(values, data_type, size)
    }

    /// Creates a new [`MutableFixedSizeListArray`] from a [`MutableArray`], [`DataType`] and size.
    pub fn new_from(values: M, data_type: DataType, size: usize) -> Self {
        assert_eq!(values.len(), 0);
        match data_type {
            DataType::FixedSizeList(..) => (),
            _ => panic!("data type must be FixedSizeList (got {data_type:?})"),
        };
        Self {
            size,
            data_type,
            values,
            validity: None,
        }
    }

    /// Returns the size (number of elements per slot) of this [`FixedSizeListArray`].
    pub const fn size(&self) -> usize {
        self.size
    }

    /// The length of this array
    pub fn len(&self) -> usize {
        self.values.len() / self.size
    }

    /// The inner values
    pub fn values(&self) -> &M {
        &self.values
    }

    /// The values as a mutable reference
    pub fn mut_values(&mut self) -> &mut M {
        &mut self.values
    }

    fn init_validity(&mut self) {
        let len = self.values.len() / self.size;

        let mut validity = MutableBitmap::new();
        validity.extend_constant(len, true);
        validity.set(len - 1, false);
        self.validity = Some(validity)
    }

    #[inline]
    /// Needs to be called when a valid value was extended to this array.
    /// This is a relatively low level function, prefer `try_push` when you can.
    pub fn try_push_valid(&mut self) -> Result<()> {
        if self.values.len() % self.size != 0 {
            return Err(Error::Overflow);
        };
        if let Some(validity) = &mut self.validity {
            validity.push(true)
        }
        Ok(())
    }

    #[inline]
    /// Needs to be called when a valid value was extended to this array.
    /// This is a relatively low level function, prefer `try_push` when you can.
    pub fn push_valid(&mut self) {
        if let Some(validity) = &mut self.validity {
            validity.push(true)
        }
    }

    #[inline]
    fn push_null(&mut self) {
        (0..self.size).for_each(|_| self.values.push_null());
        match &mut self.validity {
            Some(validity) => validity.push(false),
            None => self.init_validity(),
        }
    }

    /// Reserves `additional` slots.
    pub fn reserve(&mut self, additional: usize) {
        self.values.reserve(additional);
        if let Some(x) = self.validity.as_mut() {
            x.reserve(additional)
        }
    }

    /// Shrinks the capacity of the [`MutableFixedSizeListArray`] to fit its current length.
    pub fn shrink_to_fit(&mut self) {
        self.values.shrink_to_fit();
        if let Some(validity) = &mut self.validity {
            validity.shrink_to_fit()
        }
    }
}

impl<M: MutableArray + 'static> MutableArray for MutableFixedSizeListArray<M> {
    fn len(&self) -> usize {
        self.values.len() / self.size
    }

    fn validity(&self) -> Option<&MutableBitmap> {
        self.validity.as_ref()
    }

    fn as_box(&mut self) -> Box<dyn Array> {
        FixedSizeListArray::new(
            self.data_type.clone(),
            self.values.as_box(),
            std::mem::take(&mut self.validity).map(|x| x.into()),
        )
        .boxed()
    }

    fn as_arc(&mut self) -> Arc<dyn Array> {
        FixedSizeListArray::new(
            self.data_type.clone(),
            self.values.as_box(),
            std::mem::take(&mut self.validity).map(|x| x.into()),
        )
        .arced()
    }

    fn data_type(&self) -> &DataType {
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
        (0..self.size).for_each(|_| {
            self.values.push_null();
        });
        if let Some(validity) = &mut self.validity {
            validity.push(false)
        } else {
            self.init_validity()
        }
    }

    fn reserve(&mut self, additional: usize) {
        self.reserve(additional)
    }

    fn shrink_to_fit(&mut self) {
        self.shrink_to_fit()
    }
}

impl<M, I, T> TryExtend<Option<I>> for MutableFixedSizeListArray<M>
where
    M: MutableArray + TryExtend<Option<T>>,
    I: IntoIterator<Item = Option<T>>,
{
    #[inline]
    fn try_extend<II: IntoIterator<Item = Option<I>>>(&mut self, iter: II) -> Result<()> {
        for items in iter {
            self.try_push(items)?;
        }
        Ok(())
    }
}

impl<M, I, T> TryPush<Option<I>> for MutableFixedSizeListArray<M>
where
    M: MutableArray + TryExtend<Option<T>>,
    I: IntoIterator<Item = Option<T>>,
{
    #[inline]
    fn try_push(&mut self, item: Option<I>) -> Result<()> {
        if let Some(items) = item {
            self.values.try_extend(items)?;
            self.try_push_valid()?;
        } else {
            self.push_null();
        }
        Ok(())
    }
}

impl<M, I, T> PushUnchecked<Option<I>> for MutableFixedSizeListArray<M>
where
    M: MutableArray + Extend<Option<T>>,
    I: IntoIterator<Item = Option<T>>,
{
    /// # Safety
    /// The caller must ensure that the `I` iterates exactly over `size`
    /// items, where `size` is the fixed size width.
    #[inline]
    unsafe fn push_unchecked(&mut self, item: Option<I>) {
        if let Some(items) = item {
            self.values.extend(items);
            self.push_valid();
        } else {
            self.push_null();
        }
    }
}

impl<M> TryExtendFromSelf for MutableFixedSizeListArray<M>
where
    M: MutableArray + TryExtendFromSelf,
{
    fn try_extend_from_self(&mut self, other: &Self) -> Result<()> {
        extend_validity(self.len(), &mut self.validity, &other.validity);

        self.values.try_extend_from_self(&other.values)
    }
}
