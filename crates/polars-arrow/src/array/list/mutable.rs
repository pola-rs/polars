use std::sync::Arc;

use polars_error::{polars_err, PolarsResult};

use super::ListArray;
use crate::array::physical_binary::extend_validity;
use crate::array::{Array, MutableArray, TryExtend, TryExtendFromSelf, TryPush};
use crate::bitmap::MutableBitmap;
use crate::datatypes::{ArrowDataType, Field};
use crate::offset::{Offset, Offsets};
use crate::trusted_len::TrustedLen;

/// The mutable version of [`ListArray`].
#[derive(Debug, Clone)]
pub struct MutableListArray<O: Offset, M: MutableArray> {
    data_type: ArrowDataType,
    offsets: Offsets<O>,
    values: M,
    validity: Option<MutableBitmap>,
}

impl<O: Offset, M: MutableArray + Default> MutableListArray<O, M> {
    /// Creates a new empty [`MutableListArray`].
    pub fn new() -> Self {
        let values = M::default();
        let data_type = ListArray::<O>::default_datatype(values.data_type().clone());
        Self::new_from(values, data_type, 0)
    }

    /// Creates a new [`MutableListArray`] with a capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        let values = M::default();
        let data_type = ListArray::<O>::default_datatype(values.data_type().clone());

        let offsets = Offsets::<O>::with_capacity(capacity);
        Self {
            data_type,
            offsets,
            values,
            validity: None,
        }
    }
}

impl<O: Offset, M: MutableArray + Default> Default for MutableListArray<O, M> {
    fn default() -> Self {
        Self::new()
    }
}

impl<O: Offset, M: MutableArray> From<MutableListArray<O, M>> for ListArray<O> {
    fn from(mut other: MutableListArray<O, M>) -> Self {
        ListArray::new(
            other.data_type,
            other.offsets.into(),
            other.values.as_box(),
            other.validity.map(|x| x.into()),
        )
    }
}

impl<O, M, I, T> TryExtend<Option<I>> for MutableListArray<O, M>
where
    O: Offset,
    M: MutableArray + TryExtend<Option<T>>,
    I: IntoIterator<Item = Option<T>>,
{
    fn try_extend<II: IntoIterator<Item = Option<I>>>(&mut self, iter: II) -> PolarsResult<()> {
        let iter = iter.into_iter();
        self.reserve(iter.size_hint().0);
        for items in iter {
            self.try_push(items)?;
        }
        Ok(())
    }
}

impl<O, M, I, T> TryPush<Option<I>> for MutableListArray<O, M>
where
    O: Offset,
    M: MutableArray + TryExtend<Option<T>>,
    I: IntoIterator<Item = Option<T>>,
{
    #[inline]
    fn try_push(&mut self, item: Option<I>) -> PolarsResult<()> {
        if let Some(items) = item {
            let values = self.mut_values();
            values.try_extend(items)?;
            self.try_push_valid()?;
        } else {
            self.push_null();
        }
        Ok(())
    }
}

impl<O, M> TryExtendFromSelf for MutableListArray<O, M>
where
    O: Offset,
    M: MutableArray + TryExtendFromSelf,
{
    fn try_extend_from_self(&mut self, other: &Self) -> PolarsResult<()> {
        extend_validity(self.len(), &mut self.validity, &other.validity);

        self.values.try_extend_from_self(&other.values)?;
        self.offsets.try_extend_from_self(&other.offsets)
    }
}

impl<O: Offset, M: MutableArray> MutableListArray<O, M> {
    /// Creates a new [`MutableListArray`] from a [`MutableArray`] and capacity.
    pub fn new_from(values: M, data_type: ArrowDataType, capacity: usize) -> Self {
        let offsets = Offsets::<O>::with_capacity(capacity);
        assert_eq!(values.len(), 0);
        ListArray::<O>::get_child_field(&data_type);
        Self {
            data_type,
            offsets,
            values,
            validity: None,
        }
    }

    /// Creates a new [`MutableListArray`] from a [`MutableArray`].
    pub fn new_with_field(values: M, name: &str, nullable: bool) -> Self {
        let field = Box::new(Field::new(name, values.data_type().clone(), nullable));
        let data_type = if O::IS_LARGE {
            ArrowDataType::LargeList(field)
        } else {
            ArrowDataType::List(field)
        };
        Self::new_from(values, data_type, 0)
    }

    /// Creates a new [`MutableListArray`] from a [`MutableArray`] and capacity.
    pub fn new_with_capacity(values: M, capacity: usize) -> Self {
        let data_type = ListArray::<O>::default_datatype(values.data_type().clone());
        Self::new_from(values, data_type, capacity)
    }

    /// Creates a new [`MutableListArray`] from a [`MutableArray`], [`Offsets`] and
    /// [`MutableBitmap`].
    pub fn new_from_mutable(
        values: M,
        offsets: Offsets<O>,
        validity: Option<MutableBitmap>,
    ) -> Self {
        assert_eq!(values.len(), offsets.last().to_usize());
        let data_type = ListArray::<O>::default_datatype(values.data_type().clone());
        Self {
            data_type,
            offsets,
            values,
            validity,
        }
    }

    #[inline]
    /// Needs to be called when a valid value was extended to this array.
    /// This is a relatively low level function, prefer `try_push` when you can.
    pub fn try_push_valid(&mut self) -> PolarsResult<()> {
        let total_length = self.values.len();
        let offset = self.offsets.last().to_usize();
        let length = total_length
            .checked_sub(offset)
            .ok_or_else(|| polars_err!(ComputeError: "overflow"))?;

        self.offsets.try_push(length)?;
        if let Some(validity) = &mut self.validity {
            validity.push(true)
        }
        Ok(())
    }

    #[inline]
    fn push_null(&mut self) {
        self.offsets.extend_constant(1);
        match &mut self.validity {
            Some(validity) => validity.push(false),
            None => self.init_validity(),
        }
    }

    /// Expand this array, using elements from the underlying backing array.
    /// Assumes the expansion begins at the highest previous offset, or zero if
    /// this [`MutableListArray`] is currently empty.
    ///
    /// Panics if:
    /// - the new offsets are not in monotonic increasing order.
    /// - any new offset is not in bounds of the backing array.
    /// - the passed iterator has no upper bound.
    pub fn try_extend_from_lengths<II>(&mut self, iterator: II) -> PolarsResult<()>
    where
        II: TrustedLen<Item = Option<usize>> + Clone,
    {
        self.offsets
            .try_extend_from_lengths(iterator.clone().map(|x| x.unwrap_or_default()))?;
        if let Some(validity) = &mut self.validity {
            validity.extend_from_trusted_len_iter(iterator.map(|x| x.is_some()))
        }
        assert_eq!(self.offsets.last().to_usize(), self.values.len());
        Ok(())
    }

    /// Returns the length of this array
    #[inline]
    pub fn len(&self) -> usize {
        self.offsets.len_proxy()
    }

    /// The values
    pub fn mut_values(&mut self) -> &mut M {
        &mut self.values
    }

    /// The offsets
    pub fn offsets(&self) -> &Offsets<O> {
        &self.offsets
    }

    /// The values
    pub fn values(&self) -> &M {
        &self.values
    }

    fn init_validity(&mut self) {
        let len = self.offsets.len_proxy();

        let mut validity = MutableBitmap::with_capacity(self.offsets.capacity());
        validity.extend_constant(len, true);
        validity.set(len - 1, false);
        self.validity = Some(validity)
    }

    /// Converts itself into an [`Array`].
    pub fn into_arc(self) -> Arc<dyn Array> {
        let a: ListArray<O> = self.into();
        Arc::new(a)
    }

    /// converts itself into [`Box<dyn Array>`]
    pub fn into_box(self) -> Box<dyn Array> {
        let a: ListArray<O> = self.into();
        Box::new(a)
    }

    /// Reserves `additional` slots.
    pub fn reserve(&mut self, additional: usize) {
        self.offsets.reserve(additional);
        if let Some(x) = self.validity.as_mut() {
            x.reserve(additional)
        }
    }

    /// Shrinks the capacity of the [`MutableListArray`] to fit its current length.
    pub fn shrink_to_fit(&mut self) {
        self.values.shrink_to_fit();
        self.offsets.shrink_to_fit();
        if let Some(validity) = &mut self.validity {
            validity.shrink_to_fit()
        }
    }
}

impl<O: Offset, M: MutableArray + 'static> MutableArray for MutableListArray<O, M> {
    fn len(&self) -> usize {
        MutableListArray::len(self)
    }

    fn validity(&self) -> Option<&MutableBitmap> {
        self.validity.as_ref()
    }

    fn as_box(&mut self) -> Box<dyn Array> {
        ListArray::new(
            self.data_type.clone(),
            std::mem::take(&mut self.offsets).into(),
            self.values.as_box(),
            std::mem::take(&mut self.validity).map(|x| x.into()),
        )
        .boxed()
    }

    fn as_arc(&mut self) -> Arc<dyn Array> {
        ListArray::new(
            self.data_type.clone(),
            std::mem::take(&mut self.offsets).into(),
            self.values.as_box(),
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

    #[inline]
    fn push_null(&mut self) {
        self.push_null()
    }

    fn reserve(&mut self, additional: usize) {
        self.reserve(additional)
    }

    fn shrink_to_fit(&mut self) {
        self.shrink_to_fit();
    }
}
