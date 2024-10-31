use crate::array::{
    BinaryViewArrayGeneric, BooleanArray, MutableBinaryViewArray, MutableBooleanArray,
    MutablePrimitiveArray, PrimitiveArray, ViewType,
};
use crate::bitmap::{Bitmap, MutableBitmap};
use crate::offset::{Offset, Offsets, OffsetsBuffer};
use crate::types::NativeType;

/// A private trait representing structs that can receive elements.
pub trait Pushable<T>: Sized + Default {
    type Freeze;

    fn with_capacity(capacity: usize) -> Self {
        let mut new = Self::default();
        new.reserve(capacity);
        new
    }
    fn reserve(&mut self, additional: usize);
    fn push(&mut self, value: T);
    fn len(&self) -> usize;
    fn push_null(&mut self);
    #[inline]
    fn extend_n(&mut self, n: usize, iter: impl Iterator<Item = T>) {
        for item in iter.take(n) {
            self.push(item);
        }
    }
    fn extend_constant(&mut self, additional: usize, value: T);
    fn extend_null_constant(&mut self, additional: usize);
    fn freeze(self) -> Self::Freeze;
}

impl Pushable<bool> for MutableBitmap {
    type Freeze = Bitmap;

    #[inline]
    fn reserve(&mut self, additional: usize) {
        MutableBitmap::reserve(self, additional)
    }

    #[inline]
    fn len(&self) -> usize {
        self.len()
    }

    #[inline]
    fn push(&mut self, value: bool) {
        self.push(value)
    }

    #[inline]
    fn push_null(&mut self) {
        self.push(false)
    }

    #[inline]
    fn extend_constant(&mut self, additional: usize, value: bool) {
        self.extend_constant(additional, value)
    }

    #[inline]
    fn extend_null_constant(&mut self, additional: usize) {
        self.extend_constant(additional, false)
    }

    fn freeze(self) -> Self::Freeze {
        self.into()
    }
}

impl<T: Copy + Default> Pushable<T> for Vec<T> {
    type Freeze = Vec<T>;
    #[inline]
    fn reserve(&mut self, additional: usize) {
        Vec::reserve(self, additional)
    }
    #[inline]
    fn len(&self) -> usize {
        self.len()
    }

    #[inline]
    fn push_null(&mut self) {
        self.push(T::default())
    }

    #[inline]
    fn push(&mut self, value: T) {
        self.push(value)
    }

    #[inline]
    fn extend_n(&mut self, n: usize, iter: impl Iterator<Item = T>) {
        self.extend(iter.take(n));
    }

    #[inline]
    fn extend_constant(&mut self, additional: usize, value: T) {
        self.resize(self.len() + additional, value);
    }

    #[inline]
    fn extend_null_constant(&mut self, additional: usize) {
        self.extend_constant(additional, T::default())
    }
    fn freeze(self) -> Self::Freeze {
        self
    }
}
impl<O: Offset> Pushable<usize> for Offsets<O> {
    type Freeze = OffsetsBuffer<O>;
    fn reserve(&mut self, additional: usize) {
        self.reserve(additional)
    }
    #[inline]
    fn len(&self) -> usize {
        self.len_proxy()
    }

    #[inline]
    fn push(&mut self, value: usize) {
        self.try_push(value).unwrap()
    }

    #[inline]
    fn push_null(&mut self) {
        self.extend_constant(1);
    }

    #[inline]
    fn extend_constant(&mut self, additional: usize, _: usize) {
        self.extend_constant(additional)
    }

    #[inline]
    fn extend_null_constant(&mut self, additional: usize) {
        self.extend_constant(additional)
    }
    fn freeze(self) -> Self::Freeze {
        self.into()
    }
}

impl<T: NativeType> Pushable<Option<T>> for MutablePrimitiveArray<T> {
    type Freeze = PrimitiveArray<T>;

    #[inline]
    fn reserve(&mut self, additional: usize) {
        MutablePrimitiveArray::reserve(self, additional)
    }

    #[inline]
    fn push(&mut self, value: Option<T>) {
        MutablePrimitiveArray::push(self, value)
    }

    #[inline]
    fn len(&self) -> usize {
        self.values().len()
    }

    #[inline]
    fn push_null(&mut self) {
        self.push(None)
    }

    #[inline]
    fn extend_constant(&mut self, additional: usize, value: Option<T>) {
        MutablePrimitiveArray::extend_constant(self, additional, value)
    }

    #[inline]
    fn extend_null_constant(&mut self, additional: usize) {
        MutablePrimitiveArray::extend_constant(self, additional, None)
    }
    fn freeze(self) -> Self::Freeze {
        self.into()
    }
}

pub trait NoOption {}
impl NoOption for &str {}
impl NoOption for &[u8] {}

impl<T, K> Pushable<T> for MutableBinaryViewArray<K>
where
    T: AsRef<K> + NoOption,
    K: ViewType + ?Sized,
{
    type Freeze = BinaryViewArrayGeneric<K>;

    #[inline]
    fn reserve(&mut self, additional: usize) {
        MutableBinaryViewArray::reserve(self, additional)
    }

    #[inline]
    fn push(&mut self, value: T) {
        MutableBinaryViewArray::push_value(self, value)
    }

    #[inline]
    fn len(&self) -> usize {
        MutableBinaryViewArray::len(self)
    }

    fn push_null(&mut self) {
        MutableBinaryViewArray::push_null(self)
    }

    fn extend_constant(&mut self, additional: usize, value: T) {
        MutableBinaryViewArray::extend_constant(self, additional, Some(value));
    }

    #[inline]
    fn extend_null_constant(&mut self, additional: usize) {
        self.extend_null(additional);
    }
    fn freeze(self) -> Self::Freeze {
        self.into()
    }
}

impl<T, K> Pushable<Option<T>> for MutableBinaryViewArray<K>
where
    T: AsRef<K>,
    K: ViewType + ?Sized,
{
    type Freeze = BinaryViewArrayGeneric<K>;
    #[inline]
    fn reserve(&mut self, additional: usize) {
        MutableBinaryViewArray::reserve(self, additional)
    }

    #[inline]
    fn push(&mut self, value: Option<T>) {
        MutableBinaryViewArray::push(self, value.as_ref())
    }

    #[inline]
    fn len(&self) -> usize {
        MutableBinaryViewArray::len(self)
    }

    fn push_null(&mut self) {
        MutableBinaryViewArray::push_null(self)
    }

    fn extend_constant(&mut self, additional: usize, value: Option<T>) {
        MutableBinaryViewArray::extend_constant(self, additional, value);
    }

    #[inline]
    fn extend_null_constant(&mut self, additional: usize) {
        self.extend_null(additional);
    }
    fn freeze(self) -> Self::Freeze {
        self.into()
    }
}

impl Pushable<bool> for MutableBooleanArray {
    type Freeze = BooleanArray;
    #[inline]
    fn reserve(&mut self, additional: usize) {
        MutableBooleanArray::reserve(self, additional)
    }

    #[inline]
    fn push(&mut self, value: bool) {
        MutableBooleanArray::push_value(self, value)
    }

    #[inline]
    fn len(&self) -> usize {
        self.values().len()
    }

    #[inline]
    fn push_null(&mut self) {
        unimplemented!()
    }

    #[inline]
    fn extend_constant(&mut self, additional: usize, value: bool) {
        MutableBooleanArray::extend_constant(self, additional, Some(value))
    }

    #[inline]
    fn extend_null_constant(&mut self, _additional: usize) {
        unimplemented!()
    }
    fn freeze(self) -> Self::Freeze {
        self.into()
    }
}

impl Pushable<Option<bool>> for MutableBooleanArray {
    type Freeze = BooleanArray;
    #[inline]
    fn reserve(&mut self, additional: usize) {
        MutableBooleanArray::reserve(self, additional)
    }

    #[inline]
    fn push(&mut self, value: Option<bool>) {
        MutableBooleanArray::push(self, value)
    }

    #[inline]
    fn len(&self) -> usize {
        self.values().len()
    }

    #[inline]
    fn push_null(&mut self) {
        MutableBooleanArray::push_null(self)
    }

    #[inline]
    fn extend_constant(&mut self, additional: usize, value: Option<bool>) {
        MutableBooleanArray::extend_constant(self, additional, value)
    }

    #[inline]
    fn extend_null_constant(&mut self, additional: usize) {
        MutableBooleanArray::extend_constant(self, additional, None)
    }
    fn freeze(self) -> Self::Freeze {
        self.into()
    }
}
