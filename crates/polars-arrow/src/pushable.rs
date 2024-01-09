use crate::array::{MutableBinaryViewArray, MutablePrimitiveArray, ViewType};
use crate::bitmap::MutableBitmap;
use crate::offset::{Offset, Offsets};
use crate::types::NativeType;

/// A private trait representing structs that can receive elements.
pub trait Pushable<T>: Sized + Default {
    fn with_capacity(capacity: usize) -> Self {
        let mut new = Self::default();
        new.reserve(capacity);
        new
    }
    fn reserve(&mut self, additional: usize);
    fn push(&mut self, value: T);
    fn len(&self) -> usize;
    fn push_null(&mut self);
    fn extend_constant(&mut self, additional: usize, value: T);
}

impl Pushable<bool> for MutableBitmap {
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
}

impl<T: Copy + Default> Pushable<T> for Vec<T> {
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
    fn extend_constant(&mut self, additional: usize, value: T) {
        self.resize(self.len() + additional, value);
    }
}
impl<O: Offset> Pushable<usize> for Offsets<O> {
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
}

impl<T: NativeType> Pushable<Option<T>> for MutablePrimitiveArray<T> {
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
}

impl<T: ViewType + ?Sized> Pushable<&T> for MutableBinaryViewArray<T> {
    #[inline]
    fn reserve(&mut self, additional: usize) {
        MutableBinaryViewArray::reserve(self, additional)
    }

    #[inline]
    fn push(&mut self, value: &T) {
        MutableBinaryViewArray::push_value(self, value)
    }

    #[inline]
    fn len(&self) -> usize {
        MutableBinaryViewArray::len(self)
    }

    fn push_null(&mut self) {
        MutableBinaryViewArray::push_null(self)
    }

    fn extend_constant(&mut self, additional: usize, value: &T) {
        // First push a value to get the View
        MutableBinaryViewArray::push_value(self, value);

        // And then use that new view to extend
        let views = self.views();
        let view = *views.last().unwrap();

        let remaining = additional - 1;
        for _ in 0..remaining {
            views.push(view);
        }

        if let Some(bitmap) = self.validity() {
            bitmap.extend_constant(remaining, true)
        }
    }
}

