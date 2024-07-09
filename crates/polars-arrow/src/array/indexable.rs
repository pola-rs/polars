use std::borrow::Borrow;

use crate::array::{
    MutableArray, MutableBinaryArray, MutableBinaryValuesArray, MutableBinaryViewArray,
    MutableBooleanArray, MutableFixedSizeBinaryArray, MutablePrimitiveArray, MutableUtf8Array,
    MutableUtf8ValuesArray, ViewType,
};
use crate::offset::Offset;
use crate::types::NativeType;

/// Trait for arrays that can be indexed directly to extract a value.
pub trait Indexable {
    /// The type of the element at index `i`; may be a reference type or a value type.
    type Value<'a>: Borrow<Self::Type>
    where
        Self: 'a;

    type Type: ?Sized;

    /// Returns the element at index `i`.
    /// # Panic
    /// May panic if `i >= self.len()`.
    fn value_at(&self, index: usize) -> Self::Value<'_>;

    /// Returns the element at index `i`.
    ///
    /// # Safety
    /// Assumes that the `i < self.len`.
    #[inline]
    unsafe fn value_unchecked_at(&self, index: usize) -> Self::Value<'_> {
        self.value_at(index)
    }
}

pub trait AsIndexed<M: Indexable> {
    fn as_indexed(&self) -> &M::Type;
}

impl Indexable for MutableBooleanArray {
    type Value<'a> = bool;
    type Type = bool;

    #[inline]
    fn value_at(&self, i: usize) -> Self::Value<'_> {
        self.values().get(i)
    }
}

impl AsIndexed<MutableBooleanArray> for bool {
    #[inline]
    fn as_indexed(&self) -> &bool {
        self
    }
}

impl<O: Offset> Indexable for MutableBinaryArray<O> {
    type Value<'a> = &'a [u8];
    type Type = [u8];

    #[inline]
    fn value_at(&self, i: usize) -> Self::Value<'_> {
        // TODO: add .value() / .value_unchecked() to MutableBinaryArray?
        assert!(i < self.len());
        unsafe { self.value_unchecked_at(i) }
    }

    #[inline]
    unsafe fn value_unchecked_at(&self, i: usize) -> Self::Value<'_> {
        // TODO: add .value() / .value_unchecked() to MutableBinaryArray?
        // soundness: the invariant of the function
        let (start, end) = self.offsets().start_end_unchecked(i);
        // soundness: the invariant of the struct
        self.values().get_unchecked(start..end)
    }
}

impl<O: Offset> AsIndexed<MutableBinaryArray<O>> for &[u8] {
    #[inline]
    fn as_indexed(&self) -> &[u8] {
        self
    }
}

impl<O: Offset> Indexable for MutableBinaryValuesArray<O> {
    type Value<'a> = &'a [u8];
    type Type = [u8];

    #[inline]
    fn value_at(&self, i: usize) -> Self::Value<'_> {
        self.value(i)
    }

    #[inline]
    unsafe fn value_unchecked_at(&self, i: usize) -> Self::Value<'_> {
        self.value_unchecked(i)
    }
}

impl<O: Offset> AsIndexed<MutableBinaryValuesArray<O>> for &[u8] {
    #[inline]
    fn as_indexed(&self) -> &[u8] {
        self
    }
}

impl Indexable for MutableFixedSizeBinaryArray {
    type Value<'a> = &'a [u8];
    type Type = [u8];

    #[inline]
    fn value_at(&self, i: usize) -> Self::Value<'_> {
        self.value(i)
    }

    #[inline]
    unsafe fn value_unchecked_at(&self, i: usize) -> Self::Value<'_> {
        // soundness: the invariant of the struct
        self.value_unchecked(i)
    }
}

impl AsIndexed<MutableFixedSizeBinaryArray> for &[u8] {
    #[inline]
    fn as_indexed(&self) -> &[u8] {
        self
    }
}

impl<T: ViewType + ?Sized> Indexable for MutableBinaryViewArray<T> {
    type Value<'a> = &'a T;
    type Type = T;

    fn value_at(&self, index: usize) -> Self::Value<'_> {
        self.value(index)
    }

    unsafe fn value_unchecked_at(&self, index: usize) -> Self::Value<'_> {
        self.value_unchecked(index)
    }
}

impl<T: ViewType + ?Sized> AsIndexed<MutableBinaryViewArray<T>> for &T {
    #[inline]
    fn as_indexed(&self) -> &T {
        self
    }
}

// TODO: should NativeType derive from Hash?
impl<T: NativeType> Indexable for MutablePrimitiveArray<T> {
    type Value<'a> = T;
    type Type = T;

    #[inline]
    fn value_at(&self, i: usize) -> Self::Value<'_> {
        assert!(i < self.len());
        // TODO: add Length trait? (for both Array and MutableArray)
        unsafe { self.value_unchecked_at(i) }
    }

    #[inline]
    unsafe fn value_unchecked_at(&self, i: usize) -> Self::Value<'_> {
        *self.values().get_unchecked(i)
    }
}

impl<T: NativeType> AsIndexed<MutablePrimitiveArray<T>> for T {
    #[inline]
    fn as_indexed(&self) -> &T {
        self
    }
}

impl<O: Offset> Indexable for MutableUtf8Array<O> {
    type Value<'a> = &'a str;
    type Type = str;

    #[inline]
    fn value_at(&self, i: usize) -> Self::Value<'_> {
        self.value(i)
    }

    #[inline]
    unsafe fn value_unchecked_at(&self, i: usize) -> Self::Value<'_> {
        self.value_unchecked(i)
    }
}

impl<O: Offset, V: AsRef<str>> AsIndexed<MutableUtf8Array<O>> for V {
    #[inline]
    fn as_indexed(&self) -> &str {
        self.as_ref()
    }
}

impl<O: Offset> Indexable for MutableUtf8ValuesArray<O> {
    type Value<'a> = &'a str;
    type Type = str;

    #[inline]
    fn value_at(&self, i: usize) -> Self::Value<'_> {
        self.value(i)
    }

    #[inline]
    unsafe fn value_unchecked_at(&self, i: usize) -> Self::Value<'_> {
        self.value_unchecked(i)
    }
}

impl<O: Offset, V: AsRef<str>> AsIndexed<MutableUtf8ValuesArray<O>> for V {
    #[inline]
    fn as_indexed(&self) -> &str {
        self.as_ref()
    }
}
