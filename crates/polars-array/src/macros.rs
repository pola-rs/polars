//! Macros over the array types of this crate.

/// Implements the inherent methods every array of this crate shares.
///
/// These are the methods whose body says nothing about the array they are on: the ones that read
/// the validity mask, the ones that replace it, and the ones that slice or repeat an array
/// through the methods that do the work — `slice_unchecked`, `new_from_index_unchecked` and
/// `new_full_null`, which every array writes for itself.
///
/// The element accessors are written as well where the value type of the elements is given; an
/// array whose elements carry no value of their own leaves it out.
///
/// The generic parameters of a generic array go in brackets before it.
macro_rules! impl_array_methods {
    ([$($generics:tt)*] $array:ty, $value:ty $(,)?) => {
        $crate::impl_array_methods!([$($generics)*] $array);

        impl<$($generics)*> $array {
            /// Returns the element at `i`, or `None` if it is null.
            #[inline]
            pub fn get(&self, i: usize) -> Option<$value> {
                assert!(i < self.len(), "index out of bounds");
                // SAFETY: `i` is in bounds of the array.
                unsafe { self.get_unchecked(i) }
            }

            /// Returns the element at `i`, or `None` if it is null.
            ///
            /// # Safety
            /// `i` must be smaller than `self.len()`.
            #[inline]
            pub unsafe fn get_unchecked(&self, i: usize) -> Option<$value> {
                // SAFETY: `i` is in bounds of the array, per the caller.
                unsafe { self.is_valid_unchecked(i).then(|| self.value_unchecked(i)) }
            }
        }
    };
    ([$($generics:tt)*] $array:ty $(,)?) => {
        impl<$($generics)*> $array {
            /// Returns whether the element at `i` is valid (non-null).
            #[inline]
            pub fn is_valid(&self, i: usize) -> bool {
                assert!(i < self.len(), "index out of bounds");
                // SAFETY: `i` is in bounds of the array.
                unsafe { self.is_valid_unchecked(i) }
            }

            /// Returns whether the element at `i` is valid (non-null).
            ///
            /// # Safety
            /// `i` must be smaller than `self.len()`.
            #[inline]
            pub unsafe fn is_valid_unchecked(&self, i: usize) -> bool {
                debug_assert!(i < self.len());
                // SAFETY: `i` is in bounds of the array, and therefore of its validity mask.
                self.validity()
                    .is_none_or(|validity| unsafe { validity.get_unchecked(i) })
            }

            /// Returns whether the element at `i` is null.
            #[inline]
            pub fn is_null(&self, i: usize) -> bool {
                !self.is_valid(i)
            }

            /// Returns whether the element at `i` is null.
            ///
            /// # Safety
            /// `i` must be smaller than `self.len()`.
            #[inline]
            pub unsafe fn is_null_unchecked(&self, i: usize) -> bool {
                // SAFETY: `i` is in bounds of the array, per the caller.
                unsafe { !self.is_valid_unchecked(i) }
            }

            /// The number of null elements.
            #[inline]
            pub fn null_count(&self) -> usize {
                self.validity().map_or(0, |validity| validity.unset_bits())
            }

            /// Whether this array has at least one null element.
            #[inline]
            pub fn has_nulls(&self) -> bool {
                self.null_count() > 0
            }

            /// Returns this array with its validity mask replaced.
            #[must_use]
            pub fn with_validity(mut self, validity: Option<$crate::PlBitmap>) -> Self {
                self.set_validity(validity);
                self
            }

            /// Replaces the validity mask, which keeps the representation it is in.
            pub fn set_validity(&mut self, validity: Option<$crate::PlBitmap>) {
                let length = self.len();
                self.validity = $crate::broadcast::validity_covering(validity, length);
            }

            /// Drops the validity mask, making every element valid.
            #[must_use]
            pub fn without_validity(mut self) -> Self {
                self.validity = None;
                self
            }

            /// Slices this array in place to `length` elements starting at `offset`.
            pub fn slice(&mut self, offset: usize, length: usize) {
                assert!(
                    offset + length <= self.len(),
                    "the offset of the new slice must be smaller than the length of the array",
                );
                // SAFETY: the slice is in bounds of the array.
                unsafe { self.slice_unchecked(offset, length) }
            }

            /// Returns this array sliced to `length` elements starting at `offset`.
            #[must_use]
            pub fn sliced(&self, offset: usize, length: usize) -> Self {
                let mut sliced = self.clone();
                sliced.slice(offset, length);
                sliced
            }

            /// Returns this array sliced to `length` elements starting at `offset`.
            ///
            /// # Safety
            /// `offset + length` must not exceed `self.len()`.
            #[must_use]
            pub unsafe fn sliced_unchecked(&self, offset: usize, length: usize) -> Self {
                let mut sliced = self.clone();
                // SAFETY: the slice is in bounds of the array, per the caller.
                unsafe { sliced.slice_unchecked(offset, length) };
                sliced
            }

            /// Creates an array of `length` copies of the element at `index`.
            #[inline]
            pub fn new_from_index(&self, index: usize, length: usize) -> Self {
                assert!(index < self.len(), "index out of bounds");
                // SAFETY: `index` is in bounds of the array.
                unsafe { self.new_from_index_unchecked(index, length) }
            }
        }
    };
    ($array:ty $(, $value:ty)? $(,)?) => {
        $crate::impl_array_methods!([] $array $(, $value)?);
    };
}
pub(crate) use impl_array_methods;

/// Implements [`PlArray`](crate::PlArray) for an array of this crate.
///
/// Every method of the trait object forwards to the inherent method of the same name, which is
/// where an array's own documentation of it lives. Two cannot: `array_type` names the variant
/// this array is, and `new_full_null_like_self` forwards to the inherent `new_full_null`. A
/// trailing method overrides the one the macro would write, which is how an array whose full-null
/// constructor takes a shape — a width, a values array, fields — states what that shape is.
///
/// The generic parameters of a generic array go in brackets before it.
macro_rules! impl_pl_array {
    ([$($generics:tt)*] $array:ty, $array_type:expr $(,)?) => {
        $crate::impl_pl_array!(
            [$($generics)*] $array, $array_type,
            fn new_full_null_like_self(&self, length: usize) -> Box<dyn $crate::PlArray> {
                Box::new(Self::new_full_null(length))
            }
        );
    };
    ([$($generics:tt)*] $array:ty, $array_type:expr, $($method:item)+) => {
        impl<$($generics)*> $crate::PlArray for $array {
            #[inline]
            fn as_any(&self) -> &dyn ::std::any::Any {
                self
            }

            #[inline]
            fn as_any_mut(&mut self) -> &mut dyn ::std::any::Any {
                self
            }

            #[inline]
            fn array_type(&self) -> $crate::PlArrayType {
                $array_type
            }

            #[inline]
            fn len(&self) -> usize {
                self.len()
            }

            #[inline]
            fn is_scalar(&self) -> bool {
                self.is_scalar()
            }

            #[inline]
            fn validity(&self) -> Option<$crate::PlBitmapRef<'_>> {
                self.validity()
            }

            #[inline]
            fn slice(&mut self, offset: usize, length: usize) {
                self.slice(offset, length)
            }

            #[inline]
            unsafe fn slice_unchecked(&mut self, offset: usize, length: usize) {
                // SAFETY: the caller keeps the slice in bounds.
                unsafe { self.slice_unchecked(offset, length) }
            }

            #[inline]
            fn set_validity(&mut self, validity: Option<$crate::PlBitmap>) {
                self.set_validity(validity)
            }

            #[inline]
            unsafe fn new_from_index_unchecked(
                &self,
                index: usize,
                length: usize,
            ) -> Box<dyn $crate::PlArray> {
                // SAFETY: the caller keeps `index` in bounds.
                Box::new(unsafe { self.new_from_index_unchecked(index, length) })
            }

            #[inline]
            fn to_boxed(&self) -> Box<dyn $crate::PlArray> {
                Box::new(self.clone())
            }

            fn eq_dyn(&self, other: &dyn $crate::PlArray) -> bool {
                other
                    .as_any()
                    .downcast_ref::<Self>()
                    .is_some_and(|other| self == other)
            }

            $($method)+
        }
    };
    ($array:ty, $($rest:tt)*) => {
        $crate::impl_pl_array!([] $array, $($rest)*);
    };
}
pub(crate) use impl_pl_array;

/// Implements [`StaticArray`](crate::StaticArray) for an array of this crate.
///
/// The associated types come first, then the two methods that describe the shape of an array —
/// `builder_like` and `new_full_null` — which every array states for itself. The rest of the
/// trait is written by the macro: each method forwards to the inherent method of the same name,
/// which is where an array's own documentation of it lives. Any further method the call site
/// writes is added to the impl as it stands.
///
/// The generic parameters of a generic array go in brackets before it.
macro_rules! impl_static_array {
    ($(#[$meta:meta])* [$($generics:tt)*] $array:ty, $($item:item)*) => {
        $(#[$meta])*
        impl<$($generics)*> $crate::StaticArray for $array {
            $($item)*

            #[inline]
            unsafe fn value_unchecked(&self, i: usize) -> Self::ValueT<'_> {
                // SAFETY: the caller keeps `i` in bounds.
                unsafe { self.value_unchecked(i) }
            }

            #[inline]
            unsafe fn get_unchecked(&self, i: usize) -> Option<Self::ValueT<'_>> {
                // SAFETY: the caller keeps `i` in bounds.
                unsafe { self.get_unchecked(i) }
            }

            #[inline]
            fn values_iter(&self) -> Self::ValueIterT<'_> {
                self.values_iter()
            }

            #[inline]
            fn iter(&self) -> Self::IterT<'_> {
                self.iter()
            }

            #[inline]
            fn broadcast_values_iter(&self, length: usize) -> Self::ValueIterT<'_> {
                self.broadcast_values_iter(length)
            }

            #[inline]
            fn with_validity_typed(self, validity: Option<$crate::PlBitmap>) -> Self {
                self.with_validity(validity)
            }

            #[inline]
            fn new_from_index_typed(&self, index: usize, length: usize) -> Self {
                self.new_from_index(index, length)
            }

            #[inline]
            fn is_flat(&self) -> bool {
                self.is_flat()
            }

            #[inline]
            fn to_flat(&self) -> ::std::borrow::Cow<'_, $crate::flat::Flat<Self>> {
                self.to_flat()
            }

            #[inline]
            fn as_flat(&self) -> Option<&$crate::flat::Flat<Self>> {
                self.as_flat()
            }
        }
    };
    ($(#[$meta:meta])* $array:ty, $($rest:tt)*) => {
        $crate::impl_static_array!($(#[$meta])* [] $array, $($rest)*);
    };
}
pub(crate) use impl_static_array;

/// Implements [`PartialEq`], [`Eq`] and the comparison against a flat array for an array.
///
/// Two arrays are equal when they are of the same length and shape — a width, say, which the
/// `shape` closure compares where an array has one — under masks that make the same elements
/// null, and when the `elements` closure finds the elements they leave equal. The elements are
/// only reached where the masks leave one to compare, so neither closure is asked about an array
/// that is null throughout, and a scalar array is compared as the one element it stands for.
macro_rules! impl_array_eq {
    ($array:ty, |$lhs:ident, $rhs:ident| $elements:expr $(,)?) => {
        $crate::impl_array_eq!(
            $array,
            shape: |_lhs, _rhs| true,
            |$lhs, $rhs| $elements,
        );
    };
    (
        $array:ty,
        shape: |$shape_lhs:ident, $shape_rhs:ident| $shape:expr,
        |$lhs:ident, $rhs:ident| $elements:expr $(,)?
    ) => {
        impl PartialEq for $array {
            fn eq(&self, other: &Self) -> bool {
                let shape = |$shape_lhs: &Self, $shape_rhs: &Self| $shape;
                if self.len() != other.len() || !shape(self, other) {
                    return false;
                }

                if !$crate::bitmap::validity_eq(self.validity(), other.validity(), self.len()) {
                    return false;
                }

                // Every element is null on both sides, so every value is undetermined and there
                // is nothing left to compare. This is also what keeps comparing two fully null
                // scalar arrays `O(1)`.
                if self.len() > 0 && self.null_count() == self.len() {
                    return true;
                }

                // Never walk two scalar arrays element by element: their length is unbounded by
                // their memory use. Comparing the one element they each stand for costs that
                // element.
                if let (Some(lhs), Some(rhs)) = (self.scalar_value(), other.scalar_value()) {
                    return lhs == rhs;
                }

                let elements = |$lhs: &Self, $rhs: &Self| $elements;
                elements(self, other)
            }
        }

        impl Eq for $array {}

        /// Compares an array of unknown representation against a flat one.
        impl PartialEq<$crate::flat::Flat<$array>> for $array {
            #[inline]
            fn eq(&self, other: &$crate::flat::Flat<$array>) -> bool {
                *self == *other.as_array()
            }
        }
    };
}
pub(crate) use impl_array_eq;

/// Implements [`Debug`](std::fmt::Debug) for an array, as the list of elements it holds.
///
/// Nulls render as `null`, and an array that is scalar throughout renders as the one element it
/// stands for and the number of times it stands for it: a length unbounded by the memory use is
/// never materialized.
///
/// The generic parameters of a generic array go in brackets before it.
macro_rules! impl_element_debug {
    ([$($generics:tt)*] $array:ty, $name:literal $(,)?) => {
        impl<$($generics)*> ::std::fmt::Debug for $array {
            fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                /// Renders nulls as `null` instead of `None`.
                struct Element<V>(Option<V>);

                impl<V: ::std::fmt::Debug> ::std::fmt::Debug for Element<V> {
                    fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                        match &self.0 {
                            Some(value) => value.fmt(f),
                            None => f.write_str("null"),
                        }
                    }
                }

                f.write_str($name)?;

                // Never materialize a scalar array: its length is unbounded by its memory use.
                if self.len() > 1 {
                    if let Some(element) = self.scalar_value() {
                        return write!(f, "[{:?}; {}]", Element(element), self.len());
                    }
                }

                f.debug_list().entries(self.iter().map(Element)).finish()
            }
        }
    };
    ($array:ty, $($rest:tt)*) => {
        $crate::impl_element_debug!([] $array, $($rest)*);
    };
}
pub(crate) use impl_element_debug;

/// Implements [`IntoIterator`] for a reference to an array, over the iterator it hands out.
///
/// The generic parameters of a generic array go in brackets before it, and the lifetime the
/// elements borrow for is `'a`.
macro_rules! impl_into_iterator {
    ([$($generics:tt)*] $array:ty, $iter:ty $(,)?) => {
        impl<'a, $($generics)*> IntoIterator for &'a $array {
            type Item = <$iter as Iterator>::Item;
            type IntoIter = $iter;

            #[inline]
            fn into_iter(self) -> Self::IntoIter {
                self.iter()
            }
        }
    };
    ($array:ty, $($rest:tt)*) => {
        $crate::impl_into_iterator!([] $array, $($rest)*);
    };
}
pub(crate) use impl_into_iterator;

/// Implements the iterator traits for an iterator over the optional elements of an array.
///
/// The iterator must hold its values in a `values` field, whose items are the values of the
/// elements, and the mask that says which of them are elements in a `validity` field, and must
/// have a `split` method that hands the two of them over to be walked in one loop.
///
/// The generic parameters of a generic iterator go in brackets before it, and the lifetime the
/// elements borrow for is `'a`.
macro_rules! impl_optional_iter {
    ($(#[$meta:meta])* [$($generics:tt)*] $iter:ty, $item:ty $(,)?) => {
        $(#[$meta])*
        impl<'a, $($generics)*> Iterator for $iter {
            type Item = Option<$item>;

            #[inline]
            fn next(&mut self) -> Option<Self::Item> {
                let value = self.values.next()?;
                Some(self.validity.next().then_some(value))
            }

            #[inline]
            fn nth(&mut self, n: usize) -> Option<Self::Item> {
                // The mask is advanced alongside the values, whether or not there is a value left.
                let is_valid = self.validity.nth(n);
                let value = self.values.nth(n)?;
                Some(is_valid.then_some(value))
            }

            #[inline]
            fn size_hint(&self) -> (usize, Option<usize>) {
                self.values.size_hint()
            }

            #[inline]
            fn count(self) -> usize {
                self.values.count()
            }

            /// Walks to the last element from the back, rather than through every one before it.
            #[inline]
            fn last(mut self) -> Option<Self::Item> {
                self.next_back()
            }

            /// Hoists the validity mask out of the loop, and the representation of the values
            /// with it.
            #[inline]
            fn fold<B, F>(self, init: B, f: F) -> B
            where
                F: FnMut(B, Self::Item) -> B,
            {
                let (values, mask) = self.split();
                // SAFETY: the mask has one bit per element, and the values and the mask are
                // walked in lockstep, so it has a bit for every value left to yield.
                unsafe { mask.fold_values(values, init, f) }
            }
        }

        impl<'a, $($generics)*> DoubleEndedIterator for $iter {
            #[inline]
            fn next_back(&mut self) -> Option<Self::Item> {
                let value = self.values.next_back()?;
                Some(self.validity.next_back().then_some(value))
            }

            #[inline]
            fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
                // The mask is advanced alongside the values, whether or not there is a value left.
                let is_valid = self.validity.nth_back(n);
                let value = self.values.nth_back(n)?;
                Some(is_valid.then_some(value))
            }

            /// Hoists the validity mask out of the loop, the way [`Iterator::fold`] does.
            #[inline]
            fn rfold<B, F>(self, init: B, f: F) -> B
            where
                F: FnMut(B, Self::Item) -> B,
            {
                let (values, mask) = self.split();
                // SAFETY: the mask has a bit for every value left to yield, per `Iterator::fold`.
                unsafe { mask.rfold_values(values, init, f) }
            }
        }

        impl<'a, $($generics)*> ExactSizeIterator for $iter {
            #[inline]
            fn len(&self) -> usize {
                self.values.len()
            }
        }

        // SAFETY: the values are trusted to yield as many elements as they say they will, and the
        // mask is walked alongside them.
        unsafe impl<'a, $($generics)*> ::arrow::trusted_len::TrustedLen for $iter {}
    };
    ($(#[$meta:meta])* $iter:ty, $($rest:tt)*) => {
        $crate::impl_optional_iter!($(#[$meta])* [] $iter, $($rest)*);
    };
}
pub(crate) use impl_optional_iter;

/// Implements the iterator traits for a newtype over another iterator, mapping every item.
///
/// The iterator must be a newtype whose one field is the iterator whose items it maps, and the
/// lifetime the items borrow for is `'a`. The map runs once per item yielded, in a fold as well
/// as one item at a time, which leaves the representation of the iterator underneath hoisted out
/// of the loop.
macro_rules! impl_mapped_iter {
    (
        $(#[$meta:meta])* [$($generics:tt)*] $iter:ty, $item:ty, |$value:ident| $map:expr $(,)?
    ) => {
        $(#[$meta])*
        impl<'a, $($generics)*> Iterator for $iter {
            type Item = $item;

            #[inline]
            fn next(&mut self) -> Option<Self::Item> {
                self.0.next().map(|$value| $map)
            }

            #[inline]
            fn nth(&mut self, n: usize) -> Option<Self::Item> {
                self.0.nth(n).map(|$value| $map)
            }

            #[inline]
            fn size_hint(&self) -> (usize, Option<usize>) {
                self.0.size_hint()
            }

            #[inline]
            fn count(self) -> usize {
                self.0.count()
            }

            #[inline]
            fn last(self) -> Option<Self::Item> {
                self.0.last().map(|$value| $map)
            }

            /// Folds the iterator underneath, which hoists its representation out of the loop.
            #[inline]
            fn fold<B, F>(self, init: B, mut f: F) -> B
            where
                F: FnMut(B, Self::Item) -> B,
            {
                self.0.fold(init, |acc, $value| f(acc, $map))
            }
        }

        impl<'a, $($generics)*> DoubleEndedIterator for $iter {
            #[inline]
            fn next_back(&mut self) -> Option<Self::Item> {
                self.0.next_back().map(|$value| $map)
            }

            #[inline]
            fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
                self.0.nth_back(n).map(|$value| $map)
            }

            /// Folds the iterator underneath, the way [`Iterator::fold`] does.
            #[inline]
            fn rfold<B, F>(self, init: B, mut f: F) -> B
            where
                F: FnMut(B, Self::Item) -> B,
            {
                self.0.rfold(init, |acc, $value| f(acc, $map))
            }
        }

        impl<'a, $($generics)*> ExactSizeIterator for $iter {
            #[inline]
            fn len(&self) -> usize {
                self.0.len()
            }
        }

        // SAFETY: the iterator underneath is trusted, and mapping its items does not change how
        // many there are.
        unsafe impl<'a, $($generics)*> ::arrow::trusted_len::TrustedLen for $iter {}
    };
    ($(#[$meta:meta])* $iter:ty, $($rest:tt)*) => {
        $crate::impl_mapped_iter!($(#[$meta])* [] $iter, $($rest)*);
    };
}
pub(crate) use impl_mapped_iter;

/// Runs a body with `T` bound to the element type of a [`crate::PlPrimitiveArray`].
#[macro_export]
macro_rules! with_match_pl_primitive_array_type {
    ($array:expr, |$T:ident| $body:expr $(,)?) => {{
        use ::arrow::array::View;
        use ::arrow::types::{days_ms, i256, months_days_ns};
        use ::polars_utils::float16::pf16;

        // `NativeType` is a sealed trait, so this list of element types is exhaustive.
        $crate::__with_match_pl_primitive_array_type__! {
            $array,
            [
                i8, i16, i32, i64, i128, i256,
                u8, u16, u32, u64, u128,
                pf16, f32, f64,
                days_ms, months_days_ns, View,
            ],
            $T,
            $body
        }
    }};
}

/// The body of [`with_match_pl_primitive_array_type`], binding `$T` to the element type.
#[doc(hidden)]
#[macro_export]
macro_rules! __with_match_pl_primitive_array_type__ {(
    $array:expr, [$($element:ty),* $(,)?], $T:ident, $body:expr
) => ({
    let array: &dyn $crate::PlArray = $array;
    $(if array.as_any().is::<$crate::PlPrimitiveArray<$element>>() {
        Some({
            #[allow(dead_code)]
            type $T = $element;
            $body
        })
    } else)* {
        None
    }
})}

#[cfg(test)]
mod tests {
    use arrow::array::View;
    use arrow::types::{days_ms, i256, months_days_ns};
    use polars_utils::float16::pf16;

    use crate::{
        PlArray, PlBinaryArray, PlFixedSizeBinaryArray, PlFixedSizeListArray, PlListArray,
        PlNullArray, PlPrimitiveArray, PlStructArray,
    };

    /// Whether the body runs with `T` bound to the element type of an array of `T`.
    fn dispatches<T: arrow::types::NativeType>() -> bool {
        let array: Box<dyn PlArray> = Box::new(PlPrimitiveArray::<T>::new_empty());
        with_match_pl_primitive_array_type!(&*array, |E| {
            std::any::TypeId::of::<E>() == std::any::TypeId::of::<T>()
        })
        .unwrap()
    }

    #[test]
    fn every_element_type_is_dispatched_to() {
        assert!(dispatches::<i8>());
        assert!(dispatches::<i16>());
        assert!(dispatches::<i32>());
        assert!(dispatches::<i64>());
        assert!(dispatches::<i128>());
        assert!(dispatches::<i256>());
        assert!(dispatches::<u8>());
        assert!(dispatches::<u16>());
        assert!(dispatches::<u32>());
        assert!(dispatches::<u64>());
        assert!(dispatches::<u128>());
        assert!(dispatches::<pf16>());
        assert!(dispatches::<f32>());
        assert!(dispatches::<f64>());
        assert!(dispatches::<days_ms>());
        assert!(dispatches::<months_days_ns>());
        assert!(dispatches::<View>());
    }

    #[test]
    fn arrays_that_are_not_primitive_have_no_element_type() {
        let arrays: [Box<dyn PlArray>; 6] = [
            Box::new(PlNullArray::new(1)),
            Box::new(PlBinaryArray::new_empty()),
            Box::new(PlFixedSizeBinaryArray::new_empty(2)),
            Box::new(PlListArray::new_empty(Box::new(
                PlPrimitiveArray::<i32>::new_empty(),
            ))),
            Box::new(PlFixedSizeListArray::new_empty(
                Box::new(PlPrimitiveArray::<i32>::new_empty()),
                2,
            )),
            Box::new(PlStructArray::new_empty()),
        ];
        for array in &arrays {
            assert_eq!(
                with_match_pl_primitive_array_type!(&**array, |T| size_of::<T>()),
                None,
            );
        }
    }
}
