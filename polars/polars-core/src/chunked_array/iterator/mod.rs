use crate::datatypes::CategoricalChunked;
use crate::prelude::*;
use crate::utils::CustomIterTools;
use arrow::array::*;
use std::convert::TryFrom;
use std::ops::Deref;

type LargeStringArray = Utf8Array<i64>;
type LargeListArray = ListArray<i64>;

// If parallel feature is enable, then, activate the parallel module.
#[cfg(feature = "parallel")]
#[cfg_attr(docsrs, doc(cfg(feature = "parallel")))]
pub mod par;

/// A `PolarsIterator` is an iterator over a `ChunkedArray` which contains polars types. A `PolarsIterator`
/// must implement `ExactSizeIterator` and `DoubleEndedIterator`.
pub trait PolarsIterator:
    ExactSizeIterator + DoubleEndedIterator + Send + Sync + TrustedLen
{
}
unsafe impl<'a, I> TrustedLen for Box<dyn PolarsIterator<Item = I> + 'a> {}

/// Implement PolarsIterator for every iterator that implements the needed traits.
impl<T: ?Sized> PolarsIterator for T where
    T: ExactSizeIterator + DoubleEndedIterator + Send + Sync + TrustedLen
{
}

impl<'a, T> IntoIterator for &'a ChunkedArray<T>
where
    T: PolarsNumericType,
{
    type Item = Option<T::Native>;
    type IntoIter = Box<dyn PolarsIterator<Item = Self::Item> + 'a>;
    fn into_iter(self) -> Self::IntoIter {
        Box::new(
            self.downcast_iter()
                .flatten()
                .map(|x| x.copied())
                .trust_my_length(self.len()),
        )
    }
}

impl<'a> IntoIterator for &'a CategoricalChunked {
    type Item = Option<u32>;
    type IntoIter = Box<dyn PolarsIterator<Item = Self::Item> + 'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.deref().into_iter()
    }
}

impl<'a> IntoIterator for &'a BooleanChunked {
    type Item = Option<bool>;
    type IntoIter = Box<dyn PolarsIterator<Item = Self::Item> + 'a>;
    fn into_iter(self) -> Self::IntoIter {
        Box::new(self.downcast_iter().flatten().trust_my_length(self.len()))
    }
}

/// The no null iterator for a BooleanArray
pub struct BoolIterNoNull<'a> {
    array: &'a BooleanArray,
    current: usize,
    current_end: usize,
}

impl<'a> BoolIterNoNull<'a> {
    /// create a new iterator
    pub fn new(array: &'a BooleanArray) -> Self {
        BoolIterNoNull {
            array,
            current: 0,
            current_end: array.len(),
        }
    }
}

impl<'a> Iterator for BoolIterNoNull<'a> {
    type Item = bool;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current == self.current_end {
            None
        } else {
            let old = self.current;
            self.current += 1;
            unsafe { Some(self.array.value_unchecked(old)) }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (
            self.array.len() - self.current,
            Some(self.array.len() - self.current),
        )
    }
}

impl<'a> DoubleEndedIterator for BoolIterNoNull<'a> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.current_end == self.current {
            None
        } else {
            self.current_end -= 1;
            unsafe { Some(self.array.value_unchecked(self.current_end)) }
        }
    }
}

/// all arrays have known size.
impl<'a> ExactSizeIterator for BoolIterNoNull<'a> {}

impl BooleanChunked {
    #[allow(clippy::wrong_self_convention)]
    pub fn into_no_null_iter(
        &self,
    ) -> impl Iterator<Item = bool>
           + '_
           + Send
           + Sync
           + ExactSizeIterator
           + DoubleEndedIterator
           + TrustedLen {
        self.downcast_iter()
            .map(|bool_arr| BoolIterNoNull::new(bool_arr))
            .flatten()
            .trust_my_length(self.len())
    }
}

impl<'a> IntoIterator for &'a Utf8Chunked {
    type Item = Option<&'a str>;
    type IntoIter = Box<dyn PolarsIterator<Item = Self::Item> + 'a>;
    fn into_iter(self) -> Self::IntoIter {
        Box::new(self.downcast_iter().flatten().trust_my_length(self.len()))
    }
}

pub struct Utf8IterNoNull<'a> {
    array: &'a LargeStringArray,
    current: usize,
    current_end: usize,
}

impl<'a> Utf8IterNoNull<'a> {
    /// create a new iterator
    pub fn new(array: &'a LargeStringArray) -> Self {
        Utf8IterNoNull {
            array,
            current: 0,
            current_end: array.len(),
        }
    }
}

impl<'a> Iterator for Utf8IterNoNull<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current == self.current_end {
            None
        } else {
            let old = self.current;
            self.current += 1;
            unsafe { Some(self.array.value_unchecked(old)) }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (
            self.array.len() - self.current,
            Some(self.array.len() - self.current),
        )
    }
}

impl<'a> DoubleEndedIterator for Utf8IterNoNull<'a> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.current_end == self.current {
            None
        } else {
            self.current_end -= 1;
            unsafe { Some(self.array.value_unchecked(self.current_end)) }
        }
    }
}

/// all arrays have known size.
impl<'a> ExactSizeIterator for Utf8IterNoNull<'a> {}

impl Utf8Chunked {
    #[allow(clippy::wrong_self_convention)]
    pub fn into_no_null_iter<'a>(
        &'a self,
    ) -> impl Iterator<Item = &'a str>
           + '_
           + Send
           + Sync
           + ExactSizeIterator
           + DoubleEndedIterator
           + TrustedLen {
        self.downcast_iter()
            .map(|arr| Utf8IterNoNull::new(arr))
            .flatten()
            .trust_my_length(self.len())
    }
}

impl<'a> IntoIterator for &'a ListChunked {
    type Item = Option<Series>;
    type IntoIter = Box<dyn PolarsIterator<Item = Self::Item> + 'a>;
    fn into_iter(self) -> Self::IntoIter {
        Box::new(
            self.downcast_iter()
                .map(|arr| arr.iter())
                .flatten()
                .trust_my_length(self.len())
                .map(|arr| arr.map(|arr| Series::try_from(("", arr)).unwrap())),
        )
    }
}

pub struct ListIterNoNull<'a> {
    array: &'a LargeListArray,
    current: usize,
    current_end: usize,
}

impl<'a> ListIterNoNull<'a> {
    /// create a new iterator
    pub fn new(array: &'a LargeListArray) -> Self {
        ListIterNoNull {
            array,
            current: 0,
            current_end: array.len(),
        }
    }
}

impl<'a> Iterator for ListIterNoNull<'a> {
    type Item = Series;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current == self.current_end {
            None
        } else {
            let old = self.current;
            self.current += 1;
            unsafe { Some(Series::try_from(("", self.array.value_unchecked(old))).unwrap()) }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (
            self.array.len() - self.current,
            Some(self.array.len() - self.current),
        )
    }
}

impl<'a> DoubleEndedIterator for ListIterNoNull<'a> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.current_end == self.current {
            None
        } else {
            self.current_end -= 1;
            unsafe {
                Some(Series::try_from(("", self.array.value_unchecked(self.current_end))).unwrap())
            }
        }
    }
}

/// all arrays have known size.
impl<'a> ExactSizeIterator for ListIterNoNull<'a> {}

impl ListChunked {
    #[allow(clippy::wrong_self_convention)]
    pub fn into_no_null_iter(
        &self,
    ) -> impl Iterator<Item = Series>
           + '_
           + Send
           + Sync
           + ExactSizeIterator
           + DoubleEndedIterator
           + TrustedLen {
        self.downcast_iter()
            .map(|arr| ListIterNoNull::new(arr))
            .flatten()
            .trust_my_length(self.len())
    }
}

#[cfg(feature = "object")]
impl<'a, T> IntoIterator for &'a ObjectChunked<T>
where
    T: PolarsObject,
{
    type Item = Option<&'a T>;
    type IntoIter = Box<dyn PolarsIterator<Item = Self::Item> + 'a>;
    fn into_iter(self) -> Self::IntoIter {
        Box::new(self.downcast_iter().flatten().trust_my_length(self.len()))
    }
}

#[cfg(feature = "object")]
impl<T: PolarsObject> ObjectChunked<T> {
    #[allow(clippy::wrong_self_convention)]
    pub fn into_no_null_iter(
        &self,
    ) -> impl Iterator<Item = &T> + '_ + Send + Sync + ExactSizeIterator + DoubleEndedIterator + TrustedLen
    {
        self.downcast_iter()
            .map(|arr| arr.values().iter())
            .flatten()
            .trust_my_length(self.len())
    }
}

/// Trait for ChunkedArrays that don't have null values.
/// The result is the most efficient implementation `Iterator`, according to the number of chunks.
pub trait IntoNoNullIterator {
    type Item;
    type IntoIter: Iterator<Item = Self::Item>;

    fn into_no_null_iter(self) -> Self::IntoIter;
}

/// Wrapper struct to convert an iterator of type `T` into one of type `Option<T>`.  It is useful to make the
/// `IntoIterator` trait, in which every iterator shall return an `Option<T>`.
pub struct SomeIterator<I>(I)
where
    I: Iterator;

impl<I> Iterator for SomeIterator<I>
where
    I: Iterator,
{
    type Item = Option<I::Item>;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(Some)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl<I> DoubleEndedIterator for SomeIterator<I>
where
    I: DoubleEndedIterator,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        self.0.next_back().map(Some)
    }
}

impl<I> ExactSizeIterator for SomeIterator<I> where I: ExactSizeIterator {}

impl CategoricalChunked {
    #[allow(clippy::wrong_self_convention)]
    pub fn into_no_null_iter(
        &self,
    ) -> impl Iterator<Item = u32>
           + '_
           + Send
           + Sync
           + ExactSizeIterator
           + DoubleEndedIterator
           + TrustedLen {
        self.deref().into_no_null_iter()
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    fn out_of_bounds() {
        let mut a = UInt32Chunked::new_from_slice("a", &[1, 2, 3]);
        let b = UInt32Chunked::new_from_slice("a", &[1, 2, 3]);
        a.append(&b);

        let v = a.into_iter().collect::<Vec<_>>();
        assert_eq!(
            vec![Some(1u32), Some(2), Some(3), Some(1), Some(2), Some(3)],
            v
        )
    }

    /// Generate test for `IntoIterator` trait for chunked arrays with just one chunk and no null values.
    /// The expected return value of the iterator generated by `IntoIterator` trait is `Option<T>`, where
    /// `T` is the chunked array type.
    ///
    /// # Input
    ///
    /// test_name: The name of the test to generate.
    /// ca_type: The chunked array to use for this test. Ex: `Utf8Chunked`, `UInt32Chunked` ...
    /// first_val: The first value contained in the chunked array.
    /// second_val: The second value contained in the chunked array.
    /// third_val: The third value contained in the chunked array.
    macro_rules! impl_test_iter_single_chunk {
        ($test_name:ident, $ca_type:ty, $first_val:expr, $second_val:expr, $third_val:expr) => {
            #[test]
            fn $test_name() {
                let a = <$ca_type>::new_from_slice("test", &[$first_val, $second_val, $third_val]);

                // normal iterator
                let mut it = a.into_iter();
                assert_eq!(it.next(), Some(Some($first_val)));
                assert_eq!(it.next(), Some(Some($second_val)));
                assert_eq!(it.next(), Some(Some($third_val)));
                assert_eq!(it.next(), None);
                // ensure both sides are consumes.
                assert_eq!(it.next_back(), None);

                // reverse iterator
                let mut it = a.into_iter();
                assert_eq!(it.next_back(), Some(Some($third_val)));
                assert_eq!(it.next_back(), Some(Some($second_val)));
                assert_eq!(it.next_back(), Some(Some($first_val)));
                assert_eq!(it.next_back(), None);
                // ensure both sides are consumes.
                assert_eq!(it.next(), None);

                // iterators should not cross
                let mut it = a.into_iter();
                assert_eq!(it.next_back(), Some(Some($third_val)));
                assert_eq!(it.next(), Some(Some($first_val)));
                assert_eq!(it.next(), Some(Some($second_val)));
                // should stop here as we took this one from the back
                assert_eq!(it.next(), None);
                // ensure both sides are consumes.
                assert_eq!(it.next_back(), None);

                // do the same from the right side
                let mut it = a.into_iter();
                assert_eq!(it.next(), Some(Some($first_val)));
                assert_eq!(it.next_back(), Some(Some($third_val)));
                assert_eq!(it.next_back(), Some(Some($second_val)));
                assert_eq!(it.next_back(), None);
                // ensure both sides are consumes.
                assert_eq!(it.next(), None);
            }
        };
    }

    impl_test_iter_single_chunk!(num_iter_single_chunk, UInt32Chunked, 1, 2, 3);
    impl_test_iter_single_chunk!(utf8_iter_single_chunk, Utf8Chunked, "a", "b", "c");
    impl_test_iter_single_chunk!(bool_iter_single_chunk, BooleanChunked, true, true, false);

    /// Generate test for `IntoIterator` trait for chunked arrays with just one chunk and null values.
    /// The expected return value of the iterator generated by `IntoIterator` trait is `Option<T>`, where
    /// `T` is the chunked array type.
    ///
    /// # Input
    ///
    /// test_name: The name of the test to generate.
    /// ca_type: The chunked array to use for this test. Ex: `Utf8Chunked`, `UInt32Chunked` ...
    /// first_val: The first value contained in the chunked array. Must be an `Option<T>`.
    /// second_val: The second value contained in the chunked array. Must be an `Option<T>`.
    /// third_val: The third value contained in the chunked array. Must be an `Option<T>`.
    macro_rules! impl_test_iter_single_chunk_null_check {
        ($test_name:ident, $ca_type:ty, $first_val:expr, $second_val:expr, $third_val:expr) => {
            #[test]
            fn $test_name() {
                let a =
                    <$ca_type>::new_from_opt_slice("test", &[$first_val, $second_val, $third_val]);

                // normal iterator
                let mut it = a.into_iter();
                assert_eq!(it.next(), Some($first_val));
                assert_eq!(it.next(), Some($second_val));
                assert_eq!(it.next(), Some($third_val));
                assert_eq!(it.next(), None);
                // ensure both sides are consumes.
                assert_eq!(it.next_back(), None);

                // reverse iterator
                let mut it = a.into_iter();
                assert_eq!(it.next_back(), Some($third_val));
                assert_eq!(it.next_back(), Some($second_val));
                assert_eq!(it.next_back(), Some($first_val));
                assert_eq!(it.next_back(), None);
                // ensure both sides are consumes.
                assert_eq!(it.next(), None);

                // iterators should not cross
                let mut it = a.into_iter();
                assert_eq!(it.next_back(), Some($third_val));
                assert_eq!(it.next(), Some($first_val));
                assert_eq!(it.next(), Some($second_val));
                // should stop here as we took this one from the back
                assert_eq!(it.next(), None);
                // ensure both sides are consumes.
                assert_eq!(it.next_back(), None);

                // do the same from the right side
                let mut it = a.into_iter();
                assert_eq!(it.next(), Some($first_val));
                assert_eq!(it.next_back(), Some($third_val));
                assert_eq!(it.next_back(), Some($second_val));
                assert_eq!(it.next_back(), None);
                // ensure both sides are consumes.
                assert_eq!(it.next(), None);
            }
        };
    }

    impl_test_iter_single_chunk_null_check!(
        num_iter_single_chunk_null_check,
        UInt32Chunked,
        Some(1),
        None,
        Some(3)
    );
    impl_test_iter_single_chunk_null_check!(
        utf8_iter_single_chunk_null_check,
        Utf8Chunked,
        Some("a"),
        None,
        Some("c")
    );
    impl_test_iter_single_chunk_null_check!(
        bool_iter_single_chunk_null_check,
        BooleanChunked,
        Some(true),
        None,
        Some(false)
    );

    /// Generate test for `IntoIterator` trait for chunked arrays with many chunks and no null values.
    /// The expected return value of the iterator generated by `IntoIterator` trait is `Option<T>`, where
    /// `T` is the chunked array type.
    ///
    /// # Input
    ///
    /// test_name: The name of the test to generate.
    /// ca_type: The chunked array to use for this test. Ex: `Utf8Chunked`, `UInt32Chunked` ...
    /// first_val: The first value contained in the chunked array.
    /// second_val: The second value contained in the chunked array.
    /// third_val: The third value contained in the chunked array.
    macro_rules! impl_test_iter_many_chunk {
        ($test_name:ident, $ca_type:ty, $first_val:expr, $second_val:expr, $third_val:expr) => {
            #[test]
            fn $test_name() {
                let mut a = <$ca_type>::new_from_slice("test", &[$first_val, $second_val]);
                let a_b = <$ca_type>::new_from_slice("", &[$third_val]);
                a.append(&a_b);

                // normal iterator
                let mut it = a.into_iter();
                assert_eq!(it.next(), Some(Some($first_val)));
                assert_eq!(it.next(), Some(Some($second_val)));
                assert_eq!(it.next(), Some(Some($third_val)));
                assert_eq!(it.next(), None);
                // ensure both sides are consumes.
                assert_eq!(it.next_back(), None);

                // reverse iterator
                let mut it = a.into_iter();
                assert_eq!(it.next_back(), Some(Some($third_val)));
                assert_eq!(it.next_back(), Some(Some($second_val)));
                assert_eq!(it.next_back(), Some(Some($first_val)));
                assert_eq!(it.next_back(), None);
                // ensure both sides are consumes.
                assert_eq!(it.next(), None);

                // iterators should not cross
                let mut it = a.into_iter();
                assert_eq!(it.next_back(), Some(Some($third_val)));
                assert_eq!(it.next(), Some(Some($first_val)));
                assert_eq!(it.next(), Some(Some($second_val)));
                // should stop here as we took this one from the back
                assert_eq!(it.next(), None);
                // ensure both sides are consumes.
                assert_eq!(it.next_back(), None);

                // do the same from the right side
                let mut it = a.into_iter();
                assert_eq!(it.next(), Some(Some($first_val)));
                assert_eq!(it.next_back(), Some(Some($third_val)));
                assert_eq!(it.next_back(), Some(Some($second_val)));
                assert_eq!(it.next_back(), None);
                // ensure both sides are consumes.
                assert_eq!(it.next(), None);
            }
        };
    }

    impl_test_iter_many_chunk!(num_iter_many_chunk, UInt32Chunked, 1, 2, 3);
    impl_test_iter_many_chunk!(utf8_iter_many_chunk, Utf8Chunked, "a", "b", "c");
    impl_test_iter_many_chunk!(bool_iter_many_chunk, BooleanChunked, true, true, false);

    /// Generate test for `IntoIterator` trait for chunked arrays with many chunk and null values.
    /// The expected return value of the iterator generated by `IntoIterator` trait is `Option<T>`, where
    /// `T` is the chunked array type.
    ///
    /// # Input
    ///
    /// test_name: The name of the test to generate.
    /// ca_type: The chunked array to use for this test. Ex: `Utf8Chunked`, `UInt32Chunked` ...
    /// first_val: The first value contained in the chunked array. Must be an `Option<T>`.
    /// second_val: The second value contained in the chunked array. Must be an `Option<T>`.
    /// third_val: The third value contained in the chunked array. Must be an `Option<T>`.
    macro_rules! impl_test_iter_many_chunk_null_check {
        ($test_name:ident, $ca_type:ty, $first_val:expr, $second_val:expr, $third_val:expr) => {
            #[test]
            fn $test_name() {
                let mut a = <$ca_type>::new_from_opt_slice("test", &[$first_val, $second_val]);
                let a_b = <$ca_type>::new_from_opt_slice("", &[$third_val]);
                a.append(&a_b);

                // normal iterator
                let mut it = a.into_iter();
                assert_eq!(it.next(), Some($first_val));
                assert_eq!(it.next(), Some($second_val));
                assert_eq!(it.next(), Some($third_val));
                assert_eq!(it.next(), None);
                // ensure both sides are consumes.
                assert_eq!(it.next_back(), None);

                // reverse iterator
                let mut it = a.into_iter();
                assert_eq!(it.next_back(), Some($third_val));
                assert_eq!(it.next_back(), Some($second_val));
                assert_eq!(it.next_back(), Some($first_val));
                assert_eq!(it.next_back(), None);
                // ensure both sides are consumes.
                assert_eq!(it.next(), None);

                // iterators should not cross
                let mut it = a.into_iter();
                assert_eq!(it.next_back(), Some($third_val));
                assert_eq!(it.next(), Some($first_val));
                assert_eq!(it.next(), Some($second_val));
                // should stop here as we took this one from the back
                assert_eq!(it.next(), None);
                // ensure both sides are consumes.
                assert_eq!(it.next_back(), None);

                // do the same from the right side
                let mut it = a.into_iter();
                assert_eq!(it.next(), Some($first_val));
                assert_eq!(it.next_back(), Some($third_val));
                assert_eq!(it.next_back(), Some($second_val));
                assert_eq!(it.next_back(), None);
                // ensure both sides are consumes.
                assert_eq!(it.next(), None);
            }
        };
    }

    impl_test_iter_many_chunk_null_check!(
        num_iter_many_chunk_null_check,
        UInt32Chunked,
        Some(1),
        None,
        Some(3)
    );
    impl_test_iter_many_chunk_null_check!(
        utf8_iter_many_chunk_null_check,
        Utf8Chunked,
        Some("a"),
        None,
        Some("c")
    );
    impl_test_iter_many_chunk_null_check!(
        bool_iter_many_chunk_null_check,
        BooleanChunked,
        Some(true),
        None,
        Some(false)
    );

    /// Generate test for `IntoNoNullIterator` trait for chunked arrays with just one chunk and no null values.
    /// The expected return value of the iterator generated by `IntoNoNullIterator` trait is `T`, where
    /// `T` is the chunked array type.
    ///
    /// # Input
    ///
    /// test_name: The name of the test to generate.
    /// ca_type: The chunked array to use for this test. Ex: `Utf8Chunked`, `UInt32Chunked` ...
    /// first_val: The first value contained in the chunked array.
    /// second_val: The second value contained in the chunked array.
    /// third_val: The third value contained in the chunked array.
    macro_rules! impl_test_no_null_iter_single_chunk {
        ($test_name:ident, $ca_type:ty, $first_val:expr, $second_val:expr, $third_val:expr) => {
            #[test]
            fn $test_name() {
                let a = <$ca_type>::new_from_slice("test", &[$first_val, $second_val, $third_val]);

                // normal iterator
                let mut it = a.into_no_null_iter();
                assert_eq!(it.next(), Some($first_val));
                assert_eq!(it.next(), Some($second_val));
                assert_eq!(it.next(), Some($third_val));
                assert_eq!(it.next(), None);
                // ensure both sides are consumes.
                assert_eq!(it.next_back(), None);

                // reverse iterator
                let mut it = a.into_no_null_iter();
                assert_eq!(it.next_back(), Some($third_val));
                assert_eq!(it.next_back(), Some($second_val));
                assert_eq!(it.next_back(), Some($first_val));
                assert_eq!(it.next_back(), None);
                // ensure both sides are consumes.
                assert_eq!(it.next(), None);

                // iterators should not cross
                let mut it = a.into_no_null_iter();
                assert_eq!(it.next_back(), Some($third_val));
                assert_eq!(it.next(), Some($first_val));
                assert_eq!(it.next(), Some($second_val));
                // should stop here as we took this one from the back
                assert_eq!(it.next(), None);
                // ensure both sides are consumes.
                assert_eq!(it.next_back(), None);

                // do the same from the right side
                let mut it = a.into_no_null_iter();
                assert_eq!(it.next(), Some($first_val));
                assert_eq!(it.next_back(), Some($third_val));
                assert_eq!(it.next_back(), Some($second_val));
                assert_eq!(it.next_back(), None);
                // ensure both sides are consumes.
                assert_eq!(it.next(), None);
            }
        };
    }

    impl_test_no_null_iter_single_chunk!(num_no_null_iter_single_chunk, UInt32Chunked, 1, 2, 3);
    impl_test_no_null_iter_single_chunk!(
        utf8_no_null_iter_single_chunk,
        Utf8Chunked,
        "a",
        "b",
        "c"
    );
    impl_test_no_null_iter_single_chunk!(
        bool_no_null_iter_single_chunk,
        BooleanChunked,
        true,
        true,
        false
    );

    /// Generate test for `IntoNoNullIterator` trait for chunked arrays with many chunks and no null values.
    /// The expected return value of the iterator generated by `IntoNoNullIterator` trait is `T`, where
    /// `T` is the chunked array type.
    ///
    /// # Input
    ///
    /// test_name: The name of the test to generate.
    /// ca_type: The chunked array to use for this test. Ex: `Utf8Chunked`, `UInt32Chunked` ...
    /// first_val: The first value contained in the chunked array.
    /// second_val: The second value contained in the chunked array.
    /// third_val: The third value contained in the chunked array.
    macro_rules! impl_test_no_null_iter_many_chunk {
        ($test_name:ident, $ca_type:ty, $first_val:expr, $second_val:expr, $third_val:expr) => {
            #[test]
            fn $test_name() {
                let mut a = <$ca_type>::new_from_slice("test", &[$first_val, $second_val]);
                let a_b = <$ca_type>::new_from_slice("", &[$third_val]);
                a.append(&a_b);

                // normal iterator
                let mut it = a.into_no_null_iter();
                assert_eq!(it.next(), Some($first_val));
                assert_eq!(it.next(), Some($second_val));
                assert_eq!(it.next(), Some($third_val));
                assert_eq!(it.next(), None);
                // ensure both sides are consumes.
                assert_eq!(it.next_back(), None);

                // reverse iterator
                let mut it = a.into_no_null_iter();
                assert_eq!(it.next_back(), Some($third_val));
                assert_eq!(it.next_back(), Some($second_val));
                assert_eq!(it.next_back(), Some($first_val));
                assert_eq!(it.next_back(), None);
                // ensure both sides are consumes.
                assert_eq!(it.next(), None);

                // iterators should not cross
                let mut it = a.into_no_null_iter();
                assert_eq!(it.next_back(), Some($third_val));
                assert_eq!(it.next(), Some($first_val));
                assert_eq!(it.next(), Some($second_val));
                // should stop here as we took this one from the back
                assert_eq!(it.next(), None);
                // ensure both sides are consumes.
                assert_eq!(it.next_back(), None);

                // do the same from the right side
                let mut it = a.into_no_null_iter();
                assert_eq!(it.next(), Some($first_val));
                assert_eq!(it.next_back(), Some($third_val));
                assert_eq!(it.next_back(), Some($second_val));
                assert_eq!(it.next_back(), None);
                // ensure both sides are consumes.
                assert_eq!(it.next(), None);
            }
        };
    }

    impl_test_no_null_iter_many_chunk!(num_no_null_iter_many_chunk, UInt32Chunked, 1, 2, 3);
    impl_test_no_null_iter_many_chunk!(utf8_no_null_iter_many_chunk, Utf8Chunked, "a", "b", "c");
    impl_test_no_null_iter_many_chunk!(
        bool_no_null_iter_many_chunk,
        BooleanChunked,
        true,
        true,
        false
    );

    /// The size of the skip iterator.
    const SKIP_ITERATOR_SIZE: usize = 10;

    /// Generates tests to verify the correctness of the `skip` method.
    ///
    /// # Input
    ///
    /// test_name: The name of the test to implement, it is a function name so it shall be unique.
    /// skip_values: The number of values to skip. Keep in mind that is the number of values to skip
    ///   after performing the first next, then, skip_values = 8, will skip until index 1 + skip_values = 9.
    /// first_val: The value before skip.
    /// second_val: The value after skip.
    /// ca_init_block: The block which initialize the chunked array. It shall return the chunked array.
    macro_rules! impl_test_iter_skip {
        ($test_name:ident, $skip_values:expr, $first_val:expr, $second_val:expr, $ca_init_block:block) => {
            #[test]
            fn $test_name() {
                let a = $ca_init_block;

                // Consume first position of iterator.
                let mut it = a.into_iter();
                assert_eq!(it.next(), Some($first_val));

                // Consume `$skip_values` and check the result.
                let mut it = it.skip($skip_values);
                assert_eq!(it.next(), Some($second_val));

                // Consume more values than available and check result is None.
                let mut it = it.skip(SKIP_ITERATOR_SIZE);
                assert_eq!(it.next(), None);
            }
        };
    }

    /// Generates a `Vec` of `Strings`, where every position is the `String` representation of its index.
    fn generate_utf8_vec(size: usize) -> Vec<String> {
        (0..size).map(|n| n.to_string()).collect()
    }

    /// Generate a `Vec` of `Option<String>`, where even indexes are `Some("{idx}")` and odd indexes are `None`.
    fn generate_opt_utf8_vec(size: usize) -> Vec<Option<String>> {
        (0..size)
            .map(|n| {
                if n % 2 == 0 {
                    Some(n.to_string())
                } else {
                    None
                }
            })
            .collect()
    }

    impl_test_iter_skip!(utf8_iter_single_chunk_skip, 8, Some("0"), Some("9"), {
        Utf8Chunked::new_from_slice("test", &generate_utf8_vec(SKIP_ITERATOR_SIZE))
    });

    impl_test_iter_skip!(
        utf8_iter_single_chunk_null_check_skip,
        8,
        Some("0"),
        None,
        { Utf8Chunked::new_from_opt_slice("test", &generate_opt_utf8_vec(SKIP_ITERATOR_SIZE)) }
    );

    impl_test_iter_skip!(utf8_iter_many_chunk_skip, 18, Some("0"), Some("9"), {
        let mut a = Utf8Chunked::new_from_slice("test", &generate_utf8_vec(SKIP_ITERATOR_SIZE));
        let a_b = Utf8Chunked::new_from_slice("test", &generate_utf8_vec(SKIP_ITERATOR_SIZE));
        a.append(&a_b);
        a
    });

    impl_test_iter_skip!(utf8_iter_many_chunk_null_check_skip, 18, Some("0"), None, {
        let mut a =
            Utf8Chunked::new_from_opt_slice("test", &generate_opt_utf8_vec(SKIP_ITERATOR_SIZE));
        let a_b =
            Utf8Chunked::new_from_opt_slice("test", &generate_opt_utf8_vec(SKIP_ITERATOR_SIZE));
        a.append(&a_b);
        a
    });

    /// Generates a `Vec` of `bool`, with even indexes are true, and odd indexes are false.
    fn generate_boolean_vec(size: usize) -> Vec<bool> {
        (0..size).map(|n| n % 2 == 0).collect()
    }

    /// Generate a `Vec` of `Option<bool>`, where:
    /// - If the index is divisible by 3, then, the value is `None`.
    /// - If the index is not divisible by 3 and it is even, then, the value is `Some(true)`.
    /// - Otherwise, the value is `Some(false)`.
    fn generate_opt_boolean_vec(size: usize) -> Vec<Option<bool>> {
        (0..size)
            .map(|n| if n % 3 == 0 { None } else { Some(n % 2 == 0) })
            .collect()
    }

    impl_test_iter_skip!(bool_iter_single_chunk_skip, 8, Some(true), Some(false), {
        BooleanChunked::new_from_slice("test", &generate_boolean_vec(SKIP_ITERATOR_SIZE))
    });

    impl_test_iter_skip!(bool_iter_single_chunk_null_check_skip, 8, None, None, {
        BooleanChunked::new_from_opt_slice("test", &generate_opt_boolean_vec(SKIP_ITERATOR_SIZE))
    });

    impl_test_iter_skip!(bool_iter_many_chunk_skip, 18, Some(true), Some(false), {
        let mut a =
            BooleanChunked::new_from_slice("test", &generate_boolean_vec(SKIP_ITERATOR_SIZE));
        let a_b = BooleanChunked::new_from_slice("test", &generate_boolean_vec(SKIP_ITERATOR_SIZE));
        a.append(&a_b);
        a
    });

    impl_test_iter_skip!(bool_iter_many_chunk_null_check_skip, 18, None, None, {
        let mut a = BooleanChunked::new_from_opt_slice(
            "test",
            &generate_opt_boolean_vec(SKIP_ITERATOR_SIZE),
        );
        let a_b = BooleanChunked::new_from_opt_slice(
            "test",
            &generate_opt_boolean_vec(SKIP_ITERATOR_SIZE),
        );
        a.append(&a_b);
        a
    });
}
