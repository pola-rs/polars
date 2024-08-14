use arrow::array::*;

use crate::prelude::*;

pub mod par;

impl<T> ChunkedArray<T>
where
    T: PolarsDataType,
{
    #[inline]
    pub fn iter(&self) -> impl PolarsIterator<Item = Option<T::Physical<'_>>> {
        // SAFETY: we set the correct length of the iterator.
        unsafe {
            self.downcast_iter()
                .flat_map(|arr| arr.iter())
                .trust_my_length(self.len())
        }
    }
}

/// A [`PolarsIterator`] is an iterator over a [`ChunkedArray`] which contains polars types. A [`PolarsIterator`]
/// must implement [`ExactSizeIterator`] and [`DoubleEndedIterator`].
pub trait PolarsIterator:
    ExactSizeIterator + DoubleEndedIterator + Send + Sync + TrustedLen
{
}
unsafe impl<'a, I> TrustedLen for Box<dyn PolarsIterator<Item = I> + 'a> {}

/// Implement [`PolarsIterator`] for every iterator that implements the needed traits.
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
            // we know that we only iterate over length == self.len()
            unsafe {
                self.downcast_iter()
                    .flatten()
                    .map(|x| x.copied())
                    .trust_my_length(self.len())
            },
        )
    }
}

impl<'a> IntoIterator for &'a BooleanChunked {
    type Item = Option<bool>;
    type IntoIter = Box<dyn PolarsIterator<Item = Self::Item> + 'a>;
    fn into_iter(self) -> Self::IntoIter {
        // we know that we only iterate over length == self.len()
        unsafe { Box::new(self.downcast_iter().flatten().trust_my_length(self.len())) }
    }
}

/// The no null iterator for a [`BooleanArray`]
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
    #[doc(hidden)]
    pub fn into_no_null_iter(
        &self,
    ) -> impl '_ + Send + Sync + ExactSizeIterator<Item = bool> + DoubleEndedIterator + TrustedLen
    {
        // we know that we only iterate over length == self.len()
        unsafe {
            self.downcast_iter()
                .flat_map(BoolIterNoNull::new)
                .trust_my_length(self.len())
        }
    }
}

impl<'a> IntoIterator for &'a StringChunked {
    type Item = Option<&'a str>;
    type IntoIter = Box<dyn PolarsIterator<Item = Self::Item> + 'a>;
    fn into_iter(self) -> Self::IntoIter {
        // we know that we only iterate over length == self.len()
        unsafe { Box::new(self.downcast_iter().flatten().trust_my_length(self.len())) }
    }
}

impl StringChunked {
    #[allow(clippy::wrong_self_convention)]
    #[doc(hidden)]
    pub fn into_no_null_iter(
        &self,
    ) -> impl '_ + Send + Sync + ExactSizeIterator<Item = &str> + DoubleEndedIterator + TrustedLen
    {
        // we know that we only iterate over length == self.len()
        unsafe {
            self.downcast_iter()
                .flat_map(|arr| arr.values_iter())
                .trust_my_length(self.len())
        }
    }
}

impl<'a> IntoIterator for &'a BinaryChunked {
    type Item = Option<&'a [u8]>;
    type IntoIter = Box<dyn PolarsIterator<Item = Self::Item> + 'a>;
    fn into_iter(self) -> Self::IntoIter {
        // we know that we only iterate over length == self.len()
        unsafe { Box::new(self.downcast_iter().flatten().trust_my_length(self.len())) }
    }
}

impl BinaryChunked {
    #[allow(clippy::wrong_self_convention)]
    #[doc(hidden)]
    pub fn into_no_null_iter(
        &self,
    ) -> impl '_ + Send + Sync + ExactSizeIterator<Item = &[u8]> + DoubleEndedIterator + TrustedLen
    {
        // we know that we only iterate over length == self.len()
        unsafe {
            self.downcast_iter()
                .flat_map(|arr| arr.values_iter())
                .trust_my_length(self.len())
        }
    }
}

impl<'a> IntoIterator for &'a BinaryOffsetChunked {
    type Item = Option<&'a [u8]>;
    type IntoIter = Box<dyn PolarsIterator<Item = Self::Item> + 'a>;
    fn into_iter(self) -> Self::IntoIter {
        // we know that we only iterate over length == self.len()
        unsafe { Box::new(self.downcast_iter().flatten().trust_my_length(self.len())) }
    }
}

impl BinaryOffsetChunked {
    #[allow(clippy::wrong_self_convention)]
    #[doc(hidden)]
    pub fn into_no_null_iter(
        &self,
    ) -> impl '_ + Send + Sync + ExactSizeIterator<Item = &[u8]> + DoubleEndedIterator + TrustedLen
    {
        // we know that we only iterate over length == self.len()
        unsafe {
            self.downcast_iter()
                .flat_map(|arr| arr.values_iter())
                .trust_my_length(self.len())
        }
    }
}

impl<'a> IntoIterator for &'a ListChunked {
    type Item = Option<Series>;
    type IntoIter = Box<dyn PolarsIterator<Item = Self::Item> + 'a>;
    fn into_iter(self) -> Self::IntoIter {
        let dtype = self.inner_dtype();

        if self.null_count() == 0 {
            // we know that we only iterate over length == self.len()
            unsafe {
                Box::new(
                    self.downcast_iter()
                        .flat_map(|arr| arr.iter().unwrap_required())
                        .trust_my_length(self.len())
                        .map(move |arr| {
                            Some(Series::from_chunks_and_dtype_unchecked(
                                "",
                                vec![arr],
                                dtype,
                            ))
                        }),
                )
            }
        } else {
            // we know that we only iterate over length == self.len()
            unsafe {
                Box::new(
                    self.downcast_iter()
                        .flat_map(|arr| arr.iter())
                        .trust_my_length(self.len())
                        .map(move |arr| {
                            arr.map(|arr| {
                                Series::from_chunks_and_dtype_unchecked("", vec![arr], dtype)
                            })
                        }),
                )
            }
        }
    }
}

impl ListChunked {
    #[allow(clippy::wrong_self_convention)]
    #[doc(hidden)]
    pub fn into_no_null_iter(
        &self,
    ) -> impl '_ + Send + Sync + ExactSizeIterator<Item = Series> + DoubleEndedIterator + TrustedLen
    {
        let inner_type = self.inner_dtype();
        unsafe {
            self.downcast_iter()
                .flat_map(|arr| arr.values_iter())
                .map(move |arr| Series::from_chunks_and_dtype_unchecked("", vec![arr], inner_type))
                .trust_my_length(self.len())
        }
    }
}

#[cfg(feature = "dtype-array")]
impl<'a> IntoIterator for &'a ArrayChunked {
    type Item = Option<Series>;
    type IntoIter = Box<dyn PolarsIterator<Item = Self::Item> + 'a>;
    fn into_iter(self) -> Self::IntoIter {
        let dtype = self.inner_dtype();

        if self.null_count() == 0 {
            // we know that we only iterate over length == self.len()
            unsafe {
                Box::new(
                    self.downcast_iter()
                        .flat_map(|arr| arr.iter().unwrap_required())
                        .trust_my_length(self.len())
                        .map(move |arr| {
                            Some(Series::from_chunks_and_dtype_unchecked(
                                "",
                                vec![arr],
                                dtype,
                            ))
                        }),
                )
            }
        } else {
            // we know that we only iterate over length == self.len()
            unsafe {
                Box::new(
                    self.downcast_iter()
                        .flat_map(|arr| arr.iter())
                        .trust_my_length(self.len())
                        .map(move |arr| {
                            arr.map(|arr| {
                                Series::from_chunks_and_dtype_unchecked("", vec![arr], dtype)
                            })
                        }),
                )
            }
        }
    }
}

#[cfg(feature = "dtype-array")]
pub struct FixedSizeListIterNoNull<'a> {
    array: &'a FixedSizeListArray,
    inner_type: DataType,
    current: usize,
    current_end: usize,
}

#[cfg(feature = "dtype-array")]
impl<'a> FixedSizeListIterNoNull<'a> {
    /// create a new iterator
    pub fn new(array: &'a FixedSizeListArray, inner_type: DataType) -> Self {
        FixedSizeListIterNoNull {
            array,
            inner_type,
            current: 0,
            current_end: array.len(),
        }
    }
}

#[cfg(feature = "dtype-array")]
impl<'a> Iterator for FixedSizeListIterNoNull<'a> {
    type Item = Series;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current == self.current_end {
            None
        } else {
            let old = self.current;
            self.current += 1;
            unsafe {
                Some(Series::from_chunks_and_dtype_unchecked(
                    "",
                    vec![self.array.value_unchecked(old)],
                    &self.inner_type,
                ))
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (
            self.array.len() - self.current,
            Some(self.array.len() - self.current),
        )
    }
}

#[cfg(feature = "dtype-array")]
impl<'a> DoubleEndedIterator for FixedSizeListIterNoNull<'a> {
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
#[cfg(feature = "dtype-array")]
impl<'a> ExactSizeIterator for FixedSizeListIterNoNull<'a> {}

#[cfg(feature = "dtype-array")]
impl ArrayChunked {
    #[allow(clippy::wrong_self_convention)]
    #[doc(hidden)]
    pub fn into_no_null_iter(
        &self,
    ) -> impl '_ + Send + Sync + ExactSizeIterator<Item = Series> + DoubleEndedIterator + TrustedLen
    {
        // we know that we only iterate over length == self.len()
        let inner_type = self.inner_dtype();
        unsafe {
            self.downcast_iter()
                .flat_map(move |arr| FixedSizeListIterNoNull::new(arr, inner_type.clone()))
                .trust_my_length(self.len())
        }
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
        // we know that we only iterate over length == self.len()
        unsafe { Box::new(self.downcast_iter().flatten().trust_my_length(self.len())) }
    }
}

#[cfg(feature = "object")]
impl<T: PolarsObject> ObjectChunked<T> {
    #[allow(clippy::wrong_self_convention)]
    #[doc(hidden)]
    pub fn into_no_null_iter(
        &self,
    ) -> impl '_ + Send + Sync + ExactSizeIterator<Item = &T> + DoubleEndedIterator + TrustedLen
    {
        // we know that we only iterate over length == self.len()
        unsafe {
            self.downcast_iter()
                .flat_map(|arr| arr.values().iter())
                .trust_my_length(self.len())
        }
    }
}

/// Wrapper struct to convert an iterator of type `T` into one of type [`Option<T>`].  It is useful to make the
/// [`IntoIterator`] trait, in which every iterator shall return an [`Option<T>`].
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

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    fn out_of_bounds() {
        let mut a = UInt32Chunked::from_slice("a", &[1, 2, 3]);
        let b = UInt32Chunked::from_slice("a", &[1, 2, 3]);
        a.append(&b).unwrap();

        let v = a.into_iter().collect::<Vec<_>>();
        assert_eq!(
            vec![Some(1u32), Some(2), Some(3), Some(1), Some(2), Some(3)],
            v
        )
    }

    /// Generate test for [`IntoIterator`] trait for chunked arrays with just one chunk and no null values.
    /// The expected return value of the iterator generated by [`IntoIterator`] trait is [`Option<T>`], where
    /// `T` is the chunked array type.
    ///
    /// # Input
    ///
    /// test_name: The name of the test to generate.
    /// ca_type: The chunked array to use for this test. Ex: [`StringChunked`], [`UInt32Chunked`] ...
    /// first_val: The first value contained in the chunked array.
    /// second_val: The second value contained in the chunked array.
    /// third_val: The third value contained in the chunked array.
    macro_rules! impl_test_iter_single_chunk {
        ($test_name:ident, $ca_type:ty, $first_val:expr, $second_val:expr, $third_val:expr) => {
            #[test]
            fn $test_name() {
                let a = <$ca_type>::from_slice("test", &[$first_val, $second_val, $third_val]);

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
    impl_test_iter_single_chunk!(utf8_iter_single_chunk, StringChunked, "a", "b", "c");
    impl_test_iter_single_chunk!(bool_iter_single_chunk, BooleanChunked, true, true, false);

    /// Generate test for [`IntoIterator`] trait for chunked arrays with just one chunk and null values.
    /// The expected return value of the iterator generated by [`IntoIterator`] trait is [`Option<T>`], where
    /// `T` is the chunked array type.
    ///
    /// # Input
    ///
    /// test_name: The name of the test to generate.
    /// ca_type: The chunked array to use for this test. Ex: [`StringChunked`], [`UInt32Chunked`] ...
    /// first_val: The first value contained in the chunked array. Must be an [`Option<T>`].
    /// second_val: The second value contained in the chunked array. Must be an [`Option<T>`].
    /// third_val: The third value contained in the chunked array. Must be an [`Option<T>`].
    macro_rules! impl_test_iter_single_chunk_null_check {
        ($test_name:ident, $ca_type:ty, $first_val:expr, $second_val:expr, $third_val:expr) => {
            #[test]
            fn $test_name() {
                let a = <$ca_type>::new("test", &[$first_val, $second_val, $third_val]);

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
        StringChunked,
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

    /// Generate test for [`IntoIterator`] trait for chunked arrays with many chunks and no null values.
    /// The expected return value of the iterator generated by [`IntoIterator`] trait is [`Option<T>`], where
    /// `T` is the chunked array type.
    ///
    /// # Input
    ///
    /// test_name: The name of the test to generate.
    /// ca_type: The chunked array to use for this test. Ex: [`StringChunked`], [`UInt32Chunked`] ...
    /// first_val: The first value contained in the chunked array.
    /// second_val: The second value contained in the chunked array.
    /// third_val: The third value contained in the chunked array.
    macro_rules! impl_test_iter_many_chunk {
        ($test_name:ident, $ca_type:ty, $first_val:expr, $second_val:expr, $third_val:expr) => {
            #[test]
            fn $test_name() {
                let mut a = <$ca_type>::from_slice("test", &[$first_val, $second_val]);
                let a_b = <$ca_type>::from_slice("", &[$third_val]);
                a.append(&a_b).unwrap();

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
    impl_test_iter_many_chunk!(utf8_iter_many_chunk, StringChunked, "a", "b", "c");
    impl_test_iter_many_chunk!(bool_iter_many_chunk, BooleanChunked, true, true, false);

    /// Generate test for [`IntoIterator`] trait for chunked arrays with many chunk and null values.
    /// The expected return value of the iterator generated by [`IntoIterator`] trait is [`Option<T>`], where
    /// `T` is the chunked array type.
    ///
    /// # Input
    ///
    /// test_name: The name of the test to generate.
    /// ca_type: The chunked array to use for this test. Ex: [`StringChunked`], [`UInt32Chunked`] ...
    /// first_val: The first value contained in the chunked array. Must be an [`Option<T>`].
    /// second_val: The second value contained in the chunked array. Must be an [`Option<T>`].
    /// third_val: The third value contained in the chunked array. Must be an [`Option<T>`].
    macro_rules! impl_test_iter_many_chunk_null_check {
        ($test_name:ident, $ca_type:ty, $first_val:expr, $second_val:expr, $third_val:expr) => {
            #[test]
            fn $test_name() {
                let mut a = <$ca_type>::new("test", &[$first_val, $second_val]);
                let a_b = <$ca_type>::new("", &[$third_val]);
                a.append(&a_b).unwrap();

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
        StringChunked,
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

    /// Generate test for [`IntoNoNullIterator`] trait for chunked arrays with just one chunk and no null values.
    /// The expected return value of the iterator generated by [`IntoNoNullIterator`] trait is `T`, where
    /// `T` is the chunked array type.
    ///
    /// # Input
    ///
    /// test_name: The name of the test to generate.
    /// ca_type: The chunked array to use for this test. Ex: [`StringChunked`], [`UInt32Chunked`] ...
    /// first_val: The first value contained in the chunked array.
    /// second_val: The second value contained in the chunked array.
    /// third_val: The third value contained in the chunked array.
    macro_rules! impl_test_no_null_iter_single_chunk {
        ($test_name:ident, $ca_type:ty, $first_val:expr, $second_val:expr, $third_val:expr) => {
            #[test]
            fn $test_name() {
                let a = <$ca_type>::from_slice("test", &[$first_val, $second_val, $third_val]);

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
        StringChunked,
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

    /// Generate test for [`IntoNoNullIterator`] trait for chunked arrays with many chunks and no null values.
    /// The expected return value of the iterator generated by [`IntoNoNullIterator`] trait is `T`, where
    /// `T` is the chunked array type.
    ///
    /// # Input
    ///
    /// test_name: The name of the test to generate.
    /// ca_type: The chunked array to use for this test. Ex: [`StringChunked`], [`UInt32Chunked`] ...
    /// first_val: The first value contained in the chunked array.
    /// second_val: The second value contained in the chunked array.
    /// third_val: The third value contained in the chunked array.
    macro_rules! impl_test_no_null_iter_many_chunk {
        ($test_name:ident, $ca_type:ty, $first_val:expr, $second_val:expr, $third_val:expr) => {
            #[test]
            fn $test_name() {
                let mut a = <$ca_type>::from_slice("test", &[$first_val, $second_val]);
                let a_b = <$ca_type>::from_slice("", &[$third_val]);
                a.append(&a_b).unwrap();

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
    impl_test_no_null_iter_many_chunk!(utf8_no_null_iter_many_chunk, StringChunked, "a", "b", "c");
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
        StringChunked::from_slice("test", &generate_utf8_vec(SKIP_ITERATOR_SIZE))
    });

    impl_test_iter_skip!(
        utf8_iter_single_chunk_null_check_skip,
        8,
        Some("0"),
        None,
        { StringChunked::new("test", &generate_opt_utf8_vec(SKIP_ITERATOR_SIZE)) }
    );

    impl_test_iter_skip!(utf8_iter_many_chunk_skip, 18, Some("0"), Some("9"), {
        let mut a = StringChunked::from_slice("test", &generate_utf8_vec(SKIP_ITERATOR_SIZE));
        let a_b = StringChunked::from_slice("test", &generate_utf8_vec(SKIP_ITERATOR_SIZE));
        a.append(&a_b).unwrap();
        a
    });

    impl_test_iter_skip!(utf8_iter_many_chunk_null_check_skip, 18, Some("0"), None, {
        let mut a = StringChunked::new("test", &generate_opt_utf8_vec(SKIP_ITERATOR_SIZE));
        let a_b = StringChunked::new("test", &generate_opt_utf8_vec(SKIP_ITERATOR_SIZE));
        a.append(&a_b).unwrap();
        a
    });

    /// Generates a [`Vec`] of [`bool`], with even indexes are true, and odd indexes are false.
    fn generate_boolean_vec(size: usize) -> Vec<bool> {
        (0..size).map(|n| n % 2 == 0).collect()
    }

    /// Generate a [`Vec`] of [`Option<bool>`], where:
    /// - If the index is divisible by 3, then, the value is `None`.
    /// - If the index is not divisible by 3 and it is even, then, the value is `Some(true)`.
    /// - Otherwise, the value is `Some(false)`.
    fn generate_opt_boolean_vec(size: usize) -> Vec<Option<bool>> {
        (0..size)
            .map(|n| if n % 3 == 0 { None } else { Some(n % 2 == 0) })
            .collect()
    }

    impl_test_iter_skip!(bool_iter_single_chunk_skip, 8, Some(true), Some(false), {
        BooleanChunked::from_slice("test", &generate_boolean_vec(SKIP_ITERATOR_SIZE))
    });

    impl_test_iter_skip!(bool_iter_single_chunk_null_check_skip, 8, None, None, {
        BooleanChunked::new("test", &generate_opt_boolean_vec(SKIP_ITERATOR_SIZE))
    });

    impl_test_iter_skip!(bool_iter_many_chunk_skip, 18, Some(true), Some(false), {
        let mut a = BooleanChunked::from_slice("test", &generate_boolean_vec(SKIP_ITERATOR_SIZE));
        let a_b = BooleanChunked::from_slice("test", &generate_boolean_vec(SKIP_ITERATOR_SIZE));
        a.append(&a_b).unwrap();
        a
    });

    impl_test_iter_skip!(bool_iter_many_chunk_null_check_skip, 18, None, None, {
        let mut a = BooleanChunked::new("test", &generate_opt_boolean_vec(SKIP_ITERATOR_SIZE));
        let a_b = BooleanChunked::new("test", &generate_opt_boolean_vec(SKIP_ITERATOR_SIZE));
        a.append(&a_b).unwrap();
        a
    });
}
