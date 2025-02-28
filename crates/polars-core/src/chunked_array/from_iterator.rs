//! Implementations of upstream traits for [`ChunkedArray<T>`]
use std::borrow::{Borrow, Cow};

#[cfg(feature = "object")]
use arrow::bitmap::BitmapBuilder;

use crate::chunked_array::builder::{get_list_builder, AnonymousOwnedListBuilder};
#[cfg(feature = "object")]
use crate::chunked_array::object::builder::get_object_type;
#[cfg(feature = "object")]
use crate::chunked_array::object::ObjectArray;
use crate::prelude::*;
use crate::utils::{get_iter_capacity, NoNull};

/// FromIterator trait
impl<T> FromIterator<Option<T::Native>> for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    #[inline]
    fn from_iter<I: IntoIterator<Item = Option<T::Native>>>(iter: I) -> Self {
        // TODO: eliminate this FromIterator implementation entirely.
        iter.into_iter().collect_ca(PlSmallStr::EMPTY)
    }
}

// NoNull is only a wrapper needed for specialization
impl<T> FromIterator<T::Native> for NoNull<ChunkedArray<T>>
where
    T: PolarsNumericType,
{
    // We use Vec because it is way faster than Arrows builder. We can do this because we
    // know we don't have null values.
    #[inline]
    fn from_iter<I: IntoIterator<Item = T::Native>>(iter: I) -> Self {
        // 2021-02-07: aligned vec was ~2x faster than arrow collect.
        let av = iter.into_iter().collect::<Vec<T::Native>>();
        NoNull::new(ChunkedArray::from_vec(PlSmallStr::EMPTY, av))
    }
}

impl FromIterator<Option<bool>> for ChunkedArray<BooleanType> {
    #[inline]
    fn from_iter<I: IntoIterator<Item = Option<bool>>>(iter: I) -> Self {
        BooleanArray::from_iter(iter).into()
    }
}

impl FromIterator<bool> for BooleanChunked {
    #[inline]
    fn from_iter<I: IntoIterator<Item = bool>>(iter: I) -> Self {
        iter.into_iter().collect_ca(PlSmallStr::EMPTY)
    }
}

impl FromIterator<bool> for NoNull<BooleanChunked> {
    #[inline]
    fn from_iter<I: IntoIterator<Item = bool>>(iter: I) -> Self {
        NoNull::new(iter.into_iter().collect_ca(PlSmallStr::EMPTY))
    }
}

// FromIterator for StringChunked variants.

impl<Ptr> FromIterator<Option<Ptr>> for StringChunked
where
    Ptr: AsRef<str>,
{
    #[inline]
    fn from_iter<I: IntoIterator<Item = Option<Ptr>>>(iter: I) -> Self {
        let arr = MutableBinaryViewArray::from_iterator(iter.into_iter()).freeze();
        ChunkedArray::with_chunk(PlSmallStr::EMPTY, arr)
    }
}

/// Local [`AsRef<T>`] trait to circumvent the orphan rule.
pub trait PolarsAsRef<T: ?Sized>: AsRef<T> {}

impl PolarsAsRef<str> for String {}
impl PolarsAsRef<str> for &str {}
// &["foo", "bar"]
impl PolarsAsRef<str> for &&str {}

impl PolarsAsRef<str> for Cow<'_, str> {}
impl PolarsAsRef<[u8]> for Vec<u8> {}
impl PolarsAsRef<[u8]> for &[u8] {}
// TODO: remove!
impl PolarsAsRef<[u8]> for &&[u8] {}
impl PolarsAsRef<[u8]> for Cow<'_, [u8]> {}

impl<Ptr> FromIterator<Ptr> for StringChunked
where
    Ptr: PolarsAsRef<str>,
{
    #[inline]
    fn from_iter<I: IntoIterator<Item = Ptr>>(iter: I) -> Self {
        let arr = MutableBinaryViewArray::from_values_iter(iter.into_iter()).freeze();
        ChunkedArray::with_chunk(PlSmallStr::EMPTY, arr)
    }
}

// FromIterator for BinaryChunked variants.
impl<Ptr> FromIterator<Option<Ptr>> for BinaryChunked
where
    Ptr: AsRef<[u8]>,
{
    #[inline]
    fn from_iter<I: IntoIterator<Item = Option<Ptr>>>(iter: I) -> Self {
        let arr = MutableBinaryViewArray::from_iter(iter).freeze();
        ChunkedArray::with_chunk(PlSmallStr::EMPTY, arr)
    }
}

impl<Ptr> FromIterator<Ptr> for BinaryChunked
where
    Ptr: PolarsAsRef<[u8]>,
{
    #[inline]
    fn from_iter<I: IntoIterator<Item = Ptr>>(iter: I) -> Self {
        let arr = MutableBinaryViewArray::from_values_iter(iter.into_iter()).freeze();
        ChunkedArray::with_chunk(PlSmallStr::EMPTY, arr)
    }
}

impl<Ptr> FromIterator<Ptr> for ListChunked
where
    Ptr: Borrow<Series>,
{
    #[inline]
    fn from_iter<I: IntoIterator<Item = Ptr>>(iter: I) -> Self {
        let mut it = iter.into_iter();
        let capacity = get_iter_capacity(&it);

        // first take one to get the dtype.
        let v = match it.next() {
            Some(v) => v,
            None => return ListChunked::full_null(PlSmallStr::EMPTY, 0),
        };
        // We don't know the needed capacity. We arbitrarily choose an average of 5 elements per series.
        let mut builder = get_list_builder(
            v.borrow().dtype(),
            capacity * 5,
            capacity,
            PlSmallStr::EMPTY,
        );

        builder.append_series(v.borrow()).unwrap();
        for s in it {
            builder.append_series(s.borrow()).unwrap();
        }
        builder.finish()
    }
}

impl FromIterator<Option<Column>> for ListChunked {
    fn from_iter<T: IntoIterator<Item = Option<Column>>>(iter: T) -> Self {
        ListChunked::from_iter(
            iter.into_iter()
                .map(|c| c.map(|c| c.take_materialized_series())),
        )
    }
}

impl FromIterator<Option<Series>> for ListChunked {
    #[inline]
    fn from_iter<I: IntoIterator<Item = Option<Series>>>(iter: I) -> Self {
        let mut it = iter.into_iter();
        let capacity = get_iter_capacity(&it);

        // get first non None from iter
        let first_value;
        let mut init_null_count = 0;
        loop {
            match it.next() {
                Some(Some(s)) => {
                    first_value = Some(s);
                    break;
                },
                Some(None) => {
                    init_null_count += 1;
                },
                None => return ListChunked::full_null(PlSmallStr::EMPTY, init_null_count),
            }
        }

        match first_value {
            None => {
                // already returned full_null above
                unreachable!()
            },
            Some(ref first_s) => {
                // AnyValues with empty lists in python can create
                // Series of an unknown dtype.
                // We use the anonymousbuilder without a dtype
                // the empty arrays is then not added (we add an extra offset instead)
                // the next non-empty series then must have the correct dtype.
                if matches!(first_s.dtype(), DataType::Null) && first_s.is_empty() {
                    let mut builder =
                        AnonymousOwnedListBuilder::new(PlSmallStr::EMPTY, capacity, None);
                    for _ in 0..init_null_count {
                        builder.append_null();
                    }
                    builder.append_empty();

                    for opt_s in it {
                        builder.append_opt_series(opt_s.as_ref()).unwrap();
                    }
                    builder.finish()
                } else {
                    // We don't know the needed capacity. We arbitrarily choose an average of 5 elements per series.
                    let mut builder = get_list_builder(
                        first_s.dtype(),
                        capacity * 5,
                        capacity,
                        PlSmallStr::EMPTY,
                    );

                    for _ in 0..init_null_count {
                        builder.append_null();
                    }
                    builder.append_series(first_s).unwrap();

                    for opt_s in it {
                        builder.append_opt_series(opt_s.as_ref()).unwrap();
                    }
                    builder.finish()
                }
            },
        }
    }
}

impl FromIterator<Option<Box<dyn Array>>> for ListChunked {
    #[inline]
    fn from_iter<I: IntoIterator<Item = Option<Box<dyn Array>>>>(iter: I) -> Self {
        iter.into_iter().collect_ca(PlSmallStr::EMPTY)
    }
}

#[cfg(feature = "object")]
impl<T: PolarsObject> FromIterator<Option<T>> for ObjectChunked<T> {
    fn from_iter<I: IntoIterator<Item = Option<T>>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let size = iter.size_hint().0;
        let mut null_mask_builder = BitmapBuilder::with_capacity(size);

        let values: Vec<T> = iter
            .map(|value| match value {
                Some(value) => {
                    null_mask_builder.push(true);
                    value
                },
                None => {
                    null_mask_builder.push(false);
                    T::default()
                },
            })
            .collect();

        let arr = Box::new(
            ObjectArray::from(values).with_validity(null_mask_builder.into_opt_validity()),
        );
        ChunkedArray::new_with_compute_len(
            Arc::new(Field::new(PlSmallStr::EMPTY, get_object_type::<T>())),
            vec![arr],
        )
    }
}
