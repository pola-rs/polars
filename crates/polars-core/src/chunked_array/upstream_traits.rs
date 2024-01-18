//! Implementations of upstream traits for [`ChunkedArray<T>`]
use std::borrow::{Borrow, Cow};
use std::collections::LinkedList;
use std::iter::FromIterator;
use std::marker::PhantomData;
use std::sync::Arc;

use arrow::array::{BooleanArray, PrimitiveArray};
use arrow::bitmap::{Bitmap, MutableBitmap};
use polars_utils::sync::SyncPtr;
use rayon::iter::{FromParallelIterator, IntoParallelIterator};
use rayon::prelude::*;

use crate::chunked_array::builder::{
    get_list_builder, AnonymousListBuilder, AnonymousOwnedListBuilder,
};
#[cfg(feature = "dtype-array")]
use crate::chunked_array::builder::{AnonymousOwnedFixedSizeListBuilder, FixedSizeListBuilder};
#[cfg(feature = "object")]
use crate::chunked_array::object::builder::get_object_type;
#[cfg(feature = "object")]
use crate::chunked_array::object::ObjectArray;
use crate::prelude::*;
use crate::utils::flatten::flatten_par;
use crate::utils::{get_iter_capacity, CustomIterTools, NoNull};

impl<T: PolarsDataType> Default for ChunkedArray<T> {
    fn default() -> Self {
        ChunkedArray {
            field: Arc::new(Field::new("default", DataType::Null)),
            chunks: Default::default(),
            phantom: PhantomData,
            bit_settings: Default::default(),
            length: 0,
            null_count: 0,
        }
    }
}

/// FromIterator trait
impl<T> FromIterator<Option<T::Native>> for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn from_iter<I: IntoIterator<Item = Option<T::Native>>>(iter: I) -> Self {
        // TODO: eliminate this FromIterator implementation entirely.
        iter.into_iter().collect_ca("")
    }
}

// NoNull is only a wrapper needed for specialization
impl<T> FromIterator<T::Native> for NoNull<ChunkedArray<T>>
where
    T: PolarsNumericType,
{
    // We use Vec because it is way faster than Arrows builder. We can do this because we
    // know we don't have null values.
    fn from_iter<I: IntoIterator<Item = T::Native>>(iter: I) -> Self {
        // 2021-02-07: aligned vec was ~2x faster than arrow collect.
        let av = iter.into_iter().collect::<Vec<T::Native>>();
        NoNull::new(ChunkedArray::from_vec("", av))
    }
}

impl FromIterator<Option<bool>> for ChunkedArray<BooleanType> {
    fn from_iter<I: IntoIterator<Item = Option<bool>>>(iter: I) -> Self {
        BooleanArray::from_iter(iter).into()
    }
}

impl FromIterator<bool> for BooleanChunked {
    fn from_iter<I: IntoIterator<Item = bool>>(iter: I) -> Self {
        // 2021-02-07: this was ~70% faster than with the builder, even with the extra Option<T> added.
        BooleanArray::from_iter(iter.into_iter().map(Some)).into()
    }
}

impl FromIterator<bool> for NoNull<BooleanChunked> {
    fn from_iter<I: IntoIterator<Item = bool>>(iter: I) -> Self {
        let ca = iter.into_iter().collect::<BooleanChunked>();
        NoNull::new(ca)
    }
}

// FromIterator for StringChunked variants.

impl<Ptr> FromIterator<Option<Ptr>> for StringChunked
where
    Ptr: AsRef<str>,
{
    fn from_iter<I: IntoIterator<Item = Option<Ptr>>>(iter: I) -> Self {
        let arr = MutableBinaryViewArray::from_iterator(iter.into_iter()).freeze();
        ChunkedArray::with_chunk("", arr)
    }
}

/// Local [`AsRef<T>`] trait to circumvent the orphan rule.
pub trait PolarsAsRef<T: ?Sized>: AsRef<T> {}

impl PolarsAsRef<str> for String {}
impl PolarsAsRef<str> for &str {}
// &["foo", "bar"]
impl PolarsAsRef<str> for &&str {}

impl<'a> PolarsAsRef<str> for Cow<'a, str> {}
impl PolarsAsRef<[u8]> for Vec<u8> {}
impl PolarsAsRef<[u8]> for &[u8] {}
// TODO: remove!
impl PolarsAsRef<[u8]> for &&[u8] {}
impl<'a> PolarsAsRef<[u8]> for Cow<'a, [u8]> {}

impl<Ptr> FromIterator<Ptr> for StringChunked
where
    Ptr: PolarsAsRef<str>,
{
    fn from_iter<I: IntoIterator<Item = Ptr>>(iter: I) -> Self {
        let arr = MutableBinaryViewArray::from_values_iter(iter.into_iter()).freeze();
        ChunkedArray::with_chunk("", arr)
    }
}

// FromIterator for BinaryChunked variants.
impl<Ptr> FromIterator<Option<Ptr>> for BinaryChunked
where
    Ptr: AsRef<[u8]>,
{
    fn from_iter<I: IntoIterator<Item = Option<Ptr>>>(iter: I) -> Self {
        let arr = MutableBinaryViewArray::from_iter(iter.into_iter()).freeze();
        ChunkedArray::with_chunk("", arr)
    }
}

impl<Ptr> FromIterator<Ptr> for BinaryChunked
where
    Ptr: PolarsAsRef<[u8]>,
{
    fn from_iter<I: IntoIterator<Item = Ptr>>(iter: I) -> Self {
        let arr = MutableBinaryViewArray::from_values_iter(iter.into_iter()).freeze();
        ChunkedArray::with_chunk("", arr)
    }
}

impl<Ptr> FromIterator<Ptr> for ListChunked
where
    Ptr: Borrow<Series>,
{
    fn from_iter<I: IntoIterator<Item = Ptr>>(iter: I) -> Self {
        let mut it = iter.into_iter();
        let capacity = get_iter_capacity(&it);

        // first take one to get the dtype.
        let v = match it.next() {
            Some(v) => v,
            None => return ListChunked::full_null("", 0),
        };
        // We don't know the needed capacity. We arbitrarily choose an average of 5 elements per series.
        let mut builder =
            get_list_builder(v.borrow().dtype(), capacity * 5, capacity, "collected").unwrap();

        builder.append_series(v.borrow()).unwrap();
        for s in it {
            builder.append_series(s.borrow()).unwrap();
        }
        builder.finish()
    }
}

impl FromIterator<Option<Series>> for ListChunked {
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
                None => return ListChunked::full_null("", init_null_count),
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
                    let mut builder = AnonymousOwnedListBuilder::new("collected", capacity, None);
                    for _ in 0..init_null_count {
                        builder.append_null();
                    }
                    builder.append_empty();

                    for opt_s in it {
                        builder.append_opt_series(opt_s.as_ref()).unwrap();
                    }
                    builder.finish()
                } else {
                    match first_s.dtype() {
                        #[cfg(feature = "object")]
                        DataType::Object(_, _) => {
                            let mut builder =
                                first_s.get_list_builder("collected", capacity * 5, capacity);
                            for _ in 0..init_null_count {
                                builder.append_null();
                            }
                            builder.append_series(first_s).unwrap();

                            for opt_s in it {
                                builder.append_opt_series(opt_s.as_ref()).unwrap();
                            }
                            builder.finish()
                        },
                        _ => {
                            // We don't know the needed capacity. We arbitrarily choose an average of 5 elements per series.
                            let mut builder = get_list_builder(
                                first_s.dtype(),
                                capacity * 5,
                                capacity,
                                "collected",
                            )
                            .unwrap();

                            for _ in 0..init_null_count {
                                builder.append_null();
                            }
                            builder.append_series(first_s).unwrap();

                            for opt_s in it {
                                builder.append_opt_series(opt_s.as_ref()).unwrap();
                            }
                            builder.finish()
                        },
                    }
                }
            },
        }
    }
}

impl FromIterator<Option<Box<dyn Array>>> for ListChunked {
    fn from_iter<I: IntoIterator<Item = Option<Box<dyn Array>>>>(iter: I) -> Self {
        let mut cap = 0;
        let mut dtype: Option<DataType> = None;
        let vals = iter
            .into_iter()
            .map(|opt_arr| {
                opt_arr.map(|arr| {
                    if dtype.is_none() {
                        dtype = Some(arr.data_type().into());
                    }
                    cap += arr.len();
                    arr
                })
            })
            .collect::<Vec<_>>();

        let mut builder = AnonymousListBuilder::new("collected", cap, None);
        for val in &vals {
            builder.append_opt_array(val.as_deref());
        }
        builder.finish()
    }
}

#[cfg(feature = "dtype-array")]
impl ArrayChunked {
    pub(crate) unsafe fn from_iter_and_args<I: IntoIterator<Item = Option<Box<dyn Array>>>>(
        iter: I,
        width: usize,
        capacity: usize,
        inner_dtype: Option<DataType>,
        name: &str,
    ) -> Self {
        let mut builder =
            AnonymousOwnedFixedSizeListBuilder::new(name, width, capacity, inner_dtype);
        for val in iter {
            match val {
                None => builder.push_null(),
                Some(arr) => builder.push_unchecked(arr.as_ref(), 0),
            }
        }
        builder.finish()
    }
}

#[cfg(feature = "object")]
impl<T: PolarsObject> FromIterator<Option<T>> for ObjectChunked<T> {
    fn from_iter<I: IntoIterator<Item = Option<T>>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let size = iter.size_hint().0;
        let mut null_mask_builder = MutableBitmap::with_capacity(size);

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

        let null_bit_buffer: Option<Bitmap> = null_mask_builder.into();
        let null_bitmap = null_bit_buffer;

        let len = values.len();

        let arr = Box::new(ObjectArray {
            values: Arc::new(values),
            null_bitmap,
            offset: 0,
            len,
        });
        let mut out = ChunkedArray {
            field: Arc::new(Field::new("", get_object_type::<T>())),
            chunks: vec![arr],
            phantom: PhantomData,
            bit_settings: Default::default(),
            length: 0,
            null_count: 0,
        };
        out.compute_len();
        out
    }
}

/// FromParallelIterator trait
// Code taken from https://docs.rs/rayon/1.3.1/src/rayon/iter/extend.rs.html#356-366
fn vec_push<T>(mut vec: Vec<T>, elem: T) -> Vec<T> {
    vec.push(elem);
    vec
}

fn as_list<T>(item: T) -> LinkedList<T> {
    let mut list = LinkedList::new();
    list.push_back(item);
    list
}

fn list_append<T>(mut list1: LinkedList<T>, mut list2: LinkedList<T>) -> LinkedList<T> {
    list1.append(&mut list2);
    list1
}

fn collect_into_linked_list<I>(par_iter: I) -> LinkedList<Vec<I::Item>>
where
    I: IntoParallelIterator,
{
    let it = par_iter.into_par_iter();
    // be careful optimizing allocations. Its hard to figure out the size
    // needed
    // https://github.com/pola-rs/polars/issues/1562
    it.fold(Vec::new, vec_push)
        .map(as_list)
        .reduce(LinkedList::new, list_append)
}

fn get_capacity_from_par_results<T>(ll: &LinkedList<Vec<T>>) -> usize {
    ll.iter().map(|list| list.len()).sum()
}

fn get_capacity_from_par_results_slice<T>(bufs: &[Vec<T>]) -> usize {
    bufs.iter().map(|list| list.len()).sum()
}
fn get_offsets<T>(bufs: &[Vec<T>]) -> Vec<usize> {
    bufs.iter()
        .scan(0usize, |acc, buf| {
            let out = *acc;
            *acc += buf.len();
            Some(out)
        })
        .collect()
}

impl<T> FromParallelIterator<T::Native> for NoNull<ChunkedArray<T>>
where
    T: PolarsNumericType,
{
    fn from_par_iter<I: IntoParallelIterator<Item = T::Native>>(iter: I) -> Self {
        // Get linkedlist filled with different vec result from different threads
        let vectors = collect_into_linked_list(iter);
        let vectors = vectors.into_iter().collect::<Vec<_>>();
        let values = flatten_par(&vectors);
        NoNull::new(ChunkedArray::new_vec("", values))
    }
}

fn finish_validities(validities: Vec<(Option<Bitmap>, usize)>, capacity: usize) -> Option<Bitmap> {
    if validities.iter().any(|(v, _)| v.is_some()) {
        let mut bitmap = MutableBitmap::with_capacity(capacity);
        for (valids, len) in validities {
            if let Some(valids) = valids {
                bitmap.extend_from_bitmap(&(valids))
            } else {
                bitmap.extend_constant(len, true)
            }
        }
        Some(bitmap.into())
    } else {
        None
    }
}

impl<T> FromParallelIterator<Option<T::Native>> for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn from_par_iter<I: IntoParallelIterator<Item = Option<T::Native>>>(iter: I) -> Self {
        // Get linkedlist filled with different vec result from different threads
        let vectors = collect_into_linked_list(iter);

        let vectors = vectors.into_iter().collect::<Vec<_>>();
        let capacity: usize = get_capacity_from_par_results_slice(&vectors);
        let offsets = get_offsets(&vectors);

        let mut values_buf: Vec<T::Native> = Vec::with_capacity(capacity);
        let values_buf_ptr = unsafe { SyncPtr::new(values_buf.as_mut_ptr()) };

        let validities = offsets
            .into_par_iter()
            .zip(vectors)
            .map(|(offset, vector)| {
                let mut local_validity = None;
                let local_len = vector.len();
                let mut latest_validy_written = 0;
                unsafe {
                    let values_buf_ptr = values_buf_ptr.get().add(offset);

                    for (i, opt_v) in vector.into_iter().enumerate() {
                        match opt_v {
                            Some(v) => {
                                std::ptr::write(values_buf_ptr.add(i), v);
                            },
                            None => {
                                let validity = match &mut local_validity {
                                    None => {
                                        let validity = MutableBitmap::with_capacity(local_len);
                                        local_validity = Some(validity);
                                        local_validity.as_mut().unwrap_unchecked()
                                    },
                                    Some(validity) => validity,
                                };
                                validity.extend_constant(i - latest_validy_written, true);
                                latest_validy_written = i + 1;
                                validity.push_unchecked(false);
                                // initialize value
                                std::ptr::write(values_buf_ptr.add(i), T::Native::default());
                            },
                        }
                    }
                }
                if let Some(validity) = &mut local_validity {
                    validity.extend_constant(local_len - latest_validy_written, true);
                }
                (local_validity.map(|b| b.into()), local_len)
            })
            .collect::<Vec<_>>();
        unsafe { values_buf.set_len(capacity) };

        let validity = finish_validities(validities, capacity);

        let arr = PrimitiveArray::from_data_default(values_buf.into(), validity);
        arr.into()
    }
}

impl FromParallelIterator<bool> for BooleanChunked {
    fn from_par_iter<I: IntoParallelIterator<Item = bool>>(iter: I) -> Self {
        let vectors = collect_into_linked_list(iter);

        let capacity: usize = get_capacity_from_par_results(&vectors);

        let arr = unsafe {
            BooleanArray::from_trusted_len_values_iter(
                vectors.into_iter().flatten().trust_my_length(capacity),
            )
        };
        arr.into()
    }
}

impl FromParallelIterator<Option<bool>> for BooleanChunked {
    fn from_par_iter<I: IntoParallelIterator<Item = Option<bool>>>(iter: I) -> Self {
        // Get linkedlist filled with different vec result from different threads.
        let vectors = collect_into_linked_list(iter);
        let vectors = vectors.into_iter().collect::<Vec<_>>();
        let chunks: Vec<BooleanArray> = vectors
            .into_par_iter()
            .map(|vector| vector.into())
            .collect();
        Self::from_chunk_iter("", chunks).rechunk()
    }
}

impl<Ptr> FromParallelIterator<Ptr> for StringChunked
where
    Ptr: PolarsAsRef<str> + Send + Sync,
{
    fn from_par_iter<I: IntoParallelIterator<Item = Ptr>>(iter: I) -> Self {
        let vectors = collect_into_linked_list(iter);
        let cap = get_capacity_from_par_results(&vectors);

        let mut builder = MutableBinaryViewArray::with_capacity(cap);
        // TODO! we can do this in parallel ind just combine the buffers.
        for vec in vectors {
            for val in vec {
                builder.push_value_ignore_validity(val.as_ref())
            }
        }
        ChunkedArray::with_chunk("", builder.freeze())
    }
}

impl<Ptr> FromParallelIterator<Ptr> for BinaryChunked
where
    Ptr: PolarsAsRef<[u8]> + Send + Sync,
{
    fn from_par_iter<I: IntoParallelIterator<Item = Ptr>>(iter: I) -> Self {
        let vectors = collect_into_linked_list(iter);
        let cap = get_capacity_from_par_results(&vectors);

        let mut builder = MutableBinaryViewArray::with_capacity(cap);
        // TODO! we can do this in parallel ind just combine the buffers.
        for vec in vectors {
            for val in vec {
                builder.push_value_ignore_validity(val.as_ref())
            }
        }
        ChunkedArray::with_chunk("", builder.freeze())
    }
}

impl<Ptr> FromParallelIterator<Option<Ptr>> for StringChunked
where
    Ptr: AsRef<str> + Send + Sync,
{
    fn from_par_iter<I: IntoParallelIterator<Item = Option<Ptr>>>(iter: I) -> Self {
        let vectors = collect_into_linked_list(iter);
        let vectors = vectors.into_iter().collect::<Vec<_>>();

        let arrays = vectors
            .into_par_iter()
            .map(|vector| {
                let cap = vector.len();
                let mut mutable = MutableBinaryViewArray::with_capacity(cap);
                for opt_val in vector {
                    mutable.push(opt_val)
                }
                mutable.freeze()
            })
            .collect::<Vec<_>>();

        // TODO!
        // do this in parallel.
        let arrays = arrays
            .iter()
            .map(|arr| arr as &dyn Array)
            .collect::<Vec<_>>();
        let arr = arrow::compute::concatenate::concatenate(&arrays).unwrap();
        unsafe { StringChunked::from_chunks("", vec![arr]) }
    }
}

impl<Ptr> FromParallelIterator<Option<Ptr>> for BinaryChunked
where
    Ptr: AsRef<[u8]> + Send + Sync,
{
    fn from_par_iter<I: IntoParallelIterator<Item = Option<Ptr>>>(iter: I) -> Self {
        let vectors = collect_into_linked_list(iter);
        let vectors = vectors.into_iter().collect::<Vec<_>>();

        let arrays = vectors
            .into_par_iter()
            .map(|vector| {
                let cap = vector.len();
                let mut mutable = MutableBinaryViewArray::with_capacity(cap);
                for opt_val in vector {
                    mutable.push(opt_val)
                }
                mutable.freeze()
            })
            .collect::<Vec<_>>();

        // TODO!
        // do this in parallel.
        let arrays = arrays
            .iter()
            .map(|arr| arr as &dyn Array)
            .collect::<Vec<_>>();
        let arr = arrow::compute::concatenate::concatenate(&arrays).unwrap();
        unsafe { BinaryChunked::from_chunks("", vec![arr]) }
    }
}

/// From trait
impl<'a> From<&'a StringChunked> for Vec<Option<&'a str>> {
    fn from(ca: &'a StringChunked) -> Self {
        ca.into_iter().collect()
    }
}

impl From<StringChunked> for Vec<Option<String>> {
    fn from(ca: StringChunked) -> Self {
        ca.into_iter()
            .map(|opt| opt.map(|s| s.to_string()))
            .collect()
    }
}

impl<'a> From<&'a BooleanChunked> for Vec<Option<bool>> {
    fn from(ca: &'a BooleanChunked) -> Self {
        ca.into_iter().collect()
    }
}

impl From<BooleanChunked> for Vec<Option<bool>> {
    fn from(ca: BooleanChunked) -> Self {
        ca.into_iter().collect()
    }
}

impl<'a, T> From<&'a ChunkedArray<T>> for Vec<Option<T::Native>>
where
    T: PolarsNumericType,
{
    fn from(ca: &'a ChunkedArray<T>) -> Self {
        ca.into_iter().collect()
    }
}

impl FromParallelIterator<Option<Series>> for ListChunked {
    fn from_par_iter<I>(iter: I) -> Self
    where
        I: IntoParallelIterator<Item = Option<Series>>,
    {
        let mut dtype = None;
        let vectors = collect_into_linked_list(iter);

        let list_capacity: usize = get_capacity_from_par_results(&vectors);
        let value_capacity = vectors
            .iter()
            .map(|list| {
                list.iter()
                    .map(|opt_s| {
                        opt_s
                            .as_ref()
                            .map(|s| {
                                if dtype.is_none() && !matches!(s.dtype(), DataType::Null) {
                                    dtype = Some(s.dtype().clone())
                                }
                                s.len()
                            })
                            .unwrap_or(0)
                    })
                    .sum::<usize>()
            })
            .sum::<usize>();

        match &dtype {
            #[cfg(feature = "object")]
            Some(DataType::Object(_, _)) => {
                let s = vectors
                    .iter()
                    .flatten()
                    .find_map(|opt_s| opt_s.as_ref())
                    .unwrap();
                let mut builder = s.get_list_builder("collected", value_capacity, list_capacity);

                for v in vectors {
                    for val in v {
                        builder.append_opt_series(val.as_ref()).unwrap();
                    }
                }
                builder.finish()
            },
            Some(dtype) => {
                let mut builder =
                    get_list_builder(dtype, value_capacity, list_capacity, "collected").unwrap();
                for v in &vectors {
                    for val in v {
                        builder.append_opt_series(val.as_ref()).unwrap();
                    }
                }
                builder.finish()
            },
            None => ListChunked::full_null_with_dtype("collected", list_capacity, &DataType::Null),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    fn test_collect_into_list() {
        let s1 = Series::new("", &[true, false, true]);
        let s2 = Series::new("", &[true, false, true]);

        let ll: ListChunked = [&s1, &s2].iter().copied().collect();
        assert_eq!(ll.len(), 2);
        assert_eq!(ll.null_count(), 0);
        let ll: ListChunked = [None, Some(s2)].into_iter().collect();
        assert_eq!(ll.len(), 2);
        assert_eq!(ll.null_count(), 1);
    }
}
