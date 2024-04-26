//! Implementations of upstream traits for [`ChunkedArray<T>`]
use std::collections::LinkedList;

use arrow::bitmap::{Bitmap, MutableBitmap};
use polars_utils::sync::SyncPtr;
use rayon::prelude::*;

use super::from_iterator::PolarsAsRef;
use crate::chunked_array::builder::get_list_builder;
use crate::prelude::*;
use crate::utils::flatten::flatten_par;
use crate::utils::NoNull;

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

impl<'a, T> From<&'a ChunkedArray<T>> for Vec<Option<T::Physical<'a>>>
where
    T: PolarsDataType,
{
    fn from(ca: &'a ChunkedArray<T>) -> Self {
        let mut out = Vec::with_capacity(ca.len());
        for arr in ca.downcast_iter() {
            out.extend(arr.iter())
        }
        out
    }
}
impl From<StringChunked> for Vec<Option<String>> {
    fn from(ca: StringChunked) -> Self {
        ca.iter().map(|opt| opt.map(|s| s.to_string())).collect()
    }
}

impl From<BooleanChunked> for Vec<Option<bool>> {
    fn from(ca: BooleanChunked) -> Self {
        let mut out = Vec::with_capacity(ca.len());
        for arr in ca.downcast_iter() {
            out.extend(arr.iter())
        }
        out
    }
}

/// From trait
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
