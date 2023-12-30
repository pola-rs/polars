use arrow::legacy::kernels::concatenate::concatenate_owned_unchecked;
use arrow::offset::Offsets;
use polars_utils::unwrap::UnwrapUncheckedRelease;

use super::*;
#[cfg(feature = "dtype-struct")]
use crate::chunked_array::builder::AnonymousOwnedListBuilder;

pub trait AggList {
    /// # Safety
    ///
    /// groups should be in bounds
    unsafe fn agg_list(&self, _groups: &GroupsProxy) -> Series;
}

impl<T> AggList for ChunkedArray<T>
where
    T: PolarsNumericType,
    ChunkedArray<T>: IntoSeries,
{
    unsafe fn agg_list(&self, groups: &GroupsProxy) -> Series {
        let ca = self.rechunk();

        match groups {
            GroupsProxy::Idx(groups) => {
                let mut can_fast_explode = true;

                let arr = ca.downcast_iter().next().unwrap();
                let values = arr.values();

                let mut offsets = Vec::<i64>::with_capacity(groups.len() + 1);
                let mut length_so_far = 0i64;
                offsets.push(length_so_far);

                let mut list_values = Vec::<T::Native>::with_capacity(self.len());
                groups.iter().for_each(|(_, idx)| {
                    let idx_len = idx.len();
                    if idx_len == 0 {
                        can_fast_explode = false;
                    }

                    length_so_far += idx_len as i64;
                    // Safety:
                    // group tuples are in bounds
                    {
                        list_values.extend(idx.iter().map(|idx| {
                            debug_assert!((*idx as usize) < values.len());
                            *values.get_unchecked(*idx as usize)
                        }));
                        // Safety:
                        // we know that offsets has allocated enough slots
                        offsets.push_unchecked(length_so_far);
                    }
                });

                let validity = if arr.null_count() > 0 {
                    let old_validity = arr.validity().unwrap();
                    let mut validity = MutableBitmap::from_len_set(list_values.len());

                    let mut count = 0;
                    groups.iter().for_each(|(_, idx)| {
                        for i in idx.as_slice() {
                            if !old_validity.get_bit_unchecked(*i as usize) {
                                validity.set_bit_unchecked(count, false)
                            }
                            count += 1;
                        }
                    });
                    Some(validity.into())
                } else {
                    None
                };

                let array =
                    PrimitiveArray::new(T::get_dtype().to_arrow(), list_values.into(), validity);
                let data_type = ListArray::<i64>::default_datatype(T::get_dtype().to_arrow());
                // Safety:
                // offsets are monotonically increasing
                let arr = ListArray::<i64>::new(
                    data_type,
                    Offsets::new_unchecked(offsets).into(),
                    Box::new(array),
                    None,
                );

                let mut ca = ListChunked::with_chunk(self.name(), arr);
                if can_fast_explode {
                    ca.set_fast_explode()
                }
                ca.into()
            },
            GroupsProxy::Slice { groups, .. } => {
                let mut can_fast_explode = true;
                let arr = ca.downcast_iter().next().unwrap();
                let values = arr.values();

                let mut offsets = Vec::<i64>::with_capacity(groups.len() + 1);
                let mut length_so_far = 0i64;
                offsets.push(length_so_far);

                let mut list_values = Vec::<T::Native>::with_capacity(self.len());
                groups.iter().for_each(|&[first, len]| {
                    if len == 0 {
                        can_fast_explode = false;
                    }

                    length_so_far += len as i64;
                    list_values.extend_from_slice(&values[first as usize..(first + len) as usize]);
                    {
                        // Safety:
                        // we know that offsets has allocated enough slots
                        offsets.push_unchecked(length_so_far);
                    }
                });

                let validity = if arr.null_count() > 0 {
                    let old_validity = arr.validity().unwrap();
                    let mut validity = MutableBitmap::from_len_set(list_values.len());

                    let mut count = 0;
                    groups.iter().for_each(|[first, len]| {
                        for i in *first..(*first + *len) {
                            if !old_validity.get_bit_unchecked(i as usize) {
                                validity.set_bit_unchecked(count, false)
                            }
                            count += 1;
                        }
                    });
                    Some(validity.into())
                } else {
                    None
                };

                let array =
                    PrimitiveArray::new(T::get_dtype().to_arrow(), list_values.into(), validity);
                let data_type = ListArray::<i64>::default_datatype(T::get_dtype().to_arrow());
                let arr = ListArray::<i64>::new(
                    data_type,
                    Offsets::new_unchecked(offsets).into(),
                    Box::new(array),
                    None,
                );
                let mut ca = ListChunked::with_chunk(self.name(), arr);
                if can_fast_explode {
                    ca.set_fast_explode()
                }
                ca.into()
            },
        }
    }
}

impl AggList for BooleanChunked {
    unsafe fn agg_list(&self, groups: &GroupsProxy) -> Series {
        match groups {
            GroupsProxy::Idx(groups) => {
                let mut builder =
                    ListBooleanChunkedBuilder::new(self.name(), groups.len(), self.len());
                for idx in groups.all().iter() {
                    let ca = { self.take_unchecked(idx) };
                    builder.append(&ca)
                }
                builder.finish().into_series()
            },
            GroupsProxy::Slice { groups, .. } => {
                let mut builder =
                    ListBooleanChunkedBuilder::new(self.name(), groups.len(), self.len());
                for [first, len] in groups {
                    let ca = self.slice(*first as i64, *len as usize);
                    builder.append(&ca)
                }
                builder.finish().into_series()
            },
        }
    }
}

impl AggList for StringChunked {
    unsafe fn agg_list(&self, groups: &GroupsProxy) -> Series {
        // TODO: dispatch via binary
        match groups {
            GroupsProxy::Idx(groups) => {
                let mut builder =
                    ListStringChunkedBuilder::new(self.name(), groups.len(), self.len());
                for idx in groups.all().iter() {
                    let ca = { self.take_unchecked(idx) };
                    builder.append(&ca)
                }
                builder.finish().into_series()
            },
            GroupsProxy::Slice { groups, .. } => {
                let mut builder =
                    ListStringChunkedBuilder::new(self.name(), groups.len(), self.len());
                for [first, len] in groups {
                    let ca = self.slice(*first as i64, *len as usize);
                    builder.append(&ca)
                }
                builder.finish().into_series()
            },
        }
    }
}

impl AggList for BinaryChunked {
    unsafe fn agg_list(&self, groups: &GroupsProxy) -> Series {
        match groups {
            GroupsProxy::Idx(groups) => {
                let mut builder =
                    ListBinaryChunkedBuilder::new(self.name(), groups.len(), self.len());
                for idx in groups.all().iter() {
                    let ca = { self.take_unchecked(idx) };
                    builder.append(&ca)
                }
                builder.finish().into_series()
            },
            GroupsProxy::Slice { groups, .. } => {
                let mut builder =
                    ListBinaryChunkedBuilder::new(self.name(), groups.len(), self.len());
                for [first, len] in groups {
                    let ca = self.slice(*first as i64, *len as usize);
                    builder.append(&ca)
                }
                builder.finish().into_series()
            },
        }
    }
}

/// This aggregates into a [`ListChunked`] by slicing the array that is aggregated.
/// Used for [`List`] and [`Array`] data types.
fn agg_list_by_slicing<
    A: PolarsDataType,
    F: Fn(&ChunkedArray<A>, bool, &mut Vec<i64>, &mut i64, &mut Vec<ArrayRef>) -> bool,
>(
    ca: &ChunkedArray<A>,
    dtype: DataType,
    groups_len: usize,
    func: F,
) -> Series {
    let can_fast_explode = true;
    let mut offsets = Vec::<i64>::with_capacity(groups_len + 1);
    let mut length_so_far = 0i64;
    offsets.push(length_so_far);

    let mut list_values = Vec::with_capacity(groups_len);

    let can_fast_explode = func(
        ca,
        can_fast_explode,
        &mut offsets,
        &mut length_so_far,
        &mut list_values,
    );
    if groups_len == 0 {
        list_values.push(ca.chunks[0].sliced(0, 0))
    }
    let list_values = concatenate_owned_unchecked(&list_values).unwrap();
    let data_type = ListArray::<i64>::default_datatype(list_values.data_type().clone());
    // SAFETY:
    // offsets are monotonically increasing
    let arr = ListArray::<i64>::new(
        data_type,
        unsafe { Offsets::new_unchecked(offsets).into() },
        list_values,
        None,
    );
    let mut listarr = ListChunked::with_chunk(ca.name(), arr);
    if can_fast_explode {
        listarr.set_fast_explode()
    }
    unsafe { listarr.to_logical(dtype) };
    listarr.into_series()
}

impl AggList for ListChunked {
    unsafe fn agg_list(&self, groups: &GroupsProxy) -> Series {
        match groups {
            GroupsProxy::Idx(groups) => {
                let func = |ca: &ListChunked,
                            mut can_fast_explode: bool,
                            offsets: &mut Vec<i64>,
                            length_so_far: &mut i64,
                            list_values: &mut Vec<ArrayRef>| {
                    assert!(list_values.capacity() >= groups.len());
                    groups.iter().for_each(|(_, idx)| {
                        let idx_len = idx.len();
                        if idx_len == 0 {
                            can_fast_explode = false;
                        }

                        *length_so_far += idx_len as i64;
                        // SAFETY:
                        // group tuples are in bounds
                        {
                            let mut s = ca.take_unchecked(idx);
                            let arr = s.chunks.pop().unwrap_unchecked_release();
                            list_values.push_unchecked(arr);

                            // SAFETY:
                            // we know that offsets has allocated enough slots
                            offsets.push_unchecked(*length_so_far);
                        }
                    });
                    can_fast_explode
                };

                agg_list_by_slicing(self, self.dtype().clone(), groups.len(), func)
            },
            GroupsProxy::Slice { groups, .. } => {
                let func = |ca: &ListChunked,
                            mut can_fast_explode: bool,
                            offsets: &mut Vec<i64>,
                            length_so_far: &mut i64,
                            list_values: &mut Vec<ArrayRef>| {
                    assert!(list_values.capacity() >= groups.len());
                    groups.iter().for_each(|&[first, len]| {
                        if len == 0 {
                            can_fast_explode = false;
                        }

                        *length_so_far += len as i64;
                        let mut s = ca.slice(first as i64, len as usize);
                        let arr = s.chunks.pop().unwrap_unchecked_release();
                        list_values.push_unchecked(arr);

                        {
                            // SAFETY:
                            // we know that offsets has allocated enough slots
                            offsets.push_unchecked(*length_so_far);
                        }
                    });
                    can_fast_explode
                };

                agg_list_by_slicing(self, self.dtype().clone(), groups.len(), func)
            },
        }
    }
}

#[cfg(feature = "dtype-array")]
impl AggList for ArrayChunked {
    unsafe fn agg_list(&self, groups: &GroupsProxy) -> Series {
        match groups {
            GroupsProxy::Idx(groups) => {
                let func = |ca: &ArrayChunked,
                            mut can_fast_explode: bool,
                            offsets: &mut Vec<i64>,
                            length_so_far: &mut i64,
                            list_values: &mut Vec<ArrayRef>| {
                    assert!(list_values.capacity() >= groups.len());
                    groups.iter().for_each(|(_, idx)| {
                        let idx_len = idx.len();
                        if idx_len == 0 {
                            can_fast_explode = false;
                        }

                        *length_so_far += idx_len as i64;

                        // SAFETY: we know that offsets has allocated enough slots
                        offsets.push_unchecked(*length_so_far);

                        // SAFETY: group tuples are in bounds
                        {
                            let mut s = ca.take_unchecked(idx);
                            let arr = s.chunks.pop().unwrap_unchecked_release();
                            list_values.push_unchecked(arr);
                        }
                    });
                    can_fast_explode
                };

                agg_list_by_slicing(self, self.dtype().clone(), groups.len(), func)
            },
            GroupsProxy::Slice { groups, .. } => {
                let func = |ca: &ArrayChunked,
                            mut can_fast_explode: bool,
                            offsets: &mut Vec<i64>,
                            length_so_far: &mut i64,
                            list_values: &mut Vec<ArrayRef>| {
                    assert!(list_values.capacity() >= groups.len());
                    groups.iter().for_each(|&[first, len]| {
                        if len == 0 {
                            can_fast_explode = false;
                        }

                        *length_so_far += len as i64;
                        // SAFETY:
                        // we know that offsets has allocated enough slots
                        offsets.push_unchecked(*length_so_far);

                        let mut s = ca.slice(first as i64, len as usize);
                        let arr = s.chunks.pop().unwrap_unchecked_release();
                        list_values.push_unchecked(arr);
                    });
                    can_fast_explode
                };

                agg_list_by_slicing(self, self.dtype().clone(), groups.len(), func)
            },
        }
    }
}

#[cfg(feature = "object")]
impl<T: PolarsObject> AggList for ObjectChunked<T> {
    unsafe fn agg_list(&self, groups: &GroupsProxy) -> Series {
        let mut can_fast_explode = true;
        let mut offsets = Vec::<i64>::with_capacity(groups.len() + 1);
        let mut length_so_far = 0i64;
        offsets.push(length_so_far);

        //  we know that iterators length
        let iter = {
            groups
                .iter()
                .flat_map(|indicator| {
                    let (group_vals, len) = match indicator {
                        GroupsIndicator::Idx((_first, idx)) => {
                            // SAFETY:
                            // group tuples always in bounds
                            let group_vals = self.take_unchecked(idx);

                            (group_vals, idx.len() as IdxSize)
                        },
                        GroupsIndicator::Slice([first, len]) => {
                            let group_vals = _slice_from_offsets(self, first, len);

                            (group_vals, len)
                        },
                    };

                    if len == 0 {
                        can_fast_explode = false;
                    }
                    length_so_far += len as i64;
                    // SAFETY:
                    // we know that offsets has allocated enough slots
                    offsets.push_unchecked(length_so_far);

                    let arr = group_vals.downcast_iter().next().unwrap().clone();
                    arr.into_iter_cloned()
                })
                .trust_my_length(self.len())
        };

        let mut pe = create_extension(iter);

        // SAFETY: This is safe because we just created the PolarsExtension
        // meaning that the sentinel is heap allocated and the dereference of
        // the pointer does not fail.
        pe.set_to_series_fn::<T>();
        let extension_array = Box::new(pe.take_and_forget()) as ArrayRef;
        let extension_dtype = extension_array.data_type();

        let data_type = ListArray::<i64>::default_datatype(extension_dtype.clone());
        // SAFETY: offsets are monotonically increasing.
        let arr = ListArray::<i64>::new(
            data_type,
            Offsets::new_unchecked(offsets).into(),
            extension_array,
            None,
        );
        let mut listarr = ListChunked::with_chunk(self.name(), arr);
        if can_fast_explode {
            listarr.set_fast_explode()
        }
        listarr.into_series()
    }
}

#[cfg(feature = "dtype-struct")]
impl AggList for StructChunked {
    unsafe fn agg_list(&self, groups: &GroupsProxy) -> Series {
        let s = self.clone().into_series();
        match groups {
            GroupsProxy::Idx(groups) => {
                let mut builder = AnonymousOwnedListBuilder::new(
                    self.name(),
                    groups.len(),
                    Some(self.dtype().clone()),
                );
                for idx in groups.all().iter() {
                    let taken = s.take_slice_unchecked(idx);
                    builder.append_series(&taken).unwrap();
                }
                builder.finish().into_series()
            },
            GroupsProxy::Slice { groups, .. } => {
                let mut builder = AnonymousOwnedListBuilder::new(
                    self.name(),
                    groups.len(),
                    Some(self.dtype().clone()),
                );
                for [first, len] in groups {
                    let taken = s.slice(*first as i64, *len as usize);
                    builder.append_series(&taken).unwrap();
                }
                builder.finish().into_series()
            },
        }
    }
}
