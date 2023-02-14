#[cfg(feature = "groupby_list")]
use polars_arrow::kernels::list_bytes_iter::numeric_list_bytes_iter;
use polars_arrow::kernels::sort_partition::{create_clean_partitions, partition_to_groups};
use polars_arrow::prelude::*;
use polars_utils::{flatten, HashSingle};

use super::*;
use crate::config::verbose;
use crate::utils::{_split_offsets, copy_from_slice_unchecked};

/// Used to create the tuples for a groupby operation.
pub trait IntoGroupsProxy {
    /// Create the tuples need for a groupby operation.
    ///     * The first value in the tuple is the first index of the group.
    ///     * The second value in the tuple is are the indexes of the groups including the first value.
    fn group_tuples(&self, _multithreaded: bool, _sorted: bool) -> PolarsResult<GroupsProxy> {
        unimplemented!()
    }
}

fn group_multithreaded<T: PolarsDataType>(ca: &ChunkedArray<T>) -> bool {
    // TODO! change to something sensible
    ca.len() > 1000
}

fn num_groups_proxy<T>(ca: &ChunkedArray<T>, multithreaded: bool, sorted: bool) -> GroupsProxy
where
    T: PolarsIntegerType,
    T::Native: Hash + Eq + Send + AsU64,
    Option<T::Native>: AsU64,
{
    // set group size hint
    #[cfg(feature = "dtype-categorical")]
    let group_size_hint = if let DataType::Categorical(Some(m)) = ca.dtype() {
        ca.len() / m.len()
    } else {
        0
    };
    #[cfg(not(feature = "dtype-categorical"))]
    let group_size_hint = 0;

    if multithreaded && group_multithreaded(ca) {
        let n_partitions = _set_partition_size() as u64;

        // use the arrays as iterators
        if ca.chunks.len() == 1 {
            if !ca.has_validity() {
                let keys = vec![ca.cont_slice().unwrap()];
                groupby_threaded_num(keys, group_size_hint, n_partitions, sorted)
            } else {
                let keys = ca
                    .downcast_iter()
                    .map(|arr| arr.into_iter().map(|x| x.copied()).collect::<Vec<_>>())
                    .collect::<Vec<_>>();
                groupby_threaded_num(keys, group_size_hint, n_partitions, sorted)
            }
            // use the polars-iterators
        } else if !ca.has_validity() {
            let keys = vec![ca.into_no_null_iter().collect::<Vec<_>>()];
            groupby_threaded_num(keys, group_size_hint, n_partitions, sorted)
        } else {
            let keys = vec![ca.into_iter().collect::<Vec<_>>()];
            groupby_threaded_num(keys, group_size_hint, n_partitions, sorted)
        }
    } else if !ca.has_validity() {
        groupby(ca.into_no_null_iter(), sorted)
    } else {
        groupby(ca.into_iter(), sorted)
    }
}

impl<T> ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: NumCast,
{
    fn create_groups_from_sorted(&self, multithreaded: bool) -> GroupsSlice {
        if verbose() {
            eprintln!("groupby keys are sorted; running sorted key fast path");
        }
        let arr = self.downcast_iter().next().unwrap();
        if arr.is_empty() {
            return GroupsSlice::default();
        }
        let mut values = arr.values().as_slice();
        let null_count = arr.null_count();
        let length = values.len();

        // all nulls
        if null_count == length {
            return vec![[0, length as IdxSize]];
        }

        let mut nulls_first = false;
        if null_count > 0 {
            nulls_first = arr.get(0).is_none()
        }

        if nulls_first {
            values = &values[null_count..];
        } else {
            values = &values[..length - null_count];
        };

        let n_threads = POOL.current_num_threads();
        let groups = if multithreaded && n_threads > 1 {
            let parts = create_clean_partitions(values, n_threads, self.is_sorted_reverse_flag());
            let n_parts = parts.len();

            let first_ptr = &values[0] as *const T::Native as usize;
            let groups = POOL
                .install(|| {
                    parts.par_iter().enumerate().map(|(i, part)| {
                        // we go via usize as *const is not send
                        let first_ptr = first_ptr as *const T::Native;

                        let part_first_ptr = &part[0] as *const T::Native;
                        let mut offset =
                            unsafe { part_first_ptr.offset_from(first_ptr) } as IdxSize;

                        // nulls first: only add the nulls at the first partition
                        if nulls_first && i == 0 {
                            partition_to_groups(part, null_count as IdxSize, true, offset)
                        }
                        // nulls last: only compute at the last partition
                        else if !nulls_first && i == n_parts - 1 {
                            partition_to_groups(part, null_count as IdxSize, false, offset)
                        }
                        // other partitions
                        else {
                            if nulls_first {
                                offset += null_count as IdxSize;
                            };

                            partition_to_groups(part, 0, false, offset)
                        }
                    })
                })
                .collect::<Vec<_>>();
            flatten(&groups, None)
        } else {
            partition_to_groups(values, null_count as IdxSize, nulls_first, 0)
        };
        groups
    }
}

impl<T> IntoGroupsProxy for ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: NumCast,
{
    fn group_tuples(&self, multithreaded: bool, sorted: bool) -> PolarsResult<GroupsProxy> {
        // sorted path
        if self.is_sorted_flag() || self.is_sorted_reverse_flag() && self.chunks().len() == 1 {
            // don't have to pass `sorted` arg, GroupSlice is always sorted.
            return Ok(GroupsProxy::Slice {
                groups: self.create_groups_from_sorted(multithreaded),
                rolling: false,
            });
        }

        let out = match self.dtype() {
            DataType::UInt64 => {
                // convince the compiler that we are this type.
                let ca: &UInt64Chunked = unsafe {
                    &*(self as *const ChunkedArray<T> as *const ChunkedArray<UInt64Type>)
                };
                num_groups_proxy(ca, multithreaded, sorted)
            }
            DataType::UInt32 => {
                // convince the compiler that we are this type.
                let ca: &UInt32Chunked = unsafe {
                    &*(self as *const ChunkedArray<T> as *const ChunkedArray<UInt32Type>)
                };
                num_groups_proxy(ca, multithreaded, sorted)
            }
            DataType::Int64 | DataType::Float64 => {
                let ca = self.bit_repr_large();
                num_groups_proxy(&ca, multithreaded, sorted)
            }
            DataType::Int32 | DataType::Float32 => {
                let ca = self.bit_repr_small();
                num_groups_proxy(&ca, multithreaded, sorted)
            }
            #[cfg(feature = "performant")]
            DataType::Int8 => {
                // convince the compiler that we are this type.
                let ca: &Int8Chunked =
                    unsafe { &*(self as *const ChunkedArray<T> as *const ChunkedArray<Int8Type>) };
                let ca = ca.reinterpret_unsigned();
                num_groups_proxy(&ca, multithreaded, sorted)
            }
            #[cfg(feature = "performant")]
            DataType::UInt8 => {
                // convince the compiler that we are this type.
                let ca: &UInt8Chunked =
                    unsafe { &*(self as *const ChunkedArray<T> as *const ChunkedArray<UInt8Type>) };
                num_groups_proxy(ca, multithreaded, sorted)
            }
            #[cfg(feature = "performant")]
            DataType::Int16 => {
                // convince the compiler that we are this type.
                let ca: &Int16Chunked =
                    unsafe { &*(self as *const ChunkedArray<T> as *const ChunkedArray<Int16Type>) };
                let ca = ca.reinterpret_unsigned();
                num_groups_proxy(&ca, multithreaded, sorted)
            }
            #[cfg(feature = "performant")]
            DataType::UInt16 => {
                // convince the compiler that we are this type.
                let ca: &UInt16Chunked = unsafe {
                    &*(self as *const ChunkedArray<T> as *const ChunkedArray<UInt16Type>)
                };
                num_groups_proxy(ca, multithreaded, sorted)
            }
            _ => {
                let ca = self.cast_unchecked(&DataType::UInt32).unwrap();
                let ca = ca.u32().unwrap();
                num_groups_proxy(ca, multithreaded, sorted)
            }
        };
        Ok(out)
    }
}
impl IntoGroupsProxy for BooleanChunked {
    fn group_tuples(&self, multithreaded: bool, sorted: bool) -> PolarsResult<GroupsProxy> {
        #[cfg(feature = "performant")]
        {
            let ca = self.cast(&DataType::UInt8).unwrap();
            let ca = ca.u8().unwrap();
            ca.group_tuples(multithreaded, sorted)
        }
        #[cfg(not(feature = "performant"))]
        {
            let ca = self.cast(&DataType::UInt32).unwrap();
            let ca = ca.u32().unwrap();
            ca.group_tuples(multithreaded, sorted)
        }
    }
}

impl IntoGroupsProxy for Utf8Chunked {
    #[allow(clippy::needless_lifetimes)]
    fn group_tuples<'a>(&'a self, multithreaded: bool, sorted: bool) -> PolarsResult<GroupsProxy> {
        let hb = RandomState::default();
        let null_h = get_null_hash_value(hb.clone());

        let out = if multithreaded {
            let n_partitions = _set_partition_size();

            let split = _split_offsets(self.len(), n_partitions);

            let str_hashes = POOL.install(|| {
                split
                    .into_par_iter()
                    .map(|(offset, len)| {
                        let ca = self.slice(offset as i64, len);
                        ca.into_iter()
                            .map(|opt_s| {
                                let hash = match opt_s {
                                    Some(s) => hb.hash_single(s),
                                    None => null_h,
                                };
                                // Safety:
                                // the underlying data is tied to self
                                unsafe {
                                    std::mem::transmute::<BytesHash<'_>, BytesHash<'a>>(
                                        BytesHash::new_from_str(opt_s, hash),
                                    )
                                }
                            })
                            .collect_trusted::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            });
            groupby_threaded_num(str_hashes, 0, n_partitions as u64, sorted)
        } else {
            let str_hashes = self
                .into_iter()
                .map(|opt_s| {
                    let hash = match opt_s {
                        Some(s) => hb.hash_single(s),
                        None => null_h,
                    };
                    BytesHash::new_from_str(opt_s, hash)
                })
                .collect_trusted::<Vec<_>>();
            groupby(str_hashes.iter(), sorted)
        };
        Ok(out)
    }
}

#[cfg(feature = "dtype-binary")]
impl IntoGroupsProxy for BinaryChunked {
    #[allow(clippy::needless_lifetimes)]
    fn group_tuples<'a>(&'a self, multithreaded: bool, sorted: bool) -> PolarsResult<GroupsProxy> {
        let hb = RandomState::default();
        let null_h = get_null_hash_value(hb.clone());

        let out = if multithreaded {
            let n_partitions = _set_partition_size();

            let split = _split_offsets(self.len(), n_partitions);

            let byte_hashes = POOL.install(|| {
                split
                    .into_par_iter()
                    .map(|(offset, len)| {
                        let ca = self.slice(offset as i64, len);
                        ca.into_iter()
                            .map(|opt_b| {
                                let hash = match opt_b {
                                    Some(s) => hb.hash_single(s),
                                    None => null_h,
                                };
                                // Safety:
                                // the underlying data is tied to self
                                unsafe {
                                    std::mem::transmute::<BytesHash<'_>, BytesHash<'a>>(
                                        BytesHash::new(opt_b, hash),
                                    )
                                }
                            })
                            .collect_trusted::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            });
            groupby_threaded_num(byte_hashes, 0, n_partitions as u64, sorted)
        } else {
            let byte_hashes = self
                .into_iter()
                .map(|opt_b| {
                    let hash = match opt_b {
                        Some(s) => hb.hash_single(s),
                        None => null_h,
                    };
                    BytesHash::new(opt_b, hash)
                })
                .collect_trusted::<Vec<_>>();
            groupby(byte_hashes.iter(), sorted)
        };
        Ok(out)
    }
}

impl IntoGroupsProxy for ListChunked {
    #[allow(clippy::needless_lifetimes)]
    #[allow(unused_variables)]
    fn group_tuples<'a>(&'a self, multithreaded: bool, sorted: bool) -> PolarsResult<GroupsProxy> {
        #[cfg(feature = "groupby_list")]
        {
            if !self.inner_dtype().to_physical().is_numeric() {
                return Err(PolarsError::ComputeError(
                    "Grouping on List type is only allowed if the inner type is numeric".into(),
                ));
            }

            let hb = RandomState::default();
            let null_h = get_null_hash_value(hb.clone());

            let arr_to_hashes = |ca: &ListChunked| {
                let mut out = Vec::with_capacity(ca.len());

                for arr in ca.downcast_iter() {
                    out.extend(numeric_list_bytes_iter(arr)?.map(|opt_bytes| {
                        let hash = match opt_bytes {
                            Some(s) => hb.hash_single(s),
                            None => null_h,
                        };

                        // Safety:
                        // the underlying data is tied to self
                        unsafe {
                            std::mem::transmute::<BytesHash<'_>, BytesHash<'a>>(BytesHash::new(
                                opt_bytes, hash,
                            ))
                        }
                    }))
                }
                Ok(out)
            };

            if multithreaded {
                let n_partitions = _set_partition_size();
                let split = _split_offsets(self.len(), n_partitions);

                let groups: PolarsResult<_> = POOL.install(|| {
                    let bytes_hashes = split
                        .into_par_iter()
                        .map(|(offset, len)| {
                            let ca = self.slice(offset as i64, len);
                            arr_to_hashes(&ca)
                        })
                        .collect::<PolarsResult<Vec<_>>>()?;
                    Ok(groupby_threaded_num(
                        bytes_hashes,
                        0,
                        n_partitions as u64,
                        sorted,
                    ))
                });
                groups
            } else {
                let hashes = arr_to_hashes(self)?;
                Ok(groupby(hashes.iter(), sorted))
            }
        }
        #[cfg(not(feature = "groupby_list"))]
        {
            panic!("activate 'groupby_list' feature")
        }
    }
}

#[cfg(feature = "object")]
impl<T> IntoGroupsProxy for ObjectChunked<T>
where
    T: PolarsObject,
{
    fn group_tuples(&self, _multithreaded: bool, sorted: bool) -> PolarsResult<GroupsProxy> {
        Ok(groupby(self.into_iter(), sorted))
    }
}

/// Used to tightly two 32 bit values and null information
/// Only the bit values matter, not the meaning of the bits
#[inline]
pub(super) fn pack_u32_tuples(opt_l: Option<u32>, opt_r: Option<u32>) -> [u8; 9] {
    // 4 bytes for first value
    // 4 bytes for second value
    // last bytes' bits are used to indicate missing values
    let mut val = [0u8; 9];
    let s = &mut val;
    match (opt_l, opt_r) {
        (Some(l), Some(r)) => {
            // write to first 4 places
            unsafe { copy_from_slice_unchecked(&l.to_ne_bytes(), &mut s[..4]) }
            // write to second chunk of 4 places
            unsafe { copy_from_slice_unchecked(&r.to_ne_bytes(), &mut s[4..8]) }
            // leave last byte as is
        }
        (Some(l), None) => {
            unsafe { copy_from_slice_unchecked(&l.to_ne_bytes(), &mut s[..4]) }
            // set right null bit
            s[8] = 1;
        }
        (None, Some(r)) => {
            unsafe { copy_from_slice_unchecked(&r.to_ne_bytes(), &mut s[4..8]) }
            // set left null bit
            s[8] = 1 << 1;
        }
        (None, None) => {
            // set two null bits
            s[8] = 3;
        }
    }
    val
}

/// Used to tightly two 64 bit values and null information
/// Only the bit values matter, not the meaning of the bits
#[inline]
pub(super) fn pack_u64_tuples(opt_l: Option<u64>, opt_r: Option<u64>) -> [u8; 17] {
    // 8 bytes for first value
    // 8 bytes for second value
    // last bytes' bits are used to indicate missing values
    let mut val = [0u8; 17];
    let s = &mut val;
    match (opt_l, opt_r) {
        (Some(l), Some(r)) => {
            // write to first 4 places
            unsafe { copy_from_slice_unchecked(&l.to_ne_bytes(), &mut s[..8]) }
            // write to second chunk of 4 places
            unsafe { copy_from_slice_unchecked(&r.to_ne_bytes(), &mut s[8..16]) }
            // leave last byte as is
        }
        (Some(l), None) => {
            unsafe { copy_from_slice_unchecked(&l.to_ne_bytes(), &mut s[..8]) }
            // set right null bit
            s[16] = 1;
        }
        (None, Some(r)) => {
            unsafe { copy_from_slice_unchecked(&r.to_ne_bytes(), &mut s[8..16]) }
            // set left null bit
            s[16] = 1 << 1;
        }
        (None, None) => {
            // set two null bits
            s[16] = 3;
        }
    }
    val
}

/// Used to tightly one 32 bit and a 64 bit valued type and null information
/// Only the bit values matter, not the meaning of the bits
#[inline]
pub(super) fn pack_u32_u64_tuples(opt_l: Option<u32>, opt_r: Option<u64>) -> [u8; 13] {
    // 8 bytes for first value
    // 8 bytes for second value
    // last bytes' bits are used to indicate missing values
    let mut val = [0u8; 13];
    let s = &mut val;
    match (opt_l, opt_r) {
        (Some(l), Some(r)) => {
            // write to first 4 places
            unsafe { copy_from_slice_unchecked(&l.to_ne_bytes(), &mut s[..4]) }
            // write to second chunk of 4 places
            unsafe { copy_from_slice_unchecked(&r.to_ne_bytes(), &mut s[4..12]) }
            // leave last byte as is
        }
        (Some(l), None) => {
            unsafe { copy_from_slice_unchecked(&l.to_ne_bytes(), &mut s[..4]) }
            // set right null bit
            s[12] = 1;
        }
        (None, Some(r)) => {
            unsafe { copy_from_slice_unchecked(&r.to_ne_bytes(), &mut s[4..12]) }
            // set left null bit
            s[12] = 1 << 1;
        }
        (None, None) => {
            // set two null bits
            s[12] = 3;
        }
    }
    val
}

/// We will pack the utf8 columns into single column. Nulls are encoded in the start of the string
/// by either:
/// 11 => both valid
/// 00 => both null
/// 10 => first valid
/// 01 => second valid
pub(super) fn pack_utf8_columns(
    lhs: &Utf8Chunked,
    rhs: &Utf8Chunked,
    n_partitions: usize,
    sorted: bool,
) -> GroupsProxy {
    let splits = _split_offsets(lhs.len(), n_partitions);
    let hb = RandomState::default();
    let null_h = get_null_hash_value(hb.clone());

    let (hashes, _backing_bytes): (Vec<_>, Vec<_>) = splits
        .into_par_iter()
        .map(|(offset, len)| {
            let lhs = lhs.slice(offset as i64, len);
            let rhs = rhs.slice(offset as i64, len);

            // the additional:
            // 2 is needed for the validity
            // 1 for the '_' delimiter
            let size = lhs.get_values_size() + rhs.get_values_size() + lhs.len() * 3 + 1;

            let mut values = Vec::with_capacity(size);
            let ptr = values.as_ptr() as usize;
            let mut str_hashes = Vec::with_capacity(lhs.len());

            lhs.into_iter().zip(rhs.into_iter()).for_each(|(lhs, rhs)| {
                match (lhs, rhs) {
                    (Some(lhs), Some(rhs)) => {
                        let start = values.len();
                        values.extend_from_slice("11".as_bytes());
                        values.extend_from_slice(lhs.as_bytes());
                        values.push(b'_');
                        values.extend_from_slice(rhs.as_bytes());
                        // reallocated lifetime is invalid
                        debug_assert_eq!(ptr, values.as_ptr() as usize);
                        let end = values.len();
                        // Safety:
                        // - we know the bytes are valid utf8
                        // - we are in bounds
                        // - the lifetime as long as `values` not is dropped
                        //   so `str_val` may never leave this function
                        let str_val: &'static str = unsafe {
                            std::mem::transmute(std::str::from_utf8_unchecked(
                                values.get_unchecked(start..end),
                            ))
                        };
                        let hash = hb.hash_single(str_val);
                        str_hashes.push(BytesHash::new(Some(str_val.as_bytes()), hash))
                    }
                    (None, Some(rhs)) => {
                        let start = values.len();
                        values.extend_from_slice("01".as_bytes());
                        values.push(b'_');
                        values.extend_from_slice(rhs.as_bytes());
                        debug_assert_eq!(ptr, values.as_ptr() as usize);
                        let end = values.len();
                        let str_val: &'static str = unsafe {
                            std::mem::transmute(std::str::from_utf8_unchecked(
                                values.get_unchecked(start..end),
                            ))
                        };
                        let hash = hb.hash_single(str_val);
                        str_hashes.push(BytesHash::new(Some(str_val.as_bytes()), hash))
                    }
                    (Some(lhs), None) => {
                        let start = values.len();
                        values.extend_from_slice("10".as_bytes());
                        values.extend_from_slice(lhs.as_bytes());
                        values.push(b'_');
                        debug_assert_eq!(ptr, values.as_ptr() as usize);
                        let end = values.len();
                        let str_val: &'static str = unsafe {
                            std::mem::transmute(std::str::from_utf8_unchecked(
                                values.get_unchecked(start..end),
                            ))
                        };
                        let hash = hb.hash_single(str_val);
                        str_hashes.push(BytesHash::new(Some(str_val.as_bytes()), hash))
                    }
                    (None, None) => str_hashes.push(BytesHash::new(None, null_h)),
                }
            });
            (str_hashes, values)
        })
        .unzip();
    groupby_threaded_num(hashes, 0, n_partitions as u64, sorted)
}
