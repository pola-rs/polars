use super::*;
use crate::utils::split_offsets;
use polars_arrow::prelude::*;

/// Used to create the tuples for a groupby operation.
pub trait IntoGroupsProxy {
    /// Create the tuples need for a groupby operation.
    ///     * The first value in the tuple is the first index of the group.
    ///     * The second value in the tuple is are the indexes of the groups including the first value.
    fn group_tuples(&self, _multithreaded: bool, _sorted: bool) -> GroupsProxy {
        unimplemented!()
    }
}

fn group_multithreaded<T>(ca: &ChunkedArray<T>) -> bool {
    // TODO! change to something sensible
    ca.len() > 1000
}

fn num_groups_proxy<T>(ca: &ChunkedArray<T>, multithreaded: bool, sorted: bool) -> GroupsProxy
where
    T: PolarsIntegerType,
    T::Native: Hash + Eq + Send + AsU64,
    Option<T::Native>: AsU64,
{
    #[cfg(feature = "dtype-categorical")]
    let group_size_hint = if let Some(m) = &ca.categorical_map {
        ca.len() / m.len()
    } else {
        0
    };
    #[cfg(not(feature = "dtype-categorical"))]
    let group_size_hint = 0;
    if multithreaded && group_multithreaded(ca) {
        let n_partitions = set_partition_size() as u64;

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

impl<T> IntoGroupsProxy for ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: NumCast,
{
    fn group_tuples(&self, multithreaded: bool, sorted: bool) -> GroupsProxy {
        match self.dtype() {
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
            _ => {
                let ca = self.cast(&DataType::UInt32).unwrap();
                let ca = ca.u32().unwrap();
                num_groups_proxy(ca, multithreaded, sorted)
            }
        }
    }
}
impl IntoGroupsProxy for BooleanChunked {
    fn group_tuples(&self, multithreaded: bool, sorted: bool) -> GroupsProxy {
        let ca = self.cast(&DataType::UInt32).unwrap();
        let ca = ca.u32().unwrap();
        ca.group_tuples(multithreaded, sorted)
    }
}

impl IntoGroupsProxy for Utf8Chunked {
    #[allow(clippy::needless_lifetimes)]
    fn group_tuples<'a>(&'a self, multithreaded: bool, sorted: bool) -> GroupsProxy {
        let hb = RandomState::default();
        let null_h = get_null_hash_value(hb.clone());

        if multithreaded {
            let n_partitions = set_partition_size();

            let split = split_offsets(self.len(), n_partitions);

            let str_hashes = POOL.install(|| {
                split
                    .into_par_iter()
                    .map(|(offset, len)| {
                        let ca = self.slice(offset as i64, len);
                        ca.into_iter()
                            .map(|opt_s| {
                                let hash = match opt_s {
                                    Some(s) => str::get_hash(s, &hb),
                                    None => null_h,
                                };
                                // Safety:
                                // the underlying data is tied to self
                                unsafe {
                                    std::mem::transmute::<StrHash<'_>, StrHash<'a>>(StrHash::new(
                                        opt_s, hash,
                                    ))
                                }
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            });
            groupby_threaded_num(str_hashes, 0, n_partitions as u64, sorted)
        } else {
            let str_hashes = self
                .into_iter()
                .map(|opt_s| {
                    let hash = match opt_s {
                        Some(s) => str::get_hash(s, &hb),
                        None => null_h,
                    };
                    StrHash::new(opt_s, hash)
                })
                .collect::<Vec<_>>();
            groupby(str_hashes.iter(), sorted)
        }
    }
}

impl IntoGroupsProxy for ListChunked {
    #[cfg(feature = "groupby_list")]
    fn group_tuples(&self, _multithreaded: bool, sorted: bool) -> GroupsProxy {
        groupby(self.into_iter().map(|opt_s| opt_s.map(Wrap)), sorted)
    }
}

#[cfg(feature = "object")]
impl<T> IntoGroupsProxy for ObjectChunked<T>
where
    T: PolarsObject,
{
    fn group_tuples(&self, _multithreaded: bool, sorted: bool) -> GroupsProxy {
        groupby(self.into_iter(), sorted)
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
    let splits = split_offsets(lhs.len(), n_partitions);
    let hb = RandomState::default();
    let null_h = get_null_hash_value(hb.clone());

    let (hashes, _backing_bytes): (Vec<_>, Vec<_>) = splits
        .into_par_iter()
        .map(|(offset, len)| {
            let lhs = lhs.slice(offset as i64, len);
            let rhs = rhs.slice(offset as i64, len);

            // the additional 2 is needed for the validity
            let size = lhs.get_values_size() + rhs.get_values_size() + lhs.len() * 2 + 1;

            let mut values = Vec::with_capacity(size);
            let ptr = values.as_ptr() as usize;
            let mut str_hashes = Vec::with_capacity(lhs.len());

            lhs.into_iter().zip(rhs.into_iter()).for_each(|(lhs, rhs)| {
                match (lhs, rhs) {
                    (Some(lhs), Some(rhs)) => {
                        let start = values.len();
                        values.extend_from_slice("11".as_bytes());
                        values.extend_from_slice(lhs.as_bytes());
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
                        let hash = str::get_hash(str_val, &hb);
                        str_hashes.push(StrHash::new(Some(str_val), hash))
                    }
                    (None, Some(rhs)) => {
                        let start = values.len();
                        values.extend_from_slice("01".as_bytes());
                        values.extend_from_slice(rhs.as_bytes());
                        debug_assert_eq!(ptr, values.as_ptr() as usize);
                        let end = values.len();
                        let str_val: &'static str = unsafe {
                            std::mem::transmute(std::str::from_utf8_unchecked(
                                values.get_unchecked(start..end),
                            ))
                        };
                        let hash = str::get_hash(str_val, &hb);
                        str_hashes.push(StrHash::new(Some(str_val), hash))
                    }
                    (Some(lhs), None) => {
                        let start = values.len();
                        values.extend_from_slice("10".as_bytes());
                        values.extend_from_slice(lhs.as_bytes());
                        debug_assert_eq!(ptr, values.as_ptr() as usize);
                        let end = values.len();
                        let str_val: &'static str = unsafe {
                            std::mem::transmute(std::str::from_utf8_unchecked(
                                values.get_unchecked(start..end),
                            ))
                        };
                        let hash = str::get_hash(str_val, &hb);
                        str_hashes.push(StrHash::new(Some(str_val), hash))
                    }
                    (None, None) => str_hashes.push(StrHash::new(None, null_h)),
                }
            });
            (str_hashes, values)
        })
        .unzip();
    groupby_threaded_num(hashes, 0, n_partitions as u64, sorted)
}
