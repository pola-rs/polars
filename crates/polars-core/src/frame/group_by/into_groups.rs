#[cfg(feature = "group_by_list")]
use arrow::legacy::kernels::list_bytes_iter::numeric_list_bytes_iter;
use arrow::legacy::kernels::sort_partition::{create_clean_partitions, partition_to_groups};
use arrow::legacy::prelude::*;

use super::*;
use crate::config::verbose;
use crate::utils::_split_offsets;
use crate::utils::flatten::flatten_par;

/// Used to create the tuples for a group_by operation.
pub trait IntoGroupsProxy {
    /// Create the tuples need for a group_by operation.
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
    if multithreaded && group_multithreaded(ca) {
        let n_partitions = _set_partition_size() as u64;

        // use the arrays as iterators
        if ca.null_count() == 0 {
            let keys = ca
                .downcast_iter()
                .map(|arr| arr.values().as_slice())
                .collect::<Vec<_>>();
            group_by_threaded_slice(keys, n_partitions, sorted)
        } else {
            let keys = ca.downcast_iter().collect::<Vec<_>>();
            group_by_threaded_iter(&keys, n_partitions, sorted)
        }
    } else if !ca.has_validity() {
        group_by(ca.into_no_null_iter(), sorted)
    } else {
        group_by(ca.into_iter(), sorted)
    }
}

impl<T> ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: NumCast,
{
    fn create_groups_from_sorted(&self, multithreaded: bool) -> GroupsSlice {
        if verbose() {
            eprintln!("group_by keys are sorted; running sorted key fast path");
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
            let parts =
                create_clean_partitions(values, n_threads, self.is_sorted_descending_flag());
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
            flatten_par(&groups)
        } else {
            partition_to_groups(values, null_count as IdxSize, nulls_first, 0)
        };
        groups
    }
}

#[cfg(all(feature = "dtype-categorical", feature = "performant"))]
impl IntoGroupsProxy for CategoricalChunked {
    fn group_tuples(&self, multithreaded: bool, sorted: bool) -> PolarsResult<GroupsProxy> {
        Ok(self.group_tuples_perfect(multithreaded, sorted))
    }
}

impl<T> IntoGroupsProxy for ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: NumCast,
{
    fn group_tuples(&self, multithreaded: bool, sorted: bool) -> PolarsResult<GroupsProxy> {
        // sorted path
        if self.is_sorted_ascending_flag() || self.is_sorted_descending_flag() {
            // don't have to pass `sorted` arg, GroupSlice is always sorted.
            return Ok(GroupsProxy::Slice {
                groups: self.rechunk().create_groups_from_sorted(multithreaded),
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
            },
            DataType::UInt32 => {
                // convince the compiler that we are this type.
                let ca: &UInt32Chunked = unsafe {
                    &*(self as *const ChunkedArray<T> as *const ChunkedArray<UInt32Type>)
                };
                num_groups_proxy(ca, multithreaded, sorted)
            },
            DataType::Int64 | DataType::Float64 => {
                let ca = self.bit_repr_large();
                num_groups_proxy(&ca, multithreaded, sorted)
            },
            DataType::Int32 | DataType::Float32 => {
                let ca = self.bit_repr_small();
                num_groups_proxy(&ca, multithreaded, sorted)
            },
            #[cfg(all(feature = "performant", feature = "dtype-i8", feature = "dtype-u8"))]
            DataType::Int8 => {
                // convince the compiler that we are this type.
                let ca: &Int8Chunked =
                    unsafe { &*(self as *const ChunkedArray<T> as *const ChunkedArray<Int8Type>) };
                let s = ca.reinterpret_unsigned();
                return s.group_tuples(multithreaded, sorted);
            },
            #[cfg(all(feature = "performant", feature = "dtype-i8", feature = "dtype-u8"))]
            DataType::UInt8 => {
                // convince the compiler that we are this type.
                let ca: &UInt8Chunked =
                    unsafe { &*(self as *const ChunkedArray<T> as *const ChunkedArray<UInt8Type>) };
                num_groups_proxy(ca, multithreaded, sorted)
            },
            #[cfg(all(feature = "performant", feature = "dtype-i16", feature = "dtype-u16"))]
            DataType::Int16 => {
                // convince the compiler that we are this type.
                let ca: &Int16Chunked =
                    unsafe { &*(self as *const ChunkedArray<T> as *const ChunkedArray<Int16Type>) };
                let s = ca.reinterpret_unsigned();
                return s.group_tuples(multithreaded, sorted);
            },
            #[cfg(all(feature = "performant", feature = "dtype-i16", feature = "dtype-u16"))]
            DataType::UInt16 => {
                // convince the compiler that we are this type.
                let ca: &UInt16Chunked = unsafe {
                    &*(self as *const ChunkedArray<T> as *const ChunkedArray<UInt16Type>)
                };
                num_groups_proxy(ca, multithreaded, sorted)
            },
            _ => {
                let ca = unsafe { self.cast_unchecked(&DataType::UInt32).unwrap() };
                let ca = ca.u32().unwrap();
                num_groups_proxy(ca, multithreaded, sorted)
            },
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
        self.as_binary().group_tuples(multithreaded, sorted)
    }
}

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
                                    Some(s) => hb.hash_one(s),
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
            let byte_hashes = byte_hashes.iter().collect::<Vec<_>>();
            group_by_threaded_slice(byte_hashes, n_partitions as u64, sorted)
        } else {
            let byte_hashes = self
                .into_iter()
                .map(|opt_b| {
                    let hash = match opt_b {
                        Some(s) => hb.hash_one(s),
                        None => null_h,
                    };
                    BytesHash::new(opt_b, hash)
                })
                .collect_trusted::<Vec<_>>();
            group_by(byte_hashes.iter(), sorted)
        };
        Ok(out)
    }
}

impl IntoGroupsProxy for ListChunked {
    #[allow(clippy::needless_lifetimes)]
    #[allow(unused_variables)]
    fn group_tuples<'a>(&'a self, multithreaded: bool, sorted: bool) -> PolarsResult<GroupsProxy> {
        #[cfg(feature = "group_by_list")]
        {
            polars_ensure!(
                self.inner_dtype().to_physical().is_numeric(),
                ComputeError: "grouping on list type is only allowed if the inner type is numeric"
            );

            let hb = RandomState::default();
            let null_h = get_null_hash_value(hb.clone());

            let arr_to_hashes = |ca: &ListChunked| {
                let mut out = Vec::with_capacity(ca.len());

                for arr in ca.downcast_iter() {
                    out.extend(numeric_list_bytes_iter(arr)?.map(|opt_bytes| {
                        let hash = match opt_bytes {
                            Some(s) => hb.hash_one(s),
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
                    let bytes_hashes = bytes_hashes.iter().collect::<Vec<_>>();
                    Ok(group_by_threaded_slice(
                        bytes_hashes,
                        n_partitions as u64,
                        sorted,
                    ))
                });
                groups
            } else {
                let hashes = arr_to_hashes(self)?;
                Ok(group_by(hashes.iter(), sorted))
            }
        }
        #[cfg(not(feature = "group_by_list"))]
        {
            panic!("activate 'group_by_list' feature")
        }
    }
}

#[cfg(feature = "dtype-array")]
impl IntoGroupsProxy for ArrayChunked {
    #[allow(clippy::needless_lifetimes)]
    #[allow(unused_variables)]
    fn group_tuples<'a>(
        &'a self,
        _multithreaded: bool,
        _sorted: bool,
    ) -> PolarsResult<GroupsProxy> {
        todo!("grouping FixedSizeList not yet supported")
    }
}

#[cfg(feature = "object")]
impl<T> IntoGroupsProxy for ObjectChunked<T>
where
    T: PolarsObject,
{
    fn group_tuples(&self, _multithreaded: bool, sorted: bool) -> PolarsResult<GroupsProxy> {
        Ok(group_by(self.into_iter(), sorted))
    }
}
