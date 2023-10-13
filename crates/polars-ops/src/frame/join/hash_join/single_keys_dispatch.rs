use num_traits::NumCast;

use super::*;
use crate::series::SeriesSealed;

pub trait SeriesJoin: SeriesSealed + Sized {
    #[doc(hidden)]
    fn hash_join_left(
        &self,
        other: &Series,
        validate: JoinValidation,
    ) -> PolarsResult<LeftJoinIds> {
        let s_self = self.as_series();
        let (lhs, rhs) = (s_self.to_physical_repr(), other.to_physical_repr());
        validate.validate_probe(&lhs, &rhs, false)?;

        use DataType::*;
        match lhs.dtype() {
            Utf8 => {
                let lhs = lhs.cast(&Binary).unwrap();
                let rhs = rhs.cast(&Binary).unwrap();
                lhs.hash_join_left(&rhs, JoinValidation::ManyToMany)
            },
            Binary => {
                let lhs = lhs.binary().unwrap();
                let rhs = rhs.binary().unwrap();
                let (lhs, rhs, _, _) = prepare_binary(lhs, rhs, false);
                let lhs = lhs.iter().map(|v| v.as_slice()).collect::<Vec<_>>();
                let rhs = rhs.iter().map(|v| v.as_slice()).collect::<Vec<_>>();
                hash_join_tuples_left(lhs, rhs, None, None, validate)
            },
            _ => {
                if s_self.bit_repr_is_large() {
                    let lhs = lhs.bit_repr_large();
                    let rhs = rhs.bit_repr_large();
                    num_group_join_left(&lhs, &rhs, validate)
                } else {
                    let lhs = lhs.bit_repr_small();
                    let rhs = rhs.bit_repr_small();
                    num_group_join_left(&lhs, &rhs, validate)
                }
            },
        }
    }

    #[cfg(feature = "semi_anti_join")]
    fn hash_join_semi_anti(&self, other: &Series, anti: bool) -> Vec<IdxSize> {
        let s_self = self.as_series();
        let (lhs, rhs) = (s_self.to_physical_repr(), other.to_physical_repr());

        use DataType::*;
        match lhs.dtype() {
            Utf8 => {
                let lhs = lhs.cast(&Binary).unwrap();
                let rhs = rhs.cast(&Binary).unwrap();
                lhs.hash_join_semi_anti(&rhs, anti)
            },
            Binary => {
                let lhs = lhs.binary().unwrap();
                let rhs = rhs.binary().unwrap();
                let (lhs, rhs, _, _) = prepare_binary(lhs, rhs, false);
                let lhs = lhs.iter().map(|v| v.as_slice()).collect::<Vec<_>>();
                let rhs = rhs.iter().map(|v| v.as_slice()).collect::<Vec<_>>();
                if anti {
                    hash_join_tuples_left_anti(lhs, rhs)
                } else {
                    hash_join_tuples_left_semi(lhs, rhs)
                }
            },
            _ => {
                if s_self.bit_repr_is_large() {
                    let lhs = lhs.bit_repr_large();
                    let rhs = rhs.bit_repr_large();
                    num_group_join_anti_semi(&lhs, &rhs, anti)
                } else {
                    let lhs = lhs.bit_repr_small();
                    let rhs = rhs.bit_repr_small();
                    num_group_join_anti_semi(&lhs, &rhs, anti)
                }
            },
        }
    }

    // returns the join tuples and whether or not the lhs tuples are sorted
    fn hash_join_inner(
        &self,
        other: &Series,
        validate: JoinValidation,
    ) -> PolarsResult<(InnerJoinIds, bool)> {
        let s_self = self.as_series();
        let (lhs, rhs) = (s_self.to_physical_repr(), other.to_physical_repr());
        validate.validate_probe(&lhs, &rhs, true)?;

        use DataType::*;
        match lhs.dtype() {
            Utf8 => {
                let lhs = lhs.cast(&Binary).unwrap();
                let rhs = rhs.cast(&Binary).unwrap();
                lhs.hash_join_inner(&rhs, JoinValidation::ManyToMany)
            },
            Binary => {
                let lhs = lhs.binary().unwrap();
                let rhs = rhs.binary().unwrap();
                let (lhs, rhs, swapped, _) = prepare_binary(lhs, rhs, true);
                let lhs = lhs.iter().map(|v| v.as_slice()).collect::<Vec<_>>();
                let rhs = rhs.iter().map(|v| v.as_slice()).collect::<Vec<_>>();
                Ok((
                    hash_join_tuples_inner(lhs, rhs, swapped, validate)?,
                    !swapped,
                ))
            },
            _ => {
                if s_self.bit_repr_is_large() {
                    let lhs = s_self.bit_repr_large();
                    let rhs = other.bit_repr_large();
                    group_join_inner::<UInt64Type>(&lhs, &rhs, validate)
                } else {
                    let lhs = s_self.bit_repr_small();
                    let rhs = other.bit_repr_small();
                    group_join_inner::<UInt32Type>(&lhs, &rhs, validate)
                }
            },
        }
    }

    fn hash_join_outer(
        &self,
        other: &Series,
        validate: JoinValidation,
    ) -> PolarsResult<Vec<(Option<IdxSize>, Option<IdxSize>)>> {
        let s_self = self.as_series();
        let (lhs, rhs) = (s_self.to_physical_repr(), other.to_physical_repr());
        validate.validate_probe(&lhs, &rhs, true)?;

        use DataType::*;
        match lhs.dtype() {
            Utf8 => {
                let lhs = lhs.cast(&Binary).unwrap();
                let rhs = rhs.cast(&Binary).unwrap();
                lhs.hash_join_outer(&rhs, JoinValidation::ManyToMany)
            },
            Binary => {
                let lhs = lhs.binary().unwrap();
                let rhs = rhs.binary().unwrap();
                let (lhs, rhs, swapped, _) = prepare_binary(lhs, rhs, true);
                let lhs = lhs.iter().collect::<Vec<_>>();
                let rhs = rhs.iter().collect::<Vec<_>>();
                hash_join_tuples_outer(lhs, rhs, swapped, validate)
            },
            _ => {
                if s_self.bit_repr_is_large() {
                    let lhs = s_self.bit_repr_large();
                    let rhs = other.bit_repr_large();
                    hash_join_outer(&lhs, &rhs, validate)
                } else {
                    let lhs = s_self.bit_repr_small();
                    let rhs = other.bit_repr_small();
                    hash_join_outer(&lhs, &rhs, validate)
                }
            },
        }
    }
}

impl SeriesJoin for Series {}

fn chunks_as_slices<T>(splitted: &[ChunkedArray<T>]) -> Vec<&[T::Native]>
where
    T: PolarsNumericType,
{
    splitted
        .iter()
        .flat_map(|ca| ca.downcast_iter().map(|arr| arr.values().as_slice()))
        .collect()
}

fn get_arrays<T: PolarsDataType>(cas: &[ChunkedArray<T>]) -> Vec<&T::Array> {
    cas.iter().flat_map(|arr| arr.downcast_iter()).collect()
}

fn group_join_inner<T>(
    left: &ChunkedArray<T>,
    right: &ChunkedArray<T>,
    validate: JoinValidation,
) -> PolarsResult<(InnerJoinIds, bool)>
where
    T: PolarsDataType,
    for<'a> &'a T::Array: IntoIterator<Item = Option<&'a T::Physical<'a>>>,
    for<'a> T::Physical<'a>: Hash + Eq + Send + AsU64 + Copy + Send + Sync,
{
    let n_threads = POOL.current_num_threads();
    let (a, b, swapped) = det_hash_prone_order!(left, right);
    let splitted_a = split_ca(a, n_threads).unwrap();
    let splitted_b = split_ca(b, n_threads).unwrap();
    let splitted_a = get_arrays(&splitted_a);
    let splitted_b = get_arrays(&splitted_b);

    match (left.null_count(), right.null_count()) {
        (0, 0) => {
            let first = &splitted_a[0];
            if first.as_slice().is_some() {
                let splitted_a = splitted_a
                    .iter()
                    .map(|arr| arr.as_slice().unwrap())
                    .collect::<Vec<_>>();
                let splitted_b = splitted_b
                    .iter()
                    .map(|arr| arr.as_slice().unwrap())
                    .collect::<Vec<_>>();
                Ok((
                    hash_join_tuples_inner(splitted_a, splitted_b, swapped, validate)?,
                    !swapped,
                ))
            } else {
                Ok((
                    hash_join_tuples_inner(splitted_a, splitted_b, swapped, validate)?,
                    !swapped,
                ))
            }
        },
        _ => Ok((
            hash_join_tuples_inner(splitted_a, splitted_b, swapped, validate)?,
            !swapped,
        )),
    }
}

#[cfg(feature = "chunked_ids")]
fn create_mappings(
    chunks_left: &[ArrayRef],
    chunks_right: &[ArrayRef],
    left_len: usize,
    right_len: usize,
) -> (Option<Vec<ChunkId>>, Option<Vec<ChunkId>>) {
    let mapping_left = || {
        if chunks_left.len() > 1 {
            Some(create_chunked_index_mapping(chunks_left, left_len))
        } else {
            None
        }
    };

    let mapping_right = || {
        if chunks_right.len() > 1 {
            Some(create_chunked_index_mapping(chunks_right, right_len))
        } else {
            None
        }
    };

    POOL.join(mapping_left, mapping_right)
}

#[cfg(not(feature = "chunked_ids"))]
fn create_mappings(
    _chunks_left: &[ArrayRef],
    _chunks_right: &[ArrayRef],
    _left_len: usize,
    _right_len: usize,
) -> (Option<Vec<ChunkId>>, Option<Vec<ChunkId>>) {
    (None, None)
}

fn num_group_join_left<T>(
    left: &ChunkedArray<T>,
    right: &ChunkedArray<T>,
    validate: JoinValidation,
) -> PolarsResult<LeftJoinIds>
where
    T: PolarsIntegerType,
    T::Native: Hash + Eq + Send + AsU64,
    Option<T::Native>: AsU64,
{
    let n_threads = POOL.current_num_threads();
    let splitted_a = split_ca(left, n_threads).unwrap();
    let splitted_b = split_ca(right, n_threads).unwrap();
    match (
        left.null_count(),
        right.null_count(),
        left.chunks().len(),
        right.chunks().len(),
    ) {
        (0, 0, 1, 1) => {
            let keys_a = chunks_as_slices(&splitted_a);
            let keys_b = chunks_as_slices(&splitted_b);
            hash_join_tuples_left(keys_a, keys_b, None, None, validate)
        },
        (0, 0, _, _) => {
            let keys_a = chunks_as_slices(&splitted_a);
            let keys_b = chunks_as_slices(&splitted_b);

            let (mapping_left, mapping_right) =
                create_mappings(left.chunks(), right.chunks(), left.len(), right.len());
            hash_join_tuples_left(
                keys_a,
                keys_b,
                mapping_left.as_deref(),
                mapping_right.as_deref(),
                validate,
            )
        },
        _ => {
            let keys_a = get_arrays(&splitted_a);
            let keys_b = get_arrays(&splitted_b);
            let (mapping_left, mapping_right) =
                create_mappings(left.chunks(), right.chunks(), left.len(), right.len());
            hash_join_tuples_left(
                keys_a,
                keys_b,
                mapping_left.as_deref(),
                mapping_right.as_deref(),
                validate,
            )
        },
    }
}

fn hash_join_outer<T>(
    ca_in: &ChunkedArray<T>,
    other: &ChunkedArray<T>,
    validate: JoinValidation,
) -> PolarsResult<Vec<(Option<IdxSize>, Option<IdxSize>)>>
where
    T: PolarsIntegerType + Sync,
    T::Native: Eq + Hash + NumCast,
{
    let (a, b, swapped) = det_hash_prone_order!(ca_in, other);

    let n_partitions = _set_partition_size();
    let splitted_a = split_ca(a, n_partitions).unwrap();
    let splitted_b = split_ca(b, n_partitions).unwrap();

    match (a.null_count(), b.null_count()) {
        (0, 0) => {
            let iters_a = splitted_a
                .iter()
                .flat_map(|ca| ca.downcast_iter().map(|arr| arr.values().as_slice()))
                .collect::<Vec<_>>();
            let iters_b = splitted_b
                .iter()
                .flat_map(|ca| ca.downcast_iter().map(|arr| arr.values().as_slice()))
                .collect::<Vec<_>>();
            hash_join_tuples_outer(iters_a, iters_b, swapped, validate)
        },
        _ => {
            let iters_a = splitted_a
                .iter()
                .flat_map(|ca| ca.downcast_iter().map(|arr| arr.iter()))
                .collect::<Vec<_>>();
            let iters_b = splitted_b
                .iter()
                .flat_map(|ca| ca.downcast_iter().map(|arr| arr.iter()))
                .collect::<Vec<_>>();
            hash_join_tuples_outer(iters_a, iters_b, swapped, validate)
        },
    }
}

pub fn prepare_bytes<'a>(
    been_split: &'a [BinaryChunked],
    hb: &RandomState,
) -> Vec<Vec<BytesHash<'a>>> {
    POOL.install(|| {
        been_split
            .par_iter()
            .map(|ca| {
                ca.into_iter()
                    .map(|opt_b| {
                        let hash = hb.hash_one(opt_b);
                        BytesHash::new(opt_b, hash)
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    })
}

fn prepare_binary<'a>(
    ca: &'a BinaryChunked,
    other: &'a BinaryChunked,
    // In inner join and outer join, the shortest relation will be used to create a hash table.
    // In left join, always use the right side to create.
    build_shortest_table: bool,
) -> (
    Vec<Vec<BytesHash<'a>>>,
    Vec<Vec<BytesHash<'a>>>,
    bool,
    RandomState,
) {
    let n_threads = POOL.current_num_threads();

    let (a, b, swapped) = if build_shortest_table {
        det_hash_prone_order!(ca, other)
    } else {
        (ca, other, false)
    };

    let hb = RandomState::default();
    let splitted_a = split_ca(a, n_threads).unwrap();
    let splitted_b = split_ca(b, n_threads).unwrap();
    let str_hashes_a = prepare_bytes(&splitted_a, &hb);
    let str_hashes_b = prepare_bytes(&splitted_b, &hb);

    // SAFETY:
    // Splitting a Ca keeps the same buffers, so the lifetime is still valid.
    let str_hashes_a = unsafe { std::mem::transmute::<_, Vec<Vec<BytesHash<'a>>>>(str_hashes_a) };
    let str_hashes_b = unsafe { std::mem::transmute::<_, Vec<Vec<BytesHash<'a>>>>(str_hashes_b) };

    (str_hashes_a, str_hashes_b, swapped, hb)
}

#[cfg(feature = "semi_anti_join")]
fn num_group_join_anti_semi<T>(
    left: &ChunkedArray<T>,
    right: &ChunkedArray<T>,
    anti: bool,
) -> Vec<IdxSize>
where
    T: PolarsIntegerType,
    T::Native: Hash + Eq + Send + AsU64,
    Option<T::Native>: AsU64,
{
    let n_threads = POOL.current_num_threads();
    let splitted_a = split_ca(left, n_threads).unwrap();
    let splitted_b = split_ca(right, n_threads).unwrap();
    match (
        left.null_count(),
        right.null_count(),
        left.chunks().len(),
        right.chunks().len(),
    ) {
        (0, 0, 1, 1) => {
            let keys_a = chunks_as_slices(&splitted_a);
            let keys_b = chunks_as_slices(&splitted_b);
            if anti {
                hash_join_tuples_left_anti(keys_a, keys_b)
            } else {
                hash_join_tuples_left_semi(keys_a, keys_b)
            }
        },
        (0, 0, _, _) => {
            let keys_a = chunks_as_slices(&splitted_a);
            let keys_b = chunks_as_slices(&splitted_b);
            if anti {
                hash_join_tuples_left_anti(keys_a, keys_b)
            } else {
                hash_join_tuples_left_semi(keys_a, keys_b)
            }
        },
        _ => {
            let keys_a = get_arrays(&splitted_a);
            let keys_b = get_arrays(&splitted_b);
            if anti {
                hash_join_tuples_left_anti(keys_a, keys_b)
            } else {
                hash_join_tuples_left_semi(keys_a, keys_b)
            }
        },
    }
}
