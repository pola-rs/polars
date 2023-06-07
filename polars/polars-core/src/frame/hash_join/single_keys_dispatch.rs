use num_traits::NumCast;

use super::single_keys_inner::hash_join_tuples_inner;
use super::*;
#[cfg(feature = "chunked_ids")]
use crate::utils::create_chunked_index_mapping;

impl Series {
    #[doc(hidden)]
    pub fn hash_join_left(
        &self,
        other: &Series,
        validate: JoinValidation,
    ) -> PolarsResult<LeftJoinIds> {
        let (lhs, rhs) = (self.to_physical_repr(), other.to_physical_repr());
        validate.validate_probe(&lhs, &rhs, false)?;

        use DataType::*;
        match lhs.dtype() {
            Utf8 => {
                let lhs = lhs.utf8().unwrap();
                let rhs = rhs.utf8().unwrap();

                let lhs = lhs.as_binary();
                let rhs = rhs.as_binary();
                lhs.hash_join_left(&rhs, validate)
            }
            Binary => {
                let lhs = lhs.binary().unwrap();
                let rhs = rhs.binary().unwrap();
                lhs.hash_join_left(rhs, validate)
            }
            _ => {
                if self.bit_repr_is_large() {
                    let lhs = lhs.bit_repr_large();
                    let rhs = rhs.bit_repr_large();
                    num_group_join_left(&lhs, &rhs, validate)
                } else {
                    let lhs = lhs.bit_repr_small();
                    let rhs = rhs.bit_repr_small();
                    num_group_join_left(&lhs, &rhs, validate)
                }
            }
        }
    }

    #[cfg(feature = "semi_anti_join")]
    pub(super) fn hash_join_semi_anti(&self, other: &Series, anti: bool) -> Vec<IdxSize> {
        let (lhs, rhs) = (self.to_physical_repr(), other.to_physical_repr());

        use DataType::*;
        match lhs.dtype() {
            Utf8 => {
                let lhs = lhs.cast(&Binary).unwrap();
                let rhs = rhs.cast(&Binary).unwrap();

                let lhs = lhs.binary().unwrap();
                let rhs = rhs.binary().unwrap();
                lhs.hash_join_semi_anti(rhs, anti)
            }
            Binary => {
                let lhs = lhs.binary().unwrap();
                let rhs = rhs.binary().unwrap();
                lhs.hash_join_semi_anti(rhs, anti)
            }
            _ => {
                if self.bit_repr_is_large() {
                    let lhs = lhs.bit_repr_large();
                    let rhs = rhs.bit_repr_large();
                    num_group_join_anti_semi(&lhs, &rhs, anti)
                } else {
                    let lhs = lhs.bit_repr_small();
                    let rhs = rhs.bit_repr_small();
                    num_group_join_anti_semi(&lhs, &rhs, anti)
                }
            }
        }
    }

    // returns the join tuples and whether or not the lhs tuples are sorted
    pub(super) fn hash_join_inner(
        &self,
        other: &Series,
        validate: JoinValidation,
    ) -> PolarsResult<(InnerJoinIds, bool)> {
        let (lhs, rhs) = (self.to_physical_repr(), other.to_physical_repr());
        validate.validate_probe(&lhs, &rhs, true)?;

        use DataType::*;
        match lhs.dtype() {
            Utf8 => {
                let lhs = lhs.utf8().unwrap();
                let rhs = rhs.utf8().unwrap();

                let lhs = lhs.as_binary();
                let rhs = rhs.as_binary();
                lhs.hash_join_inner(&rhs, validate)
            }
            Binary => {
                let lhs = lhs.binary().unwrap();
                let rhs = rhs.binary().unwrap();
                lhs.hash_join_inner(rhs, validate)
            }
            _ => {
                if self.bit_repr_is_large() {
                    let lhs = self.bit_repr_large();
                    let rhs = other.bit_repr_large();
                    num_group_join_inner(&lhs, &rhs, validate)
                } else {
                    let lhs = self.bit_repr_small();
                    let rhs = other.bit_repr_small();
                    num_group_join_inner(&lhs, &rhs, validate)
                }
            }
        }
    }

    pub(super) fn hash_join_outer(
        &self,
        other: &Series,
        validate: JoinValidation,
    ) -> PolarsResult<Vec<(Option<IdxSize>, Option<IdxSize>)>> {
        let (lhs, rhs) = (self.to_physical_repr(), other.to_physical_repr());
        validate.validate_probe(&lhs, &rhs, true)?;

        use DataType::*;
        match lhs.dtype() {
            Utf8 => {
                let lhs = lhs.utf8().unwrap();
                let rhs = rhs.utf8().unwrap();

                let lhs = lhs.as_binary();
                let rhs = rhs.as_binary();
                lhs.hash_join_outer(&rhs, validate)
            }
            Binary => {
                let lhs = lhs.binary().unwrap();
                let rhs = rhs.binary().unwrap();
                lhs.hash_join_outer(rhs, validate)
            }
            _ => {
                if self.bit_repr_is_large() {
                    let lhs = self.bit_repr_large();
                    let rhs = other.bit_repr_large();
                    lhs.hash_join_outer(&rhs, validate)
                } else {
                    let lhs = self.bit_repr_small();
                    let rhs = other.bit_repr_small();
                    lhs.hash_join_outer(&rhs, validate)
                }
            }
        }
    }
}

fn splitted_to_slice<T>(splitted: &[ChunkedArray<T>]) -> Vec<&[T::Native]>
where
    T: PolarsNumericType,
{
    splitted.iter().map(|ca| ca.cont_slice().unwrap()).collect()
}

fn splitted_by_chunks<T>(splitted: &[ChunkedArray<T>]) -> Vec<&[T::Native]>
where
    T: PolarsNumericType,
{
    splitted
        .iter()
        .flat_map(|ca| ca.downcast_iter().map(|arr| arr.values().as_slice()))
        .collect()
}

fn splitted_to_opt_vec<T>(splitted: &[ChunkedArray<T>]) -> Vec<Vec<Option<T::Native>>>
where
    T: PolarsNumericType,
{
    POOL.install(|| {
        splitted
            .par_iter()
            .map(|ca| ca.into_iter().collect_trusted::<Vec<_>>())
            .collect()
    })
}

// returns the join tuples and whether or not the lhs tuples are sorted
fn num_group_join_inner<T>(
    left: &ChunkedArray<T>,
    right: &ChunkedArray<T>,
    validate: JoinValidation,
) -> PolarsResult<(InnerJoinIds, bool)>
where
    T: PolarsIntegerType,
    T::Native: Hash + Eq + Send + AsU64 + Copy,
    Option<T::Native>: AsU64,
{
    let n_threads = POOL.current_num_threads();
    let (a, b, swap) = det_hash_prone_order!(left, right);
    let splitted_a = split_ca(a, n_threads).unwrap();
    let splitted_b = split_ca(b, n_threads).unwrap();
    match (
        left.null_count() == 0,
        right.null_count() == 0,
        left.chunks.len(),
        right.chunks.len(),
    ) {
        (true, true, 1, 1) => {
            let keys_a = splitted_to_slice(&splitted_a);
            let keys_b = splitted_to_slice(&splitted_b);
            Ok((
                hash_join_tuples_inner(keys_a, keys_b, swap, validate)?,
                !swap,
            ))
        }
        (true, true, _, _) => {
            let keys_a = splitted_by_chunks(&splitted_a);
            let keys_b = splitted_by_chunks(&splitted_b);
            Ok((
                hash_join_tuples_inner(keys_a, keys_b, swap, validate)?,
                !swap,
            ))
        }
        _ => {
            let keys_a = splitted_to_opt_vec(&splitted_a);
            let keys_b = splitted_to_opt_vec(&splitted_b);
            Ok((
                hash_join_tuples_inner(keys_a, keys_b, swap, validate)?,
                !swap,
            ))
        }
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
        left.chunks.len(),
        right.chunks.len(),
    ) {
        (0, 0, 1, 1) => {
            let keys_a = splitted_to_slice(&splitted_a);
            let keys_b = splitted_to_slice(&splitted_b);
            hash_join_tuples_left(keys_a, keys_b, None, None, validate)
        }
        (0, 0, _, _) => {
            let keys_a = splitted_by_chunks(&splitted_a);
            let keys_b = splitted_by_chunks(&splitted_b);

            let (mapping_left, mapping_right) =
                create_mappings(left.chunks(), right.chunks(), left.len(), right.len());
            hash_join_tuples_left(
                keys_a,
                keys_b,
                mapping_left.as_deref(),
                mapping_right.as_deref(),
                validate,
            )
        }
        _ => {
            let keys_a = splitted_to_opt_vec(&splitted_a);
            let keys_b = splitted_to_opt_vec(&splitted_b);
            let (mapping_left, mapping_right) =
                create_mappings(left.chunks(), right.chunks(), left.len(), right.len());
            hash_join_tuples_left(
                keys_a,
                keys_b,
                mapping_left.as_deref(),
                mapping_right.as_deref(),
                validate,
            )
        }
    }
}

impl<T> ChunkedArray<T>
where
    T: PolarsIntegerType + Sync,
    T::Native: Eq + Hash + NumCast,
{
    fn hash_join_outer(
        &self,
        other: &ChunkedArray<T>,
        validate: JoinValidation,
    ) -> PolarsResult<Vec<(Option<IdxSize>, Option<IdxSize>)>> {
        let (a, b, swap) = det_hash_prone_order!(self, other);

        let n_partitions = _set_partition_size();
        let splitted_a = split_ca(a, n_partitions).unwrap();
        let splitted_b = split_ca(b, n_partitions).unwrap();

        match (a.null_count(), b.null_count()) {
            (0, 0) => {
                let iters_a = splitted_a
                    .iter()
                    .map(|ca| ca.into_no_null_iter())
                    .collect::<Vec<_>>();
                let iters_b = splitted_b
                    .iter()
                    .map(|ca| ca.into_no_null_iter())
                    .collect::<Vec<_>>();
                hash_join_tuples_outer(iters_a, iters_b, swap, validate)
            }
            _ => {
                let iters_a = splitted_a
                    .iter()
                    .map(|ca| ca.into_iter())
                    .collect::<Vec<_>>();
                let iters_b = splitted_b
                    .iter()
                    .map(|ca| ca.into_iter())
                    .collect::<Vec<_>>();
                hash_join_tuples_outer(iters_a, iters_b, swap, validate)
            }
        }
    }
}

pub(crate) fn prepare_bytes<'a>(
    been_split: &'a [BinaryChunked],
    hb: &RandomState,
) -> Vec<Vec<BytesHash<'a>>> {
    POOL.install(|| {
        been_split
            .par_iter()
            .map(|ca| {
                ca.into_iter()
                    .map(|opt_b| {
                        let mut state = hb.build_hasher();
                        opt_b.hash(&mut state);
                        let hash = state.finish();
                        BytesHash::new(opt_b, hash)
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    })
}

impl BinaryChunked {
    fn prepare(
        &self,
        other: &BinaryChunked,
        swapped: bool,
    ) -> (Vec<Self>, Vec<Self>, bool, RandomState) {
        let n_threads = POOL.current_num_threads();

        let (a, b, swap) = if swapped {
            det_hash_prone_order!(self, other)
        } else {
            (self, other, false)
        };

        let hb = RandomState::default();
        let splitted_a = split_ca(a, n_threads).unwrap();
        let splitted_b = split_ca(b, n_threads).unwrap();

        (splitted_a, splitted_b, swap, hb)
    }

    // returns the join tuples and whether or not the lhs tuples are sorted
    fn hash_join_inner(
        &self,
        other: &BinaryChunked,
        validate: JoinValidation,
    ) -> PolarsResult<(InnerJoinIds, bool)> {
        let (splitted_a, splitted_b, swap, hb) = self.prepare(other, true);
        let str_hashes_a = prepare_bytes(&splitted_a, &hb);
        let str_hashes_b = prepare_bytes(&splitted_b, &hb);
        Ok((
            hash_join_tuples_inner(str_hashes_a, str_hashes_b, swap, validate)?,
            !swap,
        ))
    }

    fn hash_join_left(
        &self,
        other: &BinaryChunked,
        validate: JoinValidation,
    ) -> PolarsResult<LeftJoinIds> {
        let (splitted_a, splitted_b, _, hb) = self.prepare(other, false);
        let str_hashes_a = prepare_bytes(&splitted_a, &hb);
        let str_hashes_b = prepare_bytes(&splitted_b, &hb);

        let (mapping_left, mapping_right) =
            create_mappings(self.chunks(), other.chunks(), self.len(), other.len());
        hash_join_tuples_left(
            str_hashes_a,
            str_hashes_b,
            mapping_left.as_deref(),
            mapping_right.as_deref(),
            validate,
        )
    }

    #[cfg(feature = "semi_anti_join")]
    fn hash_join_semi_anti(&self, other: &BinaryChunked, anti: bool) -> Vec<IdxSize> {
        let (splitted_a, splitted_b, _, hb) = self.prepare(other, false);
        let str_hashes_a = prepare_bytes(&splitted_a, &hb);
        let str_hashes_b = prepare_bytes(&splitted_b, &hb);
        if anti {
            hash_join_tuples_left_anti(str_hashes_a, str_hashes_b)
        } else {
            hash_join_tuples_left_semi(str_hashes_a, str_hashes_b)
        }
    }

    fn hash_join_outer(
        &self,
        other: &BinaryChunked,
        validate: JoinValidation,
    ) -> PolarsResult<Vec<(Option<IdxSize>, Option<IdxSize>)>> {
        let (a, b, swap) = det_hash_prone_order!(self, other);

        let n_partitions = _set_partition_size();
        let splitted_a = split_ca(a, n_partitions).unwrap();
        let splitted_b = split_ca(b, n_partitions).unwrap();

        match (a.has_validity(), b.has_validity()) {
            (false, false) => {
                let iters_a = splitted_a
                    .iter()
                    .map(|ca| ca.into_no_null_iter())
                    .collect::<Vec<_>>();
                let iters_b = splitted_b
                    .iter()
                    .map(|ca| ca.into_no_null_iter())
                    .collect::<Vec<_>>();
                hash_join_tuples_outer(iters_a, iters_b, swap, validate)
            }
            _ => {
                let iters_a = splitted_a
                    .iter()
                    .map(|ca| ca.into_iter())
                    .collect::<Vec<_>>();
                let iters_b = splitted_b
                    .iter()
                    .map(|ca| ca.into_iter())
                    .collect::<Vec<_>>();
                hash_join_tuples_outer(iters_a, iters_b, swap, validate)
            }
        }
    }
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
        left.chunks.len(),
        right.chunks.len(),
    ) {
        (0, 0, 1, 1) => {
            let keys_a = splitted_to_slice(&splitted_a);
            let keys_b = splitted_to_slice(&splitted_b);
            if anti {
                hash_join_tuples_left_anti(keys_a, keys_b)
            } else {
                hash_join_tuples_left_semi(keys_a, keys_b)
            }
        }
        (0, 0, _, _) => {
            let keys_a = splitted_by_chunks(&splitted_a);
            let keys_b = splitted_by_chunks(&splitted_b);
            if anti {
                hash_join_tuples_left_anti(keys_a, keys_b)
            } else {
                hash_join_tuples_left_semi(keys_a, keys_b)
            }
        }
        _ => {
            let keys_a = splitted_to_opt_vec(&splitted_a);
            let keys_b = splitted_to_opt_vec(&splitted_b);
            if anti {
                hash_join_tuples_left_anti(keys_a, keys_b)
            } else {
                hash_join_tuples_left_semi(keys_a, keys_b)
            }
        }
    }
}
