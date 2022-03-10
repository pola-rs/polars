use super::*;
use crate::frame::hash_join::single_keys::{
    hash_join_tuples_inner, hash_join_tuples_left, hash_join_tuples_outer,
};

impl Series {
    #[cfg(feature = "private")]
    #[doc(hidden)]
    pub fn hash_join_left(&self, other: &Series) -> Vec<(IdxSize, Option<IdxSize>)> {
        let (lhs, rhs) = (self.to_physical_repr(), other.to_physical_repr());

        use DataType::*;
        match lhs.dtype() {
            Utf8 => {
                let lhs = lhs.utf8().unwrap();
                let rhs = rhs.utf8().unwrap();
                lhs.hash_join_left(rhs)
            }
            _ => {
                if self.bit_repr_is_large() {
                    let lhs = self.bit_repr_large();
                    let rhs = other.bit_repr_large();
                    num_group_join_left(&lhs, &rhs)
                } else {
                    let lhs = self.bit_repr_small();
                    let rhs = other.bit_repr_small();
                    num_group_join_left(&lhs, &rhs)
                }
            }
        }
    }

    pub(super) fn hash_join_inner(&self, other: &Series) -> Vec<(IdxSize, IdxSize)> {
        let (lhs, rhs) = (self.to_physical_repr(), other.to_physical_repr());

        use DataType::*;
        match lhs.dtype() {
            Utf8 => {
                let lhs = lhs.utf8().unwrap();
                let rhs = rhs.utf8().unwrap();
                lhs.hash_join_inner(rhs)
            }
            _ => {
                if self.bit_repr_is_large() {
                    let lhs = self.bit_repr_large();
                    let rhs = other.bit_repr_large();
                    num_group_join_inner(&lhs, &rhs)
                } else {
                    let lhs = self.bit_repr_small();
                    let rhs = other.bit_repr_small();
                    num_group_join_inner(&lhs, &rhs)
                }
            }
        }
    }

    pub(super) fn hash_join_outer(
        &self,
        other: &Series,
    ) -> Vec<(Option<IdxSize>, Option<IdxSize>)> {
        let (lhs, rhs) = (self.to_physical_repr(), other.to_physical_repr());

        use DataType::*;
        match lhs.dtype() {
            Utf8 => {
                let lhs = lhs.utf8().unwrap();
                let rhs = rhs.utf8().unwrap();
                lhs.hash_join_outer(rhs)
            }
            _ => {
                if self.bit_repr_is_large() {
                    let lhs = self.bit_repr_large();
                    let rhs = other.bit_repr_large();
                    lhs.hash_join_outer(&rhs)
                } else {
                    let lhs = self.bit_repr_small();
                    let rhs = other.bit_repr_small();
                    lhs.hash_join_outer(&rhs)
                }
            }
        }
    }
}

pub(crate) trait HashJoin<T> {
    fn hash_join_inner(&self, _other: &ChunkedArray<T>) -> Vec<(IdxSize, IdxSize)> {
        unimplemented!()
    }
    fn hash_join_left(&self, _other: &ChunkedArray<T>) -> Vec<(IdxSize, Option<IdxSize>)> {
        unimplemented!()
    }
    fn hash_join_outer(&self, _other: &ChunkedArray<T>) -> Vec<(Option<IdxSize>, Option<IdxSize>)> {
        unimplemented!()
    }
}

impl HashJoin<Float32Type> for Float32Chunked {
    fn hash_join_outer(&self, other: &Float32Chunked) -> Vec<(Option<IdxSize>, Option<IdxSize>)> {
        let ca = self.bit_repr_small();
        let other = other.bit_repr_small();
        ca.hash_join_outer(&other)
    }
}

impl HashJoin<Float64Type> for Float64Chunked {
    fn hash_join_outer(&self, other: &Float64Chunked) -> Vec<(Option<IdxSize>, Option<IdxSize>)> {
        let ca = self.bit_repr_large();
        let other = other.bit_repr_large();
        ca.hash_join_outer(&other)
    }
}

fn num_group_join_inner<T>(
    left: &ChunkedArray<T>,
    right: &ChunkedArray<T>,
) -> Vec<(IdxSize, IdxSize)>
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
            let keys_a = splitted_a
                .iter()
                .map(|ca| ca.cont_slice().unwrap())
                .collect::<Vec<_>>();
            let keys_b = splitted_b
                .iter()
                .map(|ca| ca.cont_slice().unwrap())
                .collect::<Vec<_>>();
            hash_join_tuples_inner(keys_a, keys_b, swap)
        }
        (true, true, _, _) => {
            let keys_a = splitted_a
                .iter()
                .map(|ca| ca.into_no_null_iter().collect::<Vec<_>>())
                .collect::<Vec<_>>();
            let keys_b = splitted_b
                .iter()
                .map(|ca| ca.into_no_null_iter().collect::<Vec<_>>())
                .collect::<Vec<_>>();
            hash_join_tuples_inner(keys_a, keys_b, swap)
        }
        (_, _, 1, 1) => {
            let keys_a = splitted_a
                .iter()
                .map(|ca| {
                    ca.downcast_iter()
                        .flat_map(|v| v.into_iter().map(|v| v.copied().as_u64()))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();

            let keys_b = splitted_b
                .iter()
                .map(|ca| {
                    ca.downcast_iter()
                        .flat_map(|v| v.into_iter().map(|v| v.copied().as_u64()))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            hash_join_tuples_inner(keys_a, keys_b, swap)
        }
        _ => {
            let keys_a = splitted_a
                .iter()
                .map(|ca| ca.into_iter().map(|v| v.as_u64()).collect::<Vec<_>>())
                .collect::<Vec<_>>();
            let keys_b = splitted_b
                .iter()
                .map(|ca| ca.into_iter().map(|v| v.as_u64()).collect::<Vec<_>>())
                .collect::<Vec<_>>();
            hash_join_tuples_inner(keys_a, keys_b, swap)
        }
    }
}

fn num_group_join_left<T>(
    left: &ChunkedArray<T>,
    right: &ChunkedArray<T>,
) -> Vec<(IdxSize, Option<IdxSize>)>
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
            let keys_a = splitted_a
                .iter()
                .map(|ca| ca.cont_slice().unwrap())
                .collect::<Vec<_>>();
            let keys_b = splitted_b
                .iter()
                .map(|ca| ca.cont_slice().unwrap())
                .collect::<Vec<_>>();
            hash_join_tuples_left(keys_a, keys_b)
        }
        (0, 0, _, _) => {
            let keys_a = splitted_a
                .iter()
                .map(|ca| ca.into_no_null_iter().collect_trusted::<Vec<_>>())
                .collect::<Vec<_>>();
            let keys_b = splitted_b
                .iter()
                .map(|ca| ca.into_no_null_iter().collect_trusted::<Vec<_>>())
                .collect::<Vec<_>>();
            hash_join_tuples_left(keys_a, keys_b)
        }
        (_, _, 1, 1) => {
            let keys_a = splitted_a
                .iter()
                .map(|ca| {
                    // we know that we only iterate over length == self.len()
                    unsafe {
                        ca.downcast_iter()
                            .flat_map(|v| v.into_iter().map(|v| v.copied().as_u64()))
                            .trust_my_length(ca.len())
                            .collect_trusted::<Vec<_>>()
                    }
                })
                .collect::<Vec<_>>();

            let keys_b = splitted_b
                .iter()
                .map(|ca| {
                    // we know that we only iterate over length == self.len()
                    unsafe {
                        ca.downcast_iter()
                            .flat_map(|v| v.into_iter().map(|v| v.copied().as_u64()))
                            .trust_my_length(ca.len())
                            .collect_trusted::<Vec<_>>()
                    }
                })
                .collect::<Vec<_>>();
            hash_join_tuples_left(keys_a, keys_b)
        }
        _ => {
            let keys_a = splitted_a
                .iter()
                .map(|ca| {
                    ca.into_iter()
                        .map(|v| v.as_u64())
                        .collect_trusted::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let keys_b = splitted_b
                .iter()
                .map(|ca| {
                    ca.into_iter()
                        .map(|v| v.as_u64())
                        .collect_trusted::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            hash_join_tuples_left(keys_a, keys_b)
        }
    }
}

impl<T> HashJoin<T> for ChunkedArray<T>
where
    T: PolarsIntegerType + Sync,
    T::Native: Eq + Hash + num::NumCast,
{
    fn hash_join_outer(&self, other: &ChunkedArray<T>) -> Vec<(Option<IdxSize>, Option<IdxSize>)> {
        let (a, b, swap) = det_hash_prone_order!(self, other);

        let n_partitions = set_partition_size();
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
                hash_join_tuples_outer(iters_a, iters_b, swap)
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
                hash_join_tuples_outer(iters_a, iters_b, swap)
            }
        }
    }
}

impl HashJoin<BooleanType> for BooleanChunked {
    fn hash_join_outer(&self, other: &BooleanChunked) -> Vec<(Option<IdxSize>, Option<IdxSize>)> {
        let (a, b, swap) = det_hash_prone_order!(self, other);

        let n_partitions = set_partition_size();
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
                hash_join_tuples_outer(iters_a, iters_b, swap)
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
                hash_join_tuples_outer(iters_a, iters_b, swap)
            }
        }
    }
}

pub(crate) fn prepare_strs<'a>(
    been_split: &'a [Utf8Chunked],
    hb: &RandomState,
) -> Vec<Vec<StrHash<'a>>> {
    POOL.install(|| {
        been_split
            .par_iter()
            .map(|ca| {
                ca.into_iter()
                    .map(|opt_s| {
                        let mut state = hb.build_hasher();
                        opt_s.hash(&mut state);
                        let hash = state.finish();
                        StrHash::new(opt_s, hash)
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    })
}

impl HashJoin<Utf8Type> for Utf8Chunked {
    fn hash_join_inner(&self, other: &Utf8Chunked) -> Vec<(IdxSize, IdxSize)> {
        let n_threads = POOL.current_num_threads();

        let (a, b, swap) = det_hash_prone_order!(self, other);

        let hb = RandomState::default();
        let splitted_a = split_ca(a, n_threads).unwrap();
        let splitted_b = split_ca(b, n_threads).unwrap();

        let str_hashes_a = prepare_strs(&splitted_a, &hb);
        let str_hashes_b = prepare_strs(&splitted_b, &hb);
        hash_join_tuples_inner(str_hashes_a, str_hashes_b, swap)
    }

    fn hash_join_left(&self, other: &Utf8Chunked) -> Vec<(IdxSize, Option<IdxSize>)> {
        let n_threads = POOL.current_num_threads();

        let hb = RandomState::default();
        let splitted_a = split_ca(self, n_threads).unwrap();
        let splitted_b = split_ca(other, n_threads).unwrap();

        let str_hashes_a = prepare_strs(&splitted_a, &hb);
        let str_hashes_b = prepare_strs(&splitted_b, &hb);
        hash_join_tuples_left(str_hashes_a, str_hashes_b)
    }

    fn hash_join_outer(&self, other: &Utf8Chunked) -> Vec<(Option<IdxSize>, Option<IdxSize>)> {
        let (a, b, swap) = det_hash_prone_order!(self, other);

        let n_partitions = set_partition_size();
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
                hash_join_tuples_outer(iters_a, iters_b, swap)
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
                hash_join_tuples_outer(iters_a, iters_b, swap)
            }
        }
    }
}
