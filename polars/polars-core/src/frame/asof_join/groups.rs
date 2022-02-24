use super::*;
use crate::utils::{split_ca, split_df};
use crate::vector_hasher::{df_rows_to_hashes_threaded, AsU64};
use crate::POOL;
use ahash::RandomState;
use rayon::prelude::*;
use std::fmt::Debug;
use std::hash::Hash;

use crate::frame::hash_join::{
    create_probe_table, get_hash_tbl_threaded_join_partitioned, multiple_keys as mk, prepare_strs,
};

fn find_latest_leq<T>(left_val: T, right_asof: &[T], subset_idx: &[IdxSize]) -> Option<IdxSize>
where
    T: Copy + PartialOrd,
{
    subset_idx
        .iter()
        .rev()
        .find(|&&i| {
            debug_assert!((i as usize) < right_asof.len());
            // Safety:
            // idx are in bounds
            unsafe { *right_asof.get_unchecked(i as usize) <= left_val }
        })
        .copied()
}

pub(super) unsafe fn join_asof_backward_with_indirection<T: PartialOrd + Copy + Debug>(
    val_l: T,
    right: &[T],
    offsets: &[IdxSize],
) -> (Option<IdxSize>, usize) {
    if offsets.is_empty() {
        return (None, 0);
    }
    let mut previous = *offsets.get_unchecked(0);
    let first = *right.get_unchecked(previous as usize);
    if val_l < first {
        (None, 0)
    } else {
        for (idx, &offset) in offsets.iter().enumerate() {
            let val_r = *right.get_unchecked(offset as usize);
            if val_r > val_l {
                return (Some(previous), idx);
            }
            previous = offset
        }
        (Some(previous), offsets.len())
    }
}

fn asof_join_by_numeric<T, S>(
    by_left: &ChunkedArray<S>,
    by_right: &ChunkedArray<S>,
    left_asof: &ChunkedArray<T>,
    right_asof: &ChunkedArray<T>,
) -> Vec<Option<IdxSize>>
where
    T: PolarsNumericType,
    S: PolarsNumericType,
    S::Native: Hash + Eq + AsU64,
{
    let left_asof = left_asof.rechunk();
    let left_asof = left_asof.cont_slice().unwrap();

    let right_asof = right_asof.rechunk();
    let right_asof = right_asof.cont_slice().unwrap();

    let n_threads = POOL.current_num_threads();
    let splitted_left = split_ca(by_left, n_threads).unwrap();
    let splitted_right = split_ca(by_right, n_threads).unwrap();

    let vals_left = splitted_left
        .iter()
        .map(|ca| ca.cont_slice().unwrap())
        .collect::<Vec<_>>();
    let vals_right = splitted_right
        .iter()
        .map(|ca| ca.cont_slice().unwrap())
        .collect::<Vec<_>>();

    let hash_tbls = create_probe_table(vals_right);

    // we determine the offset so that we later know which index to store in the join tuples
    let offsets = vals_left
        .iter()
        .map(|ph| ph.len())
        .scan(0, |state, val| {
            let out = *state;
            *state += val;
            Some(out)
        })
        .collect::<Vec<_>>();

    let n_tables = hash_tbls.len() as u64;
    debug_assert!(n_tables.is_power_of_two());

    // next we probe the right relation
    POOL.install(|| {
        vals_left
            .into_par_iter()
            .zip(offsets)
            // probes_hashes: Vec<u64> processed by this thread
            // offset: offset index
            .map(|(vals_left, offset)| {
                // local reference
                let hash_tbls = &hash_tbls;

                // assume the result tuples equal lenght of the no. of hashes processed by this thread.
                let mut results = Vec::with_capacity(vals_left.len());

                let mut right_tbl_offsets = PlHashMap::with_capacity(64);

                vals_left.iter().enumerate().for_each(|(idx_a, k)| {
                    let idx_a = (idx_a + offset) as IdxSize;
                    // probe table that contains the hashed value
                    let current_probe_table = unsafe {
                        get_hash_tbl_threaded_join_partitioned(k.as_u64(), hash_tbls, n_tables)
                    };

                    // we already hashed, so we don't have to hash again.
                    let value = current_probe_table.get(k);

                    match value {
                        // left and right matches
                        Some(indexes_b) => {
                            let (offset_slice, previous_join_idx) =
                                *right_tbl_offsets.get(k).unwrap_or(&(0usize, None));
                            let val_l = left_asof[idx_a as usize];
                            // Safety;
                            // elide bound checks
                            let (join_idx, offset_slice_add) = unsafe {
                                join_asof_backward_with_indirection(
                                    val_l,
                                    right_asof,
                                    &indexes_b[offset_slice..],
                                )
                            };
                            let offset_slice = offset_slice + offset_slice_add;

                            match join_idx {
                                Some(_) => {
                                    results.push(join_idx);
                                    right_tbl_offsets.insert(k, (offset_slice, join_idx));
                                }
                                None => results.push(previous_join_idx),
                            }
                        }
                        // only left values, right = null
                        None => results.push(None),
                    }
                });
                results
            })
            .flatten()
            .collect()
    })
}

fn asof_join_by_utf8<T>(
    by_left: &Utf8Chunked,
    by_right: &Utf8Chunked,
    left_asof: &ChunkedArray<T>,
    right_asof: &ChunkedArray<T>,
) -> Vec<Option<IdxSize>>
where
    T: PolarsNumericType,
{
    let left_asof = left_asof.rechunk();
    let left_asof = left_asof.cont_slice().unwrap();

    let right_asof = right_asof.rechunk();
    let right_asof = right_asof.cont_slice().unwrap();

    let n_threads = POOL.current_num_threads();
    let splitted_left = split_ca(by_left, n_threads).unwrap();
    let splitted_right = split_ca(by_right, n_threads).unwrap();

    let hb = RandomState::default();
    let vals_left = prepare_strs(&splitted_left, &hb);
    let vals_right = prepare_strs(&splitted_right, &hb);

    let hash_tbls = create_probe_table(vals_right);

    // we determine the offset so that we later know which index to store in the join tuples
    let offsets = vals_left
        .iter()
        .map(|ph| ph.len())
        .scan(0, |state, val| {
            let out = *state;
            *state += val;
            Some(out)
        })
        .collect::<Vec<_>>();

    let n_tables = hash_tbls.len() as u64;
    debug_assert!(n_tables.is_power_of_two());

    // next we probe the right relation
    POOL.install(|| {
        vals_left
            .into_par_iter()
            .zip(offsets)
            // probes_hashes: Vec<u64> processed by this thread
            // offset: offset index
            .map(|(vals_left, offset)| {
                // local reference
                let hash_tbls = &hash_tbls;

                // assume the result tuples equal lenght of the no. of hashes processed by this thread.
                let mut results = Vec::with_capacity(vals_left.len());

                let mut right_tbl_offsets = PlHashMap::with_capacity(64);

                vals_left.iter().enumerate().for_each(|(idx_a, k)| {
                    let idx_a = (idx_a + offset) as IdxSize;
                    // probe table that contains the hashed value
                    let current_probe_table = unsafe {
                        get_hash_tbl_threaded_join_partitioned(k.as_u64(), hash_tbls, n_tables)
                    };

                    // we already hashed, so we don't have to hash again.
                    let value = current_probe_table.get(k);

                    match value {
                        // left and right matches
                        Some(indexes_b) => {
                            let (offset_slice, previous_join_idx) =
                                *right_tbl_offsets.get(k).unwrap_or(&(0usize, None));
                            let val_l = left_asof[idx_a as usize];
                            // Safety;
                            // elide bound checks
                            let (join_idx, offset_slice_add) = unsafe {
                                join_asof_backward_with_indirection(
                                    val_l,
                                    right_asof,
                                    &indexes_b[offset_slice..],
                                )
                            };
                            let offset_slice = offset_slice + offset_slice_add;

                            match join_idx {
                                Some(_) => {
                                    results.push(join_idx);
                                    right_tbl_offsets.insert(k, (offset_slice, join_idx));
                                }
                                None => results.push(previous_join_idx),
                            }
                        }
                        // only left values, right = null
                        None => results.push(None),
                    }
                });
                results
            })
            .flatten()
            .collect()
    })
}

// TODO! optimize this. This does a full scan backwards. Use the same strategy as in the single `by`
// implementations
fn asof_join_by_multiple<T>(
    a: &DataFrame,
    b: &DataFrame,
    left_asof: &ChunkedArray<T>,
    right_asof: &ChunkedArray<T>,
) -> Vec<Option<IdxSize>>
where
    T: PolarsNumericType,
{
    let left_asof = left_asof.rechunk();
    let left_asof = left_asof.cont_slice().unwrap();

    let right_asof = right_asof.rechunk();
    let right_asof = right_asof.cont_slice().unwrap();

    let n_threads = POOL.current_num_threads();
    let dfs_a = split_df(a, n_threads).unwrap();
    let dfs_b = split_df(b, n_threads).unwrap();

    let (build_hashes, random_state) = df_rows_to_hashes_threaded(&dfs_b, None);
    let (probe_hashes, _) = df_rows_to_hashes_threaded(&dfs_a, Some(random_state));

    let hash_tbls = mk::create_build_table(&build_hashes, b);
    // early drop to reduce memory pressure
    drop(build_hashes);

    let n_tables = hash_tbls.len() as u64;
    let offsets = mk::get_offsets(&probe_hashes);

    // next we probe the other relation
    // code duplication is because we want to only do the swap check once
    POOL.install(|| {
        probe_hashes
            .into_par_iter()
            .zip(offsets)
            .map(|(probe_hashes, offset)| {
                // local reference
                let hash_tbls = &hash_tbls;
                let mut results =
                    Vec::with_capacity(probe_hashes.len() / POOL.current_num_threads());
                let local_offset = offset;

                let mut idx_a = local_offset as IdxSize;
                for probe_hashes in probe_hashes.data_views() {
                    for (idx, &h) in probe_hashes.iter().enumerate() {
                        debug_assert!(idx + offset < left_asof.len());
                        // Safety:
                        // idx are in bounds
                        let left_val = unsafe { *left_asof.get_unchecked(idx + offset) };

                        // probe table that contains the hashed value
                        let current_probe_table = unsafe {
                            get_hash_tbl_threaded_join_partitioned(h, hash_tbls, n_tables)
                        };

                        let entry = current_probe_table.raw_entry().from_hash(h, |idx_hash| {
                            let idx_b = idx_hash.idx;
                            // Safety:
                            // indices in a join operation are always in bounds.
                            unsafe { mk::compare_df_rows2(a, b, idx_a as usize, idx_b as usize) }
                        });

                        match entry {
                            // left and right matches
                            Some((_, indexes_b)) => {
                                results.push(find_latest_leq(left_val, right_asof, indexes_b))
                            }
                            // only left values, right = null
                            None => results.push(None),
                        }
                        idx_a += 1;
                    }
                }

                results
            })
            .flatten()
            .collect()
    })
}

impl DataFrame {
    /// This is similar to a left-join except that we match on nearest key rather than equal keys.
    /// The keys must be sorted to perform an asof join. This is a special implementation of an asof join
    /// that searches for the nearest keys within a subgroup set by `by`.
    #[cfg_attr(docsrs, doc(cfg(feature = "asof_join")))]
    pub fn join_asof_by<I, S>(
        &self,
        other: &DataFrame,
        left_on: &str,
        right_on: &str,
        left_by: I,
        right_by: I,
        strategy: AsofStrategy,
    ) -> Result<DataFrame>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        if let AsofStrategy::Forward = strategy {
            panic!("forward strategy + groupby not yet implemented");
        }

        use DataType::*;
        let left_asof = self.column(left_on)?;
        let right_asof = other.column(right_on)?;
        let right_asof_name = right_asof.name();

        check_asof_columns(left_asof, right_asof)?;

        let left_by = self.select(left_by)?;
        let right_by = other.select(right_by)?;

        let left_by_s = &left_by.get_columns()[0];
        let right_by_s = &right_by.get_columns()[0];

        let right_join_tuples = if left_asof.bit_repr_is_large() {
            // we cannot use bit repr as that loses ordering
            let left_asof = left_asof.cast(&DataType::Int64)?;
            let right_asof = right_asof.cast(&DataType::Int64)?;
            let left_asof = left_asof.i64().unwrap();
            let right_asof = right_asof.i64().unwrap();

            if left_by.width() == 1 {
                match left_by_s.dtype() {
                    Utf8 => asof_join_by_utf8(
                        left_by_s.utf8().unwrap(),
                        right_by_s.utf8().unwrap(),
                        left_asof,
                        right_asof,
                    ),
                    _ => {
                        if left_by_s.bit_repr_is_large() {
                            let left_by = left_by_s.bit_repr_large();
                            let right_by = right_by_s.bit_repr_large();
                            asof_join_by_numeric(&left_by, &right_by, left_asof, right_asof)
                        } else {
                            let left_by = left_by_s.bit_repr_small();
                            let right_by = right_by_s.bit_repr_small();
                            asof_join_by_numeric(&left_by, &right_by, left_asof, right_asof)
                        }
                    }
                }
            } else {
                asof_join_by_multiple(&left_by, &right_by, left_asof, right_asof)
            }
        } else {
            // we cannot use bit repr as that loses ordering
            let left_asof = left_asof.cast(&DataType::Int32)?;
            let right_asof = right_asof.cast(&DataType::Int32)?;
            let left_asof = left_asof.i32().unwrap();
            let right_asof = right_asof.i32().unwrap();

            if left_by.width() == 1 {
                match left_by_s.dtype() {
                    Utf8 => asof_join_by_utf8(
                        left_by_s.utf8().unwrap(),
                        right_by_s.utf8().unwrap(),
                        left_asof,
                        right_asof,
                    ),
                    _ => {
                        if left_by_s.bit_repr_is_large() {
                            let left_by = left_by_s.bit_repr_large();
                            let right_by = right_by_s.bit_repr_large();
                            asof_join_by_numeric(&left_by, &right_by, left_asof, right_asof)
                        } else {
                            let left_by = left_by_s.bit_repr_small();
                            let right_by = right_by_s.bit_repr_small();
                            asof_join_by_numeric(&left_by, &right_by, left_asof, right_asof)
                        }
                    }
                }
            } else {
                asof_join_by_multiple(&left_by, &right_by, left_asof, right_asof)
            }
        };

        let mut drop_these = right_by.get_column_names();
        drop_these.push(right_asof_name);

        let cols = other
            .get_columns()
            .iter()
            .filter_map(|s| {
                if drop_these.contains(&s.name()) {
                    None
                } else {
                    Some(s.clone())
                }
            })
            .collect();
        let other = DataFrame::new_no_checks(cols);

        // Safety:
        // join tuples are in bounds
        let right_df = unsafe {
            other.take_opt_iter_unchecked(
                right_join_tuples
                    .into_iter()
                    .map(|opt_idx| opt_idx.map(|idx| idx as usize)),
            )
        };

        self.finish_join(self.clone(), right_df, None)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_asof_by() -> Result<()> {
        let a = df![
        "a" => [-1, 2, 3, 3, 3, 4],
        "b" => ["a", "b", "c", "d", "e", "f"]
        ]?;

        let b = df![
         "a" => [1, 2, 3, 3],
            "b" => ["a", "b", "c", "d"],
            "right_vals" => [1, 2, 3, 4]
        ]?;

        let out = a.join_asof_by(&b, "a", "a", ["b"], ["b"], AsofStrategy::Backward)?;
        assert_eq!(out.get_column_names(), &["a", "b", "right_vals"]);
        let out = out.column("right_vals").unwrap();
        let out = out.i32().unwrap();
        assert_eq!(
            Vec::from(out),
            &[None, Some(2), Some(3), Some(4), None, None]
        );
        Ok(())
    }

    #[test]
    fn test_asof_by2() -> Result<()> {
        let trades = df![
            "time" => [23i64, 38, 48, 48, 48],
            "ticker" => ["MSFT", "MSFT", "GOOG", "GOOG", "AAPL"],
            "groups_numeric" => [1, 1, 2, 2, 3],
            "bid" => [51.95, 51.95, 720.77, 720.92, 98.0]
        ]?;

        let quotes = df![
                   "time" => [23i64,
        23,
        30,
        41,
        48,
        49,
        72,
        75],
                   "ticker" => ["GOOG", "MSFT", "MSFT", "MSFT", "GOOG", "AAPL", "GOOG", "MSFT"],
                   "groups_numeric" => [2, 1, 1, 1, 2, 3, 2, 1],
                   "bid" => [720.5, 51.95, 51.97, 51.99, 720.5, 97.99, 720.5, 52.01]

               ]?;

        let out = trades.join_asof_by(
            &quotes,
            "time",
            "time",
            ["ticker"],
            ["ticker"],
            AsofStrategy::Backward,
        )?;
        let a = out.column("bid_right").unwrap();
        let a = a.f64().unwrap();
        let expected = &[Some(51.95), Some(51.97), Some(720.5), Some(720.5), None];

        assert_eq!(Vec::from(a), expected);

        let out = trades.join_asof_by(
            &quotes,
            "time",
            "time",
            ["groups_numeric"],
            ["groups_numeric"],
            AsofStrategy::Backward,
        )?;
        let a = out.column("bid_right").unwrap();
        let a = a.f64().unwrap();

        assert_eq!(Vec::from(a), expected);

        Ok(())
    }
}
