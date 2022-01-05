use super::*;
use crate::frame::select::Selection;
use crate::utils::split_df;
use crate::vector_hasher::df_rows_to_hashes_threaded;
use crate::POOL;
use rayon::prelude::*;

use crate::frame::hash_join::{get_hash_tbl_threaded_join_partitioned, multiple_keys as mk};

fn find_latest_leq<T>(left_val: T, right_asof: &[T], subset_idx: &[u32]) -> Option<u32>
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

// TODO! add faster implementation that has a single groupby key
fn asof_join_by<T>(
    a: &DataFrame,
    b: &DataFrame,
    left_asof: &ChunkedArray<T>,
    right_asof: &ChunkedArray<T>,
) -> Vec<Option<u32>>
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

                let mut idx_a = local_offset as u32;
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
    pub fn join_asof_by<'a, S, J>(
        &self,
        other: &DataFrame,
        left_on: &str,
        right_on: &str,
        left_by: S,
        right_by: S,
    ) -> Result<DataFrame>
    where
        S: Selection<'a, J>,
    {
        let left_asof = self.column(left_on)?;
        let right_asof = other.column(right_on)?;
        let right_asof_name = right_asof.name();

        let left_by = self.select(left_by)?;
        let right_by = other.select(right_by)?;

        let right_join_tuples = if left_asof.bit_repr_is_large() {
            let left_asof = left_asof.cast(&DataType::Int64)?;
            let right_asof = right_asof.cast(&DataType::Int64)?;
            let left_asof = left_asof.i64().unwrap();
            let right_asof = right_asof.i64().unwrap();

            asof_join_by(&left_by, &right_by, left_asof, right_asof)
        } else {
            let left_asof = left_asof.cast(&DataType::Int32)?;
            let right_asof = right_asof.cast(&DataType::Int32)?;
            let left_asof = left_asof.i32().unwrap();
            let right_asof = right_asof.i32().unwrap();
            asof_join_by(&left_by, &right_by, left_asof, right_asof)
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

        let out = a.join_asof_by(&b, "a", "a", "b", "b")?;
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
            "time" => [1464183000023i64, 1464183000038, 1464183000048, 1464183000048, 1464183000048],
            "ticker" => ["MSFT", "MSFT", "GOOG", "GOOG", "AAPL"],
            "bid" => [51.95, 51.95, 720.77, 720.92, 98.0]
        ]?;

        let quotes = df![
                   "time" => [1464183000023i64,
        1464183000023,
        1464183000030,
        1464183000041,
        1464183000048,
        1464183000049,
        1464183000072,
        1464183000075],
                   "ticker" => ["GOOG", "MSFT", "MSFT", "MSFT", "GOOG", "AAPL", "GOOG", "MSFT"],
                   "bid" => [720.5, 51.95, 51.97, 51.99, 720.5, 97.99, 720.5, 52.01]

               ]?;

        let out = trades.join_asof_by(&quotes, "time", "time", "ticker", "ticker")?;
        let a = out.column("bid_right").unwrap();
        let a = a.f64().unwrap();

        assert_eq!(
            Vec::from(a),
            &[Some(51.95), Some(51.97), Some(720.5), Some(720.5), None]
        );

        Ok(())
    }
}
