pub(super) mod multiple_keys;
pub(super) mod single_keys;
mod single_keys_dispatch;
mod single_keys_inner;
mod single_keys_left;
mod single_keys_outer;
#[cfg(feature = "semi_anti_join")]
mod single_keys_semi_anti;
pub(super) mod sort_merge;
mod zip_outer;

pub use args::*;
pub use multiple_keys::private_left_join_multiple_keys;
pub(super) use multiple_keys::*;
use polars_core::utils::{_set_partition_size, slice_slice, split_ca};
use polars_core::POOL;
pub(super) use single_keys::*;
#[cfg(feature = "asof_join")]
pub(super) use single_keys_dispatch::prepare_bytes;
pub use single_keys_dispatch::SeriesJoin;
use single_keys_inner::*;
use single_keys_left::*;
use single_keys_outer::*;
#[cfg(feature = "semi_anti_join")]
use single_keys_semi_anti::*;
pub use sort_merge::*;
pub(super) use zip_outer::zip_outer_join_column;

pub use super::*;

pub fn default_join_ids() -> ChunkJoinOptIds {
    #[cfg(feature = "chunked_ids")]
    {
        Either::Left(vec![])
    }
    #[cfg(not(feature = "chunked_ids"))]
    {
        vec![]
    }
}

macro_rules! det_hash_prone_order {
    ($self:expr, $other:expr) => {{
        // The shortest relation will be used to create a hash table.
        let left_first = $self.len() > $other.len();
        let a;
        let b;
        if left_first {
            a = $self;
            b = $other;
        } else {
            b = $self;
            a = $other;
        }

        (a, b, !left_first)
    }};
}

#[cfg(feature = "performant")]
use arrow::legacy::conversion::primitive_to_vec;
pub(super) use det_hash_prone_order;
use polars_utils::hash_to_partition;

pub(super) unsafe fn get_hash_tbl_threaded_join_partitioned<Item>(
    h: u64,
    hash_tables: &[Item],
    len: u64,
) -> &Item {
    let i = hash_to_partition(h, len as usize);
    hash_tables.get_unchecked(i)
}

#[allow(clippy::type_complexity)]
unsafe fn get_hash_tbl_threaded_join_mut_partitioned<T, H>(
    h: u64,
    hash_tables: &mut [HashMap<T, (bool, Vec<IdxSize>), H>],
    len: u64,
) -> &mut HashMap<T, (bool, Vec<IdxSize>), H> {
    let i = hash_to_partition(h, len as usize);
    hash_tables.get_unchecked_mut(i)
}

pub trait JoinDispatch: IntoDf {
    /// # Safety
    /// Join tuples must be in bounds
    #[cfg(feature = "chunked_ids")]
    unsafe fn create_left_df_chunked(&self, chunk_ids: &[ChunkId], left_join: bool) -> DataFrame {
        let df_self = self.to_df();
        if left_join && chunk_ids.len() == df_self.height() {
            df_self.clone()
        } else {
            // left join keys are in ascending order
            let sorted = if left_join {
                IsSorted::Ascending
            } else {
                IsSorted::Not
            };
            df_self._take_chunked_unchecked(chunk_ids, sorted)
        }
    }

    /// # Safety
    /// Join tuples must be in bounds
    unsafe fn _create_left_df_from_slice(
        &self,
        join_tuples: &[IdxSize],
        left_join: bool,
        sorted_tuple_idx: bool,
    ) -> DataFrame {
        let df_self = self.to_df();
        if left_join && join_tuples.len() == df_self.height() {
            df_self.clone()
        } else {
            // left join tuples are always in ascending order
            let sorted = if left_join || sorted_tuple_idx {
                IsSorted::Ascending
            } else {
                IsSorted::Not
            };

            df_self._take_unchecked_slice_sorted(join_tuples, true, sorted)
        }
    }

    #[cfg(not(feature = "chunked_ids"))]
    fn _finish_left_join(
        &self,
        ids: LeftJoinIds,
        other: &DataFrame,
        args: JoinArgs,
    ) -> PolarsResult<DataFrame> {
        let ca_self = self.to_df();
        let (left_idx, right_idx) = ids;
        let materialize_left =
            || unsafe { ca_self._create_left_df_from_slice(&left_idx, true, true) };

        let materialize_right = || {
            let right_idx = &*right_idx;
            unsafe { other.take_unchecked(&right_idx.iter().copied().collect_ca("")) }
        };
        let (df_left, df_right) = POOL.join(materialize_left, materialize_right);

        _finish_join(df_left, df_right, args.suffix.as_deref())
    }

    #[cfg(feature = "chunked_ids")]
    fn _finish_left_join(
        &self,
        ids: LeftJoinIds,
        other: &DataFrame,
        args: JoinArgs,
    ) -> PolarsResult<DataFrame> {
        let ca_self = self.to_df();
        let suffix = &args.suffix;
        let (left_idx, right_idx) = ids;
        let materialize_left = || match left_idx {
            ChunkJoinIds::Left(left_idx) => unsafe {
                let mut left_idx = &*left_idx;
                if let Some((offset, len)) = args.slice {
                    left_idx = slice_slice(left_idx, offset, len);
                }
                ca_self._create_left_df_from_slice(left_idx, true, true)
            },
            ChunkJoinIds::Right(left_idx) => unsafe {
                let mut left_idx = &*left_idx;
                if let Some((offset, len)) = args.slice {
                    left_idx = slice_slice(left_idx, offset, len);
                }
                ca_self.create_left_df_chunked(left_idx, true)
            },
        };

        let materialize_right = || match right_idx {
            ChunkJoinOptIds::Left(right_idx) => unsafe {
                let mut right_idx = &*right_idx;
                if let Some((offset, len)) = args.slice {
                    right_idx = slice_slice(right_idx, offset, len);
                }
                other.take_unchecked(&right_idx.iter().copied().collect_ca(""))
            },
            ChunkJoinOptIds::Right(right_idx) => unsafe {
                let mut right_idx = &*right_idx;
                if let Some((offset, len)) = args.slice {
                    right_idx = slice_slice(right_idx, offset, len);
                }
                other._take_opt_chunked_unchecked(right_idx)
            },
        };
        let (df_left, df_right) = POOL.join(materialize_left, materialize_right);

        _finish_join(df_left, df_right, suffix.as_deref())
    }

    fn _left_join_from_series(
        &self,
        other: &DataFrame,
        s_left: &Series,
        s_right: &Series,
        args: JoinArgs,
        verbose: bool,
    ) -> PolarsResult<DataFrame> {
        let ca_self = self.to_df();
        #[cfg(feature = "dtype-categorical")]
        _check_categorical_src(s_left.dtype(), s_right.dtype())?;

        let mut left = ca_self.clone();
        let mut s_left = s_left.clone();
        // Eagerly limit left if possible.
        if let Some((offset, len)) = args.slice {
            if offset == 0 {
                left = left.slice(0, len);
                s_left = s_left.slice(0, len);
            }
        }

        // Ensure that the chunks are aligned otherwise we go OOB.
        let mut right = other.clone();
        let mut s_right = s_right.clone();
        if left.should_rechunk() {
            left.as_single_chunk_par();
            s_left = s_left.rechunk();
        }
        if right.should_rechunk() {
            right.as_single_chunk_par();
            s_right = s_right.rechunk();
        }
        let ids = sort_or_hash_left(&s_left, &s_right, verbose, args.validation)?;
        left._finish_left_join(ids, &right.drop(s_right.name()).unwrap(), args)
    }

    #[cfg(feature = "semi_anti_join")]
    /// # Safety
    /// `idx` must be in bounds
    unsafe fn _finish_anti_semi_join(
        &self,
        mut idx: &[IdxSize],
        slice: Option<(i64, usize)>,
    ) -> DataFrame {
        let ca_self = self.to_df();
        if let Some((offset, len)) = slice {
            idx = slice_slice(idx, offset, len);
        }
        // idx from anti-semi join should always be sorted
        ca_self._take_unchecked_slice_sorted(idx, true, IsSorted::Ascending)
    }

    #[cfg(feature = "semi_anti_join")]
    fn _semi_anti_join_from_series(
        &self,
        s_left: &Series,
        s_right: &Series,
        slice: Option<(i64, usize)>,
        anti: bool,
    ) -> PolarsResult<DataFrame> {
        let ca_self = self.to_df();
        #[cfg(feature = "dtype-categorical")]
        _check_categorical_src(s_left.dtype(), s_right.dtype())?;

        let idx = s_left.hash_join_semi_anti(s_right, anti);
        // Safety:
        // indices are in bounds
        Ok(unsafe { ca_self._finish_anti_semi_join(&idx, slice) })
    }
    fn _outer_join_from_series(
        &self,
        other: &DataFrame,
        s_left: &Series,
        s_right: &Series,
        args: JoinArgs,
    ) -> PolarsResult<DataFrame> {
        let ca_self = self.to_df();
        #[cfg(feature = "dtype-categorical")]
        _check_categorical_src(s_left.dtype(), s_right.dtype())?;

        // store this so that we can keep original column order.
        let join_column_index = ca_self
            .iter()
            .position(|s| s.name() == s_left.name())
            .unwrap();

        // Get the indexes of the joined relations
        let opt_join_tuples = s_left.hash_join_outer(s_right, args.validation)?;
        let mut opt_join_tuples = &*opt_join_tuples;

        if let Some((offset, len)) = args.slice {
            opt_join_tuples = slice_slice(opt_join_tuples, offset, len);
        }

        // Take the left and right dataframes by join tuples
        let (mut df_left, df_right) = POOL.join(
            || unsafe {
                ca_self.drop(s_left.name()).unwrap().take_unchecked(
                    &opt_join_tuples
                        .iter()
                        .copied()
                        .map(|(left, _right)| left)
                        .collect_ca("outer-join-left-indices"),
                )
            },
            || unsafe {
                other.drop(s_right.name()).unwrap().take_unchecked(
                    &opt_join_tuples
                        .iter()
                        .copied()
                        .map(|(_left, right)| right)
                        .collect_ca("outer-join-right-indices"),
                )
            },
        );

        let s = unsafe {
            zip_outer_join_column(
                &s_left.to_physical_repr(),
                &s_right.to_physical_repr(),
                opt_join_tuples,
            )
            .with_name(s_left.name())
        };
        let s = match s_left.dtype() {
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical(_) => {
                let ca_left = s_left.categorical().unwrap();
                let new_rev_map = ca_left._merge_categorical_map(s_right.categorical().unwrap())?;
                let logical = s.u32().unwrap().clone();
                // safety:
                // categorical maps are merged
                unsafe {
                    CategoricalChunked::from_cats_and_rev_map_unchecked(logical, new_rev_map)
                        .into_series()
                }
            },
            dt @ DataType::Datetime(_, _)
            | dt @ DataType::Time
            | dt @ DataType::Date
            | dt @ DataType::Duration(_) => s.cast(dt).unwrap(),
            _ => s,
        };

        unsafe { df_left.get_columns_mut().insert(join_column_index, s) };
        _finish_join(df_left, df_right, args.suffix.as_deref())
    }
}

impl JoinDispatch for DataFrame {}
