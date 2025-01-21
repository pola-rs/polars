pub(super) mod single_keys;
mod single_keys_dispatch;
mod single_keys_inner;
mod single_keys_left;
mod single_keys_outer;
#[cfg(feature = "semi_anti_join")]
mod single_keys_semi_anti;
pub(super) mod sort_merge;
use arrow::array::ArrayRef;
use polars_core::utils::_set_partition_size;
use polars_core::POOL;
use polars_utils::index::ChunkId;
pub(super) use single_keys::*;
#[cfg(feature = "asof_join")]
pub(super) use single_keys_dispatch::prepare_binary;
pub use single_keys_dispatch::SeriesJoin;
use single_keys_inner::*;
use single_keys_left::*;
use single_keys_outer::*;
#[cfg(feature = "semi_anti_join")]
use single_keys_semi_anti::*;
pub(crate) use sort_merge::*;

pub use super::*;
#[cfg(feature = "chunked_ids")]
use crate::chunked_array::gather::chunked::TakeChunkedHorPar;

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
        if $self.len() > $other.len() {
            ($self, $other, false)
        } else {
            ($other, $self, true)
        }
    }};
}

#[cfg(feature = "performant")]
use arrow::legacy::conversion::primitive_to_vec;
pub(super) use det_hash_prone_order;

pub trait JoinDispatch: IntoDf {
    /// # Safety
    /// Join tuples must be in bounds
    #[cfg(feature = "chunked_ids")]
    unsafe fn create_left_df_chunked(
        &self,
        chunk_ids: &[ChunkId],
        left_join: bool,
        was_sliced: bool,
    ) -> DataFrame {
        let df_self = self.to_df();

        let left_join_no_duplicate_matches =
            left_join && !was_sliced && chunk_ids.len() == df_self.height();

        if left_join_no_duplicate_matches {
            df_self.clone()
        } else {
            // left join keys are in ascending order
            let sorted = if left_join {
                IsSorted::Ascending
            } else {
                IsSorted::Not
            };
            df_self._take_chunked_unchecked_hor_par(chunk_ids, sorted)
        }
    }

    /// # Safety
    /// Join tuples must be in bounds
    unsafe fn _create_left_df_from_slice(
        &self,
        join_tuples: &[IdxSize],
        left_join: bool,
        was_sliced: bool,
        sorted_tuple_idx: bool,
    ) -> DataFrame {
        let df_self = self.to_df();

        let left_join_no_duplicate_matches =
            sorted_tuple_idx && left_join && !was_sliced && join_tuples.len() == df_self.height();

        if left_join_no_duplicate_matches {
            df_self.clone()
        } else {
            let sorted = if sorted_tuple_idx {
                IsSorted::Ascending
            } else {
                IsSorted::Not
            };

            df_self._take_unchecked_slice_sorted(join_tuples, true, sorted)
        }
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
        join_nulls: bool,
    ) -> PolarsResult<DataFrame> {
        let ca_self = self.to_df();
        #[cfg(feature = "dtype-categorical")]
        _check_categorical_src(s_left.dtype(), s_right.dtype())?;

        let idx = s_left.hash_join_semi_anti(s_right, anti, join_nulls)?;
        // SAFETY:
        // indices are in bounds
        Ok(unsafe { ca_self._finish_anti_semi_join(&idx, slice) })
    }
    fn _full_join_from_series(
        &self,
        other: &DataFrame,
        s_left: &Series,
        s_right: &Series,
        args: JoinArgs,
    ) -> PolarsResult<DataFrame> {
        let df_self = self.to_df();
        #[cfg(feature = "dtype-categorical")]
        _check_categorical_src(s_left.dtype(), s_right.dtype())?;

        // Get the indexes of the joined relations
        let (mut join_idx_l, mut join_idx_r) =
            s_left.hash_join_outer(s_right, args.validation, args.join_nulls)?;

        check_signals()?;
        if let Some((offset, len)) = args.slice {
            let (offset, len) = slice_offsets(offset, len, join_idx_l.len());
            join_idx_l.slice(offset, len);
            join_idx_r.slice(offset, len);
        }
        let idx_ca_l = IdxCa::with_chunk("a".into(), join_idx_l);
        let idx_ca_r = IdxCa::with_chunk("b".into(), join_idx_r);

        let (df_left, df_right) = if args.maintain_order != MaintainOrderJoin::None {
            let mut df = DataFrame::new(vec![
                idx_ca_l.into_series().into(),
                idx_ca_r.into_series().into(),
            ])?;

            let options = SortMultipleOptions::new()
                .with_order_descending(false)
                .with_maintain_order(true)
                .with_nulls_last(true);

            let columns = match args.maintain_order {
                MaintainOrderJoin::Left => vec!["a"],
                MaintainOrderJoin::LeftRight => vec!["a", "b"],
                MaintainOrderJoin::Right => vec!["b"],
                MaintainOrderJoin::RightLeft => vec!["b", "a"],
                _ => unreachable!(),
            };

            df.sort_in_place(columns, options)?;

            let join_tuples_left = df.column("a").unwrap().idx().unwrap();
            let join_tuples_right = df.column("b").unwrap().idx().unwrap();
            POOL.join(
                || unsafe { df_self.take_unchecked(join_tuples_left) },
                || unsafe { other.take_unchecked(join_tuples_right) },
            )
        } else {
            POOL.join(
                || unsafe { df_self.take_unchecked(&idx_ca_l) },
                || unsafe { other.take_unchecked(&idx_ca_r) },
            )
        };

        let coalesce = args.coalesce.coalesce(&JoinType::Full);
        let out = _finish_join(df_left, df_right, args.suffix.clone());
        if coalesce {
            Ok(_coalesce_full_join(
                out?,
                &[s_left.name().clone()],
                &[s_right.name().clone()],
                args.suffix.clone(),
                df_self,
            ))
        } else {
            out
        }
    }
}

impl JoinDispatch for DataFrame {}
