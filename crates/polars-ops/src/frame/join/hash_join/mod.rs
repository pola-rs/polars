#![allow(unsafe_op_in_unsafe_fn)]
pub(super) mod single_keys;
mod single_keys_dispatch;
mod single_keys_inner;
mod single_keys_left;
mod single_keys_outer;
#[cfg(feature = "semi_anti_join")]
mod single_keys_semi_anti;
pub(super) mod sort_merge;
use arrow::array::ArrayRef;
use polars_core::runtime::RAYON;
use polars_core::utils::_set_partition_size;
use polars_utils::index::ChunkId;
use polars_utils::unique_column_name;
pub(super) use single_keys::*;
pub use single_keys_dispatch::SeriesJoin;
#[cfg(feature = "asof_join")]
pub(super) use single_keys_dispatch::prepare_binary;
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
        nulls_equal: bool,
    ) -> PolarsResult<DataFrame> {
        let ca_self = self.to_df();

        let idx = s_left.hash_join_semi_anti(s_right, anti, nulls_equal)?;
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
        args.validate_indicator(df_self.schema(), other.schema())?;

        // Get the indexes of the joined relations
        let (mut join_idx_l, mut join_idx_r) =
            s_left.hash_join_outer(s_right, args.validation, args.nulls_equal)?;

        try_raise_keyboard_interrupt();
        if let Some((offset, len)) = args.slice {
            let (offset, len) = slice_offsets(offset, len, join_idx_l.len());
            join_idx_l.slice(offset, len);
            join_idx_r.slice(offset, len);
        }
        let idx_ca_l = IdxCa::with_chunk("a".into(), join_idx_l);
        let idx_ca_r = IdxCa::with_chunk("b".into(), join_idx_r);

        // Helper: derive indicator values from two index ChunkedArrays.
        // IdxCa::iter() yields Option<IdxSize>; None means "no match on that side".
        //   left None  -> row came from right only
        //   right None -> row came from left only
        //   both Some  -> row matched on both sides
        let build_indicator = |name: &PlSmallStr, l: &IdxCa, r: &IdxCa| -> Series {
            let values: Vec<&str> = l
                .iter()
                .zip(r.iter())
                .map(|(lv, rv)| match (lv, rv) {
                    (None, _) => "right_only",
                    (_, None) => "left_only",
                    _ => "both",
                })
                .collect();
            Series::new(name.clone(), values)
        };

        let (df_left, df_right, indicator_series) =
            if args.maintain_order != MaintainOrderJoin::None {
                let mut df = unsafe {
                    DataFrame::new_unchecked_infer_height(vec![
                        idx_ca_l.into_series().into(),
                        idx_ca_r.into_series().into(),
                    ])
                };

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

                // Build indicator from the sorted index arrays before RAYON moves them.
                let ind = args
                    .indicator
                    .as_ref()
                    .map(|name| build_indicator(name, join_tuples_left, join_tuples_right));

                let (dfl, dfr) = RAYON.join(
                    || unsafe { df_self.take_unchecked(join_tuples_left) },
                    || unsafe { other.take_unchecked(join_tuples_right) },
                );
                (dfl, dfr, ind)
            } else {
                // Build indicator before RAYON borrows the arrays.
                let ind = args
                    .indicator
                    .as_ref()
                    .map(|name| build_indicator(name, &idx_ca_l, &idx_ca_r));

                let (dfl, dfr) = RAYON.join(
                    || unsafe { df_self.take_unchecked(&idx_ca_l) },
                    || unsafe { other.take_unchecked(&idx_ca_r) },
                );
                (dfl, dfr, ind)
            };

        let coalesce = args.coalesce.coalesce(&JoinType::Full);
        let mut out = if coalesce {
            let tmp_right_name = unique_column_name();
            let mut df_right = df_right;
            df_right.rename(s_right.name().as_str(), tmp_right_name.clone())?;
            let out = _finish_join(df_left, df_right, args.suffix.clone())?;
            Ok(_coalesce_full_join(
                out,
                &[s_left.name().clone()],
                &[tmp_right_name],
                args.suffix,
                df_self,
            ))
        } else {
            _finish_join(df_left, df_right, args.suffix.clone())
        }?;

        if let Some(ind) = indicator_series {
            out.with_column(ind.into())?;
        }

        Ok(out)
    }
}

impl JoinDispatch for DataFrame {}
