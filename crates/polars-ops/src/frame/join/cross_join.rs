use polars_core::utils::{
    _set_partition_size, CustomIterTools, NoNull, accumulate_dataframes_vertical_unchecked,
    concat_df_unchecked, split,
};
use polars_utils::pl_str::PlSmallStr;

use super::*;

fn slice_take(
    total_rows: IdxSize,
    n_rows_right: IdxSize,
    slice: Option<(i64, usize)>,
    inner: fn(IdxSize, IdxSize, IdxSize) -> IdxCa,
) -> IdxCa {
    match slice {
        None => inner(0, total_rows, n_rows_right),
        Some((offset, len)) => {
            let (offset, len) = slice_offsets(offset, len, total_rows as usize);
            inner(offset as IdxSize, (len + offset) as IdxSize, n_rows_right)
        },
    }
}

fn take_left(total_rows: IdxSize, n_rows_right: IdxSize, slice: Option<(i64, usize)>) -> IdxCa {
    fn inner(offset: IdxSize, total_rows: IdxSize, n_rows_right: IdxSize) -> IdxCa {
        let mut take: NoNull<IdxCa> = (offset..total_rows)
            .map(|i| i / n_rows_right)
            .collect_trusted();
        take.set_sorted_flag(IsSorted::Ascending);
        take.into_inner()
    }
    slice_take(total_rows, n_rows_right, slice, inner)
}

fn take_right(total_rows: IdxSize, n_rows_right: IdxSize, slice: Option<(i64, usize)>) -> IdxCa {
    fn inner(offset: IdxSize, total_rows: IdxSize, n_rows_right: IdxSize) -> IdxCa {
        let take: NoNull<IdxCa> = (offset..total_rows)
            .map(|i| i % n_rows_right)
            .collect_trusted();
        take.into_inner()
    }
    slice_take(total_rows, n_rows_right, slice, inner)
}

pub trait CrossJoin: IntoDf {
    /// Creates the Cartesian product from both frames, preserves the order of the left keys.
    fn cross_join(
        &self,
        other: &DataFrame,
        suffix: Option<PlSmallStr>,
        slice: Option<(i64, usize)>,
        maintain_order: MaintainOrderJoin,
    ) -> PolarsResult<DataFrame> {
        let (l_df, r_df) = cross_join_dfs(self.to_df(), other, slice, true, maintain_order)?;

        _finish_join(l_df, r_df, suffix)
    }
}

impl CrossJoin for DataFrame {}

fn cross_join_dfs<'a>(
    mut df_self: &'a DataFrame,
    mut other: &'a DataFrame,
    slice: Option<(i64, usize)>,
    parallel: bool,
    maintain_order: MaintainOrderJoin,
) -> PolarsResult<(DataFrame, DataFrame)> {
    if df_self.height() == 0 || other.height() == 0 {
        return Ok((df_self.clear(), other.clear()));
    }

    let left_is_primary = match maintain_order {
        MaintainOrderJoin::None => true,
        MaintainOrderJoin::Left | MaintainOrderJoin::LeftRight => true,
        MaintainOrderJoin::Right | MaintainOrderJoin::RightLeft => false,
    };

    if !left_is_primary {
        core::mem::swap(&mut df_self, &mut other);
    }

    let n_rows_left = df_self.height() as IdxSize;
    let n_rows_right = other.height() as IdxSize;
    let Some(total_rows) = n_rows_left.checked_mul(n_rows_right) else {
        polars_bail!(
            ComputeError: "cross joins would produce more rows than fits into 2^32; \
            consider compiling with polars-big-idx feature, or set 'streaming'"
        );
    };

    // the left side has the Nth row combined with every row from right.
    // So let's say we have the following no. of rows
    // left: 3
    // right: 4
    //
    // left take idx:   000011112222
    // right take idx:  012301230123

    let create_left_df = || {
        // SAFETY:
        // take left is in bounds
        unsafe {
            df_self.take_unchecked_impl(&take_left(total_rows, n_rows_right, slice), parallel)
        }
    };

    let create_right_df = || {
        // concatenation of dataframes is very expensive if we need to make the series mutable
        // many times, these are atomic operations
        // so we choose a different strategy at > 100 rows (arbitrarily small number)
        if n_rows_left > 100 || slice.is_some() {
            // SAFETY:
            // take right is in bounds
            unsafe {
                other.take_unchecked_impl(&take_right(total_rows, n_rows_right, slice), parallel)
            }
        } else {
            let iter = (0..n_rows_left).map(|_| other);
            concat_df_unchecked(iter)
        }
    };
    let (l_df, r_df) = if parallel {
        try_raise_keyboard_interrupt();
        POOL.install(|| rayon::join(create_left_df, create_right_df))
    } else {
        (create_left_df(), create_right_df())
    };
    if left_is_primary {
        Ok((l_df, r_df))
    } else {
        Ok((r_df, l_df))
    }
}

pub(super) fn fused_cross_filter(
    left: &DataFrame,
    right: &DataFrame,
    suffix: Option<PlSmallStr>,
    cross_join_options: &CrossJoinOptions,
) -> PolarsResult<DataFrame> {
    let unfiltered_size = (left.height() as u64).saturating_mul(right.height() as u64);
    let chunk_size = (unfiltered_size / _set_partition_size() as u64).clamp(1, 100_000);
    let num_chunks = (unfiltered_size / chunk_size).max(1) as usize;

    let left_is_primary = match cross_join_options.maintain_order {
        MaintainOrderJoin::None => true,
        MaintainOrderJoin::Left | MaintainOrderJoin::LeftRight => true,
        MaintainOrderJoin::Right | MaintainOrderJoin::RightLeft => false,
    };

    let split_chunks;
    let cartesian_prod = if left_is_primary {
        split_chunks = split(left, num_chunks);
        split_chunks.iter().map(|l| (l, right)).collect::<Vec<_>>()
    } else {
        split_chunks = split(right, num_chunks);
        split_chunks.iter().map(|r| (left, r)).collect::<Vec<_>>()
    };

    let names = _finish_join(left.clear(), right.clear(), suffix)?;
    let rename_names = names.get_column_names();
    let rename_names = &rename_names[left.width()..];

    let dfs = POOL
        .install(|| {
            cartesian_prod.par_iter().map(|(left, right)| {
                let (mut left, right) =
                    cross_join_dfs(left, right, None, false, cross_join_options.maintain_order)?;
                let mut right_columns = right.take_columns();

                for (c, name) in right_columns.iter_mut().zip(rename_names) {
                    c.rename((*name).clone());
                }

                unsafe { left.hstack_mut_unchecked(&right_columns) };

                cross_join_options.predicate.apply(left)
            })
        })
        .collect::<PolarsResult<Vec<_>>>()?;

    Ok(accumulate_dataframes_vertical_unchecked(dfs))
}
