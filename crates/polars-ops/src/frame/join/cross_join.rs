use polars_core::utils::{
    concat_df_unchecked, CustomIterTools, NoNull, _set_partition_size,
    accumulate_dataframes_vertical_unchecked, split,
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
    #[doc(hidden)]
    /// used by streaming
    fn _cross_join_with_names(
        &self,
        other: &DataFrame,
        names: &[PlSmallStr],
    ) -> PolarsResult<DataFrame> {
        let (mut l_df, r_df) = cross_join_dfs(self.to_df(), other, None, false)?;
        l_df.clear_schema();

        unsafe {
            l_df.get_columns_mut().extend_from_slice(r_df.get_columns());

            l_df.get_columns_mut()
                .iter_mut()
                .zip(names)
                .for_each(|(s, name)| {
                    if s.name() != name {
                        s.rename(name.clone());
                    }
                });
        }
        Ok(l_df)
    }

    /// Creates the Cartesian product from both frames, preserves the order of the left keys.
    fn cross_join(
        &self,
        other: &DataFrame,
        suffix: Option<PlSmallStr>,
        slice: Option<(i64, usize)>,
    ) -> PolarsResult<DataFrame> {
        let (l_df, r_df) = cross_join_dfs(self.to_df(), other, slice, true)?;

        _finish_join(l_df, r_df, suffix)
    }
}

impl CrossJoin for DataFrame {}

fn cross_join_dfs(
    df_self: &DataFrame,
    other: &DataFrame,
    slice: Option<(i64, usize)>,
    parallel: bool,
) -> PolarsResult<(DataFrame, DataFrame)> {
    let n_rows_left = df_self.height() as IdxSize;
    let n_rows_right = other.height() as IdxSize;
    let Some(total_rows) = n_rows_left.checked_mul(n_rows_right) else {
        polars_bail!(
            ComputeError: "cross joins would produce more rows than fits into 2^32; \
            consider compiling with polars-big-idx feature, or set 'streaming'"
        );
    };
    if n_rows_left == 0 || n_rows_right == 0 {
        return Ok((df_self.clear(), other.clear()));
    }

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
    Ok((l_df, r_df))
}

pub(super) fn fused_cross_filter(
    left: &DataFrame,
    right: &DataFrame,
    suffix: Option<PlSmallStr>,
    cross_join_options: &CrossJoinOptions,
) -> PolarsResult<DataFrame> {
    // Because we do a cartesian product, the number of partitions is squared.
    // We take the sqrt, but we don't expect every partition to produce results and work can be
    // imbalanced, so we multiply the number of partitions by 2;
    let n_partitions = (_set_partition_size() as f32).sqrt() as usize * 2;
    let splitted_a = split(left, n_partitions);
    let splitted_b = split(right, n_partitions);

    let cartesian_prod = splitted_a
        .iter()
        .flat_map(|l| splitted_b.iter().map(move |r| (l, r)))
        .collect::<Vec<_>>();

    let names = _finish_join(left.clear(), right.clear(), suffix)?;
    let rename_names = names.get_column_names();
    let rename_names = &rename_names[left.width()..];

    let dfs = POOL
        .install(|| {
            cartesian_prod.par_iter().map(|(left, right)| {
                let (mut left, right) = cross_join_dfs(left, right, None, false)?;
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
