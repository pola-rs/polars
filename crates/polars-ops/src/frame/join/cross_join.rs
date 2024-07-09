use polars_core::utils::{concat_df_unchecked, CustomIterTools, NoNull};
use smartstring::alias::String as SmartString;

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
    fn cross_join_dfs(
        &self,
        other: &DataFrame,
        slice: Option<(i64, usize)>,
        parallel: bool,
    ) -> PolarsResult<(DataFrame, DataFrame)> {
        let df_self = self.to_df();
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
            unsafe { df_self.take_unchecked(&take_left(total_rows, n_rows_right, slice)) }
        };

        let create_right_df = || {
            // concatenation of dataframes is very expensive if we need to make the series mutable
            // many times, these are atomic operations
            // so we choose a different strategy at > 100 rows (arbitrarily small number)
            if n_rows_left > 100 || slice.is_some() {
                // SAFETY:
                // take right is in bounds
                unsafe { other.take_unchecked(&take_right(total_rows, n_rows_right, slice)) }
            } else {
                let iter = (0..n_rows_left).map(|_| other);
                concat_df_unchecked(iter)
            }
        };
        let (l_df, r_df) = if parallel {
            POOL.install(|| rayon::join(create_left_df, create_right_df))
        } else {
            (create_left_df(), create_right_df())
        };
        Ok((l_df, r_df))
    }

    #[doc(hidden)]
    /// used by streaming
    fn _cross_join_with_names(
        &self,
        other: &DataFrame,
        names: &[SmartString],
    ) -> PolarsResult<DataFrame> {
        let (mut l_df, r_df) = self.cross_join_dfs(other, None, false)?;

        unsafe {
            l_df.get_columns_mut().extend_from_slice(r_df.get_columns());

            l_df.get_columns_mut()
                .iter_mut()
                .zip(names)
                .for_each(|(s, name)| {
                    if s.name() != name {
                        s.rename(name);
                    }
                });
        }
        Ok(l_df)
    }

    /// Creates the Cartesian product from both frames, preserves the order of the left keys.
    fn cross_join(
        &self,
        other: &DataFrame,
        suffix: Option<&str>,
        slice: Option<(i64, usize)>,
    ) -> PolarsResult<DataFrame> {
        let (l_df, r_df) = self.cross_join_dfs(other, slice, true)?;

        _finish_join(l_df, r_df, suffix)
    }
}

impl CrossJoin for DataFrame {}
