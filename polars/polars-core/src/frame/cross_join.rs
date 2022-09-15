use crate::prelude::*;
use crate::series::IsSorted;
use crate::utils::{concat_df_unchecked, slice_offsets, CustomIterTools, NoNull};
use crate::POOL;

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
        }
    }
}

fn take_left(total_rows: IdxSize, n_rows_right: IdxSize, slice: Option<(i64, usize)>) -> IdxCa {
    fn inner(offset: IdxSize, total_rows: IdxSize, n_rows_right: IdxSize) -> IdxCa {
        let mut take: NoNull<IdxCa> = (offset..total_rows)
            .map(|i| i / n_rows_right)
            .collect_trusted();
        take.set_sorted2(IsSorted::Ascending);
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

impl DataFrame {
    /// Creates the cartesian product from both frames, preserves the order of the left keys.
    pub(crate) fn cross_join(
        &self,
        other: &DataFrame,
        suffix: Option<String>,
        slice: Option<(i64, usize)>,
    ) -> PolarsResult<DataFrame> {
        let n_rows_left = self.height() as IdxSize;
        let n_rows_right = other.height() as IdxSize;
        let total_rows = n_rows_right * n_rows_left;

        // the left side has the Nth row combined with every row from right.
        // So let's say we have the following no. of rows
        // left: 3
        // right: 4
        //
        // left take idx:   000011112222
        // right take idx:  012301230123

        let create_left_df = || {
            // Safety:
            // take left is in bounds
            unsafe { self.take_unchecked(&take_left(total_rows, n_rows_right, slice)) }
        };

        let create_right_df = || {
            // concatenation of dataframes is very expensive if we need to make the series mutable
            // many times, these are atomic operations
            // so we choose a different strategy at > 100 rows (arbitrarily small number)
            if n_rows_left > 100 || slice.is_some() {
                // Safety:
                // take right is in bounds
                unsafe { other.take_unchecked(&take_right(total_rows, n_rows_right, slice)) }
            } else {
                let iter = (0..n_rows_left).map(|_| other);
                concat_df_unchecked(iter)
            }
        };

        let (l_df, r_df) = POOL.install(|| rayon::join(create_left_df, create_right_df));

        self.finish_join(l_df, r_df, suffix)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::df;

    #[test]
    fn test_cross_join() -> PolarsResult<()> {
        let df_a = df![
            "a" => [1, 2],
            "b" => ["foo", "spam"]
        ]?;

        let df_b = df![
            "b" => ["a", "b", "c"]
        ]?;

        let out = df_a.cross_join(&df_b, None, None)?;
        let expected = df![
            "a" => [1, 1, 1, 2, 2, 2],
            "b" => ["foo", "foo", "foo", "spam", "spam", "spam"],
            "b_right" => ["a", "b", "c", "a", "b", "c"]
        ]?;

        assert!(out.frame_equal(&expected));

        Ok(())
    }
}
