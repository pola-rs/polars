use crate::prelude::*;
use crate::series::IsSorted;
use crate::utils::{concat_df_unchecked, CustomIterTools, NoNull};
use crate::POOL;

impl DataFrame {
    /// Creates the cartesian product from both frames, preserves the order of the left keys.
    pub fn cross_join(&self, other: &DataFrame, suffix: Option<String>) -> Result<DataFrame> {
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
            let mut take_left: NoNull<IdxCa> =
                (0..total_rows).map(|i| i / n_rows_right).collect_trusted();
            take_left.set_sorted2(IsSorted::Ascending);
            // Safety:
            // take left is in bounds
            unsafe { self.take_unchecked(&take_left.into_inner()) }
        };

        let create_right_df = || {
            // concatenation of dataframes is very expensive if we need to make the series mutable
            // many times, these are atomic operations
            // so we choose a different strategy at > 100 rows (arbitrarily small number)
            if n_rows_left > 100 {
                let take_right: NoNull<IdxCa> =
                    (0..total_rows).map(|i| i % n_rows_right).collect_trusted();
                // Safety:
                // take right is in bounds
                unsafe { other.take_unchecked(&take_right.into_inner()) }
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
    fn test_cross_join() -> Result<()> {
        let df_a = df![
            "a" => [1, 2],
            "b" => ["foo", "spam"]
        ]?;

        let df_b = df![
            "b" => ["a", "b", "c"]
        ]?;

        let out = df_a.cross_join(&df_b, None)?;
        let expected = df![
            "a" => [1, 1, 1, 2, 2, 2],
            "b" => ["foo", "foo", "foo", "spam", "spam", "spam"],
            "b_right" => ["a", "b", "c", "a", "b", "c"]
        ]?;

        assert!(out.frame_equal(&expected));

        Ok(())
    }
}
