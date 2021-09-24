use crate::prelude::*;
use crate::utils::{concat_df, CustomIterTools, NoNull};
use crate::POOL;

impl DataFrame {
    /// Creates the cartesian product from both frames, preserves the order of the left keys.
    pub fn cross_join(&self, other: &DataFrame) -> Result<DataFrame> {
        let n_rows_left = self.height() as u32;
        let n_rows_right = other.height() as u32;
        let total_rows = n_rows_right * n_rows_left;

        // the left side has the Nth row combined with every row from right.
        // So let's say we have the following no. of rows
        // left: 3
        // right: .as_slice()4
        //
        // left take idx:   000011112222
        // right take idx:  012301230123

        let create_left_df = || {
            let take_left: NoNull<UInt32Chunked> =
                (0..total_rows).map(|i| i / n_rows_right).collect_trusted();
            // Safety:
            // take left is in bounds
            unsafe { self.take_unchecked(&take_left.into_inner()) }
        };

        let create_right_df = || {
            let iter = (0..n_rows_left).map(|_| other);
            concat_df(iter).unwrap()
        };

        let (l_df, r_df) = POOL.install(|| rayon::join(create_left_df, create_right_df));

        self.finish_join(l_df, r_df, None)
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

        let out = df_a.cross_join(&df_b)?;
        let expected = df![
            "a" => [1, 1, 1, 2, 2, 2],
            "b" => ["foo", "foo", "foo", "spam", "spam", "spam"],
            "b_right" => ["a", "b", "c", "a", "b", "c"]
        ]?;

        assert!(out.frame_equal(&expected));

        Ok(())
    }
}
