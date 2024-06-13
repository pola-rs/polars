use smartstring::alias::String as SmartString;

use super::*;
use crate::prelude::sort::arg_bottom_k::_arg_bottom_k;

impl DataFrame {
    /// get a DataFrame with the top k rows.
    ///
    /// # Example
    /// ```
    /// # use polars_core::prelude::*;
    ///
    /// let df = df!(
    ///     "foo" => &[1,2,3],
    ///     "bar" => &["c","b","a"]
    /// )?;
    ///
    /// let df_top1 = df.top_k(1, true, ["foo"])?;
    /// assert_eq!(df_top1, df!(
    ///     "foo" => &[3],
    ///     "bar" => &["a"]
    /// )?);
    ///
    /// let df_top2 = df.top_k(2, false, ["bar"])?;
    /// assert_eq!(df_top2, df!(
    ///     "foo" => &[3, 2],
    ///     "bar" => &["a", "b"]
    /// )?);
    ///
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn top_k(
        &self,
        k: usize,
        by_column: impl IntoVec<SmartString>,
        sort_options: SortMultipleOptions,
    ) -> PolarsResult<DataFrame> {
        let by_column = self.select_series(by_column)?;
        self.bottom_k_impl(k, by_column, sort_options.with_order_reversed())
    }

    pub(crate) fn bottom_k_impl(
        &self,
        k: usize,
        by_column: Vec<Series>,
        mut sort_options: SortMultipleOptions,
    ) -> PolarsResult<DataFrame> {
        let first_descending = sort_options.descending[0];
        let first_by_column = by_column[0].name().to_string();

        let idx = _arg_bottom_k(k, &by_column, &mut sort_options)?;

        let mut df = unsafe { self.take_unchecked(&idx.into_inner()) };

        // Mark the first sort column as sorted
        // if the column did not exists it is ok, because we sorted by an expression
        // not present in the dataframe
        let _ = df.apply(&first_by_column, |s| {
            let mut s = s.clone();
            if first_descending {
                s.set_sorted_flag(IsSorted::Descending)
            } else {
                s.set_sorted_flag(IsSorted::Ascending)
            }
            s
        });
        Ok(df)
    }
}
