use crate::prelude::*;

impl Series {
    /// Create a [`DataFrame`] with the unique `values` of this [`Series`] and a column `"counts"`
    /// with dtype [`IdxType`]
    pub fn value_counts(&self) -> Result<DataFrame> {
        let groups = self.group_tuples(true, false);
        let values = self.agg_first(&groups);
        let counts = groups.group_lengths("counts");
        let cols = vec![values.into_series(), counts.into_series()];
        let df = DataFrame::new_no_checks(cols);
        df.sort(&["counts"], true)
    }
}
