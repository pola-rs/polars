use polars_core::frame::DataFrame;

/// Wrapper that exposes `split_off_front()` that avoids allocating for the remainder.
pub struct DfWithOffset {
    df: DataFrame,
    offset: usize,
}

impl DfWithOffset {
    pub fn new(df: DataFrame) -> Self {
        Self { df, offset: 0 }
    }

    /// Note: This performs a linear scan across chunks on each call.
    pub fn split_off_front(&mut self, n_rows: usize) -> DataFrame {
        let ret = self.df.slice(self.offset as i64, n_rows);
        self.offset += n_rows;
        assert!(self.offset <= self.df.height());
        ret
    }

    pub fn into_df(self) -> DataFrame {
        self.df
            .slice(self.offset as i64, self.df.height() - self.offset)
    }

    pub fn height(&self) -> usize {
        self.df.height() - self.offset
    }
}
