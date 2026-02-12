use polars_core::frame::DataFrame;

pub type SinkMorselPermit = tokio::sync::OwnedSemaphorePermit;

/// In-flight morsel in the IO sink. Holds a permit against a semaphore that restricts
/// the total number of sink morsels in memory.
pub struct SinkMorsel {
    df: DataFrame,
    /// Should only be dropped once the data associated with this morsel has been dropped from memory.
    permit: SinkMorselPermit,
}

impl SinkMorsel {
    pub fn new(df: DataFrame, permit: SinkMorselPermit) -> Self {
        Self { df, permit }
    }

    pub fn into_inner(self) -> (DataFrame, SinkMorselPermit) {
        (self.df, self.permit)
    }

    pub fn df(&self) -> &DataFrame {
        &self.df
    }

    pub fn df_mut(&mut self) -> &mut DataFrame {
        &mut self.df
    }
}
