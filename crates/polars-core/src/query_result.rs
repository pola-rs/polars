use crate::frame::DataFrame;

pub enum QueryResult {
    Single(DataFrame),
    /// Collected to multiple in-memory sinks
    Multiple(Vec<DataFrame>),
}

impl QueryResult {
    pub fn unwrap_single(self) -> DataFrame {
        use QueryResult::*;
        match self {
            Single(df) => df,
            Multiple(_) => panic!(),
        }
    }

    pub fn unwrap_multiple(self) -> Vec<DataFrame> {
        use QueryResult::*;
        match self {
            Single(_) => panic!(),
            Multiple(dfs) => dfs,
        }
    }
}
