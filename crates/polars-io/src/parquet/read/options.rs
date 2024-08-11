#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Copy, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ParquetOptions {
    pub parallel: ParallelStrategy,
    pub low_memory: bool,
    pub use_statistics: bool,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Default, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ParallelStrategy {
    /// Don't parallelize
    None,
    /// Parallelize over the columns
    Columns,
    /// Parallelize over the row groups
    RowGroups,
    /// First evaluates the pushed-down predicates in parallel and determines a mask of which rows
    /// to read. Then, it parallelizes over both the columns and the row groups while filtering out
    /// rows that do not need to be read. This can provide significant speedups for large files
    /// (i.e. many row-groups) with a predicate that filters clustered rows or filters heavily. In
    /// other cases, this may slow down the scan compared other strategies.
    ///
    /// If no predicate is given, this falls back to back to [`ParallelStrategy::Auto`].
    Prefiltered,
    /// Automatically determine over which unit to parallelize
    /// This will choose the most occurring unit.
    #[default]
    Auto,
}
