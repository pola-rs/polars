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
    /// Automatically determine over which unit to parallelize
    /// This will choose the most occurring unit.
    #[default]
    Auto,
}
