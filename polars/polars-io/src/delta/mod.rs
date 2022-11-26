//! Reading a delta dataset.
//! 
//! A delta dataset is made out of delta metadata and parquet files.


mod read;
mod write;

pub use read::*;
pub use write::*;