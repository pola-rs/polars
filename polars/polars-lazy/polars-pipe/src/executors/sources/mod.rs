#[cfg(feature = "csv-file")]
mod csv;
mod frame;
#[cfg(feature = "parquet")]
mod parquet;
mod union;

#[cfg(feature = "csv-file")]
pub(crate) use csv::CsvSource;
pub(crate) use frame::*;
#[cfg(feature = "parquet")]
pub(crate) use parquet::*;
pub(crate) use union::*;

use super::*;
