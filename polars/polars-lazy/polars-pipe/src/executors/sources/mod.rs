#[cfg(feature = "csv-file")]
mod csv;
#[cfg(feature = "parquet")]
mod parquet;
mod union;

#[cfg(feature = "csv-file")]
pub(crate) use csv::CsvSource;
#[cfg(feature = "parquet")]
pub(crate) use parquet::*;
pub(crate) use union::*;

use super::*;
