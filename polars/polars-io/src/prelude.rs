pub use crate::{SerReader, SerWriter};

#[cfg(feature = "csv-file")]
pub use crate::csv::*;
#[cfg(feature = "ipc")]
pub use crate::ipc::*;
#[cfg(feature = "json")]
pub use crate::json::*;
#[cfg(feature = "parquet")]
pub use crate::parquet::*;

#[cfg(feature = "private")]
pub use crate::utils::*;

#[cfg(test)]
use polars_core::prelude::*;
#[cfg(test)]
pub(crate) fn create_df() -> DataFrame {
    let s0 = Series::new("days", [0, 1, 2, 3, 4].as_ref());
    let s1 = Series::new("temp", [22.1, 19.9, 7., 2., 3.].as_ref());
    DataFrame::new(vec![s0, s1]).unwrap()
}
