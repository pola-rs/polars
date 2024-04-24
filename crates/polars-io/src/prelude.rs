#[cfg(test)]
use polars_core::prelude::*;

pub use crate::cloud;
#[cfg(feature = "csv")]
pub use crate::csv::{read::*, write::*};
#[cfg(any(feature = "ipc", feature = "ipc_streaming"))]
pub use crate::ipc::*;
#[cfg(feature = "json")]
pub use crate::json::*;
#[cfg(feature = "json")]
pub use crate::ndjson::core::*;
#[cfg(feature = "parquet")]
pub use crate::parquet::{metadata::*, read::*, write::*};
pub use crate::shared::{SerReader, SerWriter};
pub use crate::utils::*;

#[cfg(test)]
pub(crate) fn create_df() -> DataFrame {
    let s0 = Series::new("days", [0, 1, 2, 3, 4].as_ref());
    let s1 = Series::new("temp", [22.1, 19.9, 7., 2., 3.].as_ref());
    DataFrame::new(vec![s0, s1]).unwrap()
}
