mod csv;

#[cfg(feature = "json")]
mod json;

#[cfg(feature = "parquet")]
mod parquet;

#[cfg(feature = "avro")]
mod avro;

#[cfg(feature = "ipc")]
mod ipc;
#[cfg(feature = "ipc_streaming")]
mod ipc_stream;

use polars::prelude::*;

pub(crate) fn create_df() -> DataFrame {
    let s0 = Series::new("days", [0, 1, 2, 3, 4].as_ref());
    let s1 = Series::new("temp", [22.1, 19.9, 7., 2., 3.].as_ref());
    DataFrame::new(vec![s0, s1]).unwrap()
}
