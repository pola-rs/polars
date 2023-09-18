#![forbid(unsafe_code)]
//! Contains modules to interface with other formats such as [`csv`],
//! [`parquet`], [`json`], [`ipc`], [`mod@print`] and [`avro`].

#[cfg(feature = "io_odbc")]
pub mod odbc;

#[cfg(feature = "io_orc")]
#[cfg_attr(docsrs, doc(cfg(feature = "io_orc")))]
pub mod orc;

#[cfg(any(feature = "io_json_read", feature = "io_json_write"))]
#[cfg_attr(docsrs, doc(cfg(feature = "io_json")))]
pub mod json;
#[cfg(any(feature = "io_json_read", feature = "io_json_write"))]
#[cfg_attr(docsrs, doc(cfg(feature = "io_json")))]
pub mod ndjson;

#[cfg(feature = "io_ipc")]
#[cfg_attr(docsrs, doc(cfg(feature = "io_ipc")))]
pub mod ipc;

#[cfg(feature = "io_flight")]
#[cfg_attr(docsrs, doc(cfg(feature = "io_flight")))]
pub mod flight;

#[cfg(feature = "io_json_integration")]
#[cfg_attr(docsrs, doc(cfg(feature = "io_json_integration")))]
pub mod json_integration;

#[cfg(feature = "io_parquet")]
#[cfg_attr(docsrs, doc(cfg(feature = "io_parquet")))]
pub mod parquet;

#[cfg(feature = "io_avro")]
#[cfg_attr(docsrs, doc(cfg(feature = "io_avro")))]
pub mod avro;

#[cfg(feature = "io_print")]
#[cfg_attr(docsrs, doc(cfg(feature = "io_print")))]
pub mod print;

#[cfg(any(
    feature = "io_csv_write",
    feature = "io_avro",
    feature = "io_json_write",
    feature = "io_json_read"
))]
mod iterator;
