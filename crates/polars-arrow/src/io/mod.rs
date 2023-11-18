#![forbid(unsafe_code)]
//! Contains modules to interface with other formats such as [`csv`],
//! [`parquet`], [`json`], [`ipc`], [`mod@print`] and [`avro`].

#[cfg(feature = "io_ipc")]
#[cfg_attr(docsrs, doc(cfg(feature = "io_ipc")))]
pub mod ipc;

#[cfg(feature = "io_flight")]
#[cfg_attr(docsrs, doc(cfg(feature = "io_flight")))]
pub mod flight;

#[cfg(feature = "io_avro")]
#[cfg_attr(docsrs, doc(cfg(feature = "io_avro")))]
pub mod avro;

pub mod iterator;
