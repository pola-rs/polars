//! Read and write from and to Apache Avro

mod read;
#[cfg(feature = "avro")]
mod read_async;
mod write;
#[cfg(feature = "avro")]
mod write_async;
