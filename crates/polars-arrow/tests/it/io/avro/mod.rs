//! Read and write from and to Apache Avro

mod read;
#[cfg(feature = "io_avro_async")]
mod read_async;
mod write;
#[cfg(feature = "io_avro_async")]
mod write_async;
