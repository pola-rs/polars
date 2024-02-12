#[cfg(feature = "io_print")]
mod print;

#[cfg(feature = "io_json")]
mod json;

#[cfg(feature = "io_json")]
mod ndjson;

#[cfg(feature = "io_json_integration")]
mod ipc;

#[cfg(feature = "io_json_integration")]
mod ipc2;

#[cfg(feature = "io_parquet")]
mod parquet;

#[cfg(feature = "io_avro")]
mod avro;

#[cfg(feature = "io_orc")]
mod orc;

#[cfg(any(
    feature = "io_csv_read",
    feature = "io_csv_write",
    feature = "io_csv_read_async"
))]
mod csv;

#[cfg(feature = "io_flight")]
mod flight;
