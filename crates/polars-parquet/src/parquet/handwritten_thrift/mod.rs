//! Hand-written Thrift compact decoders, ported from apache/arrow-rs 57.0.
//!
//! Replaces the generic `polars_parquet_format::thrift::TCompactInputProtocol`
//! on the file-metadata footer decode path. Page-header and PageIndex decode
//! still go through the generic codec. Porting those is a follow-up task.

// `#[macro_use]` brings the `thrift_struct!` / `thrift_union!` / `thrift_enum!`
// / `write_thrift_field!` / `__thrift_*` macros into scope for sibling modules
// *before* they compile. Needed because `parquet_thrift.rs` uses
// `write_thrift_field!`; without this ordering macro resolution fails at
// `write_thrift_field!(i8, FieldType::Byte);` etc.
#[macro_use]
mod parquet_macros;
mod file_metadata_thrift;
mod parquet_thrift;

pub(crate) use file_metadata_thrift::decode_file_metadata;
