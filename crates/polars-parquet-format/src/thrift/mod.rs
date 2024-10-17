pub mod protocol;
pub mod varint;

mod errors;
pub use errors::*;

/// Result type returned by all runtime library functions.
///
/// As is convention this is a typedef of `std::result::Result`
/// with `E` defined as the `thrift::Error` type.
pub type Result<T> = std::result::Result<T, self::Error>;
