//! Misc utilities used in different places in the crate.

#[cfg(any(
    feature = "compute",
    feature = "io_csv_write",
    feature = "io_csv_read",
    feature = "io_json",
    feature = "io_json_write",
    feature = "compute_cast"
))]
mod lexical;
#[cfg(any(
    feature = "compute",
    feature = "io_csv_write",
    feature = "io_csv_read",
    feature = "io_json",
    feature = "io_json_write",
    feature = "compute_cast"
))]
pub use lexical::*;

#[cfg(feature = "benchmarks")]
#[cfg_attr(docsrs, doc(cfg(feature = "benchmarks")))]
pub mod bench_util;
