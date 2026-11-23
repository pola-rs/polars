//! Native database scan plugin for Polars (`scan_database`), loaded through the generic I/O
//! plugin contracts of `polars_io_ext_ffi`:
//! * `polars.io_plugin.lazyframe_resolver.v1` ([`resolver`]): planning. Connects, finds the
//!   schema, translates the pushed filters, splits the scan into partitions and renders their SQL.
//! * `polars.io_plugin.file_reader.v1` ([`reader`]): reads one partition per source.
//!
//! [`database`] holds the database logic (dialects, SQL, partitioning, types, drivers). The Python
//! extension module (feature `python`) exposes the contracts as PyCapsules, the `scan_database`
//! entry point and the Python driver backends.
pub mod database;
pub mod ffi;
pub mod filter;
pub mod reader;
pub mod resolver;

#[cfg(feature = "python")]
pub mod python;

/// Python package name, as recorded in query plans and used in error messages.
pub const PACKAGE_NAME: &str = "polars_database";

/// The version of the serialized plan formats of [`resolver`] and [`reader`]: bincode of
/// the structs, so positional. Bump it whenever a serialized struct (or one inside it,
/// [`database::DatabaseScanOptions`] and the like) changes, so that a plan made by another
/// version fails with a clear error instead of being misread. One decimal digit.
const SERDE_VERSION: u8 = 5;
const _: () = assert!(SERDE_VERSION < 10);

/// The prefix of a serialized plan of `kind` (`PLDBRS` or `PLDBRB`): the kind and the
/// version. (The first format, before it was versioned, is version 1: `PLDBRS1`.)
fn serde_header(kind: &str) -> Vec<u8> {
    let mut out = kind.as_bytes().to_vec();
    out.push(b'0' + SERDE_VERSION);
    out
}

/// The bytes after the prefix of a serialized plan of `kind`; an error for another kind
/// of bytes or a version this one can't read.
fn strip_serde_header<'a>(bytes: &'a [u8], kind: &str) -> polars_error::PolarsResult<&'a [u8]> {
    use polars_error::polars_err;

    let rest = bytes.strip_prefix(kind.as_bytes()).ok_or_else(|| {
        polars_err!(
            ComputeError:
            "cannot deserialize a scan_database plan: these bytes are not one made by {}",
            PACKAGE_NAME
        )
    })?;
    match rest.split_first() {
        Some((&v, rest)) if v == b'0' + SERDE_VERSION => Ok(rest),
        Some((&v, _)) if v.is_ascii_digit() => Err(polars_err!(
            ComputeError:
            "cannot deserialize a scan_database plan: it was made by an incompatible version \
            of {} (plan format {}, this one reads {}); make the query again with the \
            installed version",
            PACKAGE_NAME, v - b'0', SERDE_VERSION
        )),
        _ => Err(polars_err!(
            ComputeError:
            "cannot deserialize a scan_database plan: it has no format version"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serde_header_round_trip_and_rejects_others() {
        for kind in ["PLDBRS", "PLDBRB"] {
            let mut bytes = serde_header(kind);
            bytes.extend_from_slice(b"payload");
            assert_eq!(strip_serde_header(&bytes, kind).unwrap(), b"payload");

            // The first format was unversioned. Also check the format immediately before this one.
            for version in [1, SERDE_VERSION - 1] {
                let old = [kind.as_bytes(), &[b'0' + version], b"payload"].concat();
                let err = strip_serde_header(&old, kind).unwrap_err().to_string();
                assert!(err.contains("incompatible version"), "{err}");
                assert!(err.contains(&format!("plan format {version}")), "{err}");
            }

            // another kind of plan, or foreign bytes
            let other = if kind == "PLDBRS" { "PLDBRB" } else { "PLDBRS" };
            for bad in [serde_header(other), b"garbage".to_vec(), Vec::new()] {
                let err = strip_serde_header(&bad, kind).unwrap_err().to_string();
                assert!(err.contains("cannot deserialize"), "{err}");
            }
            let err = strip_serde_header(kind.as_bytes(), kind).unwrap_err();
            assert!(err.to_string().contains("no format version"), "{err}");
        }
    }
}
