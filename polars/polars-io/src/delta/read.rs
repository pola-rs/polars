
use std::io::{Read, Seek};

/// Read Delta lake format into a DataFrame.
#[must_use]
pub struct DeltaReader<R: Read + Seek> {
    reader: R,
}