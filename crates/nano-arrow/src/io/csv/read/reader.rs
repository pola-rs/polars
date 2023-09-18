use std::io::Read;

use super::{ByteRecord, Reader};

use crate::error::{Error, Result};

/// Reads `len` rows from `reader` into `row`, skiping the first `skip`.
/// This operation has minimal CPU work and is thus the fastest way to read through a CSV
/// without deserializing the contents to Arrow.
pub fn read_rows<R: Read>(
    reader: &mut Reader<R>,
    skip: usize,
    rows: &mut [ByteRecord],
) -> Result<usize> {
    // skip first `start` rows.
    let mut row = ByteRecord::new();
    for _ in 0..skip {
        let res = reader.read_byte_record(&mut row);
        if !res.unwrap_or(false) {
            break;
        }
    }

    let mut row_number = 0;
    for row in rows.iter_mut() {
        let has_more = reader
            .read_byte_record(row)
            .map_err(|e| Error::External(format!(" at line {}", skip + row_number), Box::new(e)))?;
        if !has_more {
            break;
        }
        row_number += 1;
    }
    Ok(row_number)
}
