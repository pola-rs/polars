use std::io::Read;

use super::super::avro_decode;
use crate::error::{Error, Result};

pub fn zigzag_i64<R: Read>(reader: &mut R) -> Result<i64> {
    let z = decode_variable(reader)?;
    Ok(if z & 0x1 == 0 {
        (z >> 1) as i64
    } else {
        !(z >> 1) as i64
    })
}

fn decode_variable<R: Read>(reader: &mut R) -> Result<u64> {
    avro_decode!(reader)
}
