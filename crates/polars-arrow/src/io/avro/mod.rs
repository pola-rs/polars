//! Read and write from and to Apache Avro

pub use avro_schema;

pub mod read;
pub mod write;

// macros that can operate in sync and async code.
macro_rules! avro_decode {
    ($reader:ident $($_await:tt)*) => {
        {
            let mut i = 0u64;
            let mut buf = [0u8; 1];
            let mut j = 0;
            loop {
                if j > 9 {
                    // if j * 7 > 64
                    polars_error::polars_bail!(oos = "zigzag decoding failed - corrupt avro file")
                }
                $reader.read_exact(&mut buf[..])$($_await)*?;
                i |= (u64::from(buf[0] & 0x7F)) << (j * 7);
                if (buf[0] >> 7) == 0 {
                    break;
                } else {
                    j += 1;
                }
            }

            Ok(i)
        }
    }
}

pub(crate) use avro_decode;
