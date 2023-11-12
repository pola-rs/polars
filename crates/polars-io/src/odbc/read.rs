use arrow2::array::{Array, Int32Array, Utf8Array};
use arrow2::chunk::Chunk;
use arrow2::datatypes::{DataType, Field};
use arrow2::error::Result;
use arrow2::io::odbc::api::Cursor;
use arrow2::io::odbc::{api, read, write};

pub struct OdbcReader
