use arrow::array::{Array, Int32Array, Utf8Array};
use arrow::datatypes::{DataType, Field};
use arrow::error::Result;
use arrow::io::odbc::api::Cursor;
use arrow::io::odbc::{api, read, write};

use super::*;

pub struct OdbcReader<R> {
    reader: R,
    connection_string: AsRef<str>,
}

impl<'a, R> SerReader<R> for OdbcReader<'a, R>
where
    R: 'a + MmapBytesReader,
{
    fn new(reader: R) -> Self {
        OdbcReader { reader }
    }

    fn set_connection_string(mut self, connection_string: AsRef<str>) -> Self {
        self.connection_string = connection_string.as_ref();
        self
    }

    fn finish(mut self) -> PolarsResult<DataFrame> {
        todo!()
    }
}
