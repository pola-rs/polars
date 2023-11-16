use arrow::array::{Array, Int32Array, Utf8Array};
use arrow::datatypes::{DataType, Field};
use arrow::error::Result;
use arrow::io::odbc::api::Cursor;
use arrow::io::odbc::{api, read, write};

use super::{DataFrame, MmapBytesReader, PolarsResult, SerReader};

pub struct OdbcReader<R> {
    reader: R,
    connector: AsRef<str>,
    max_batch_size: i32,
    query: AsRef<str>,
}

impl<'a, R> SerReader<R> for OdbcReader<'a, R>
where
    R: 'a + MmapBytesReader,
{
    fn new(reader: R) -> Self {
        OdbcReader { reader }
    }

    fn set_connector(mut self, connector: AsRef<str>) -> Self {
        self.connection_string = connection_string.as_ref();
        self
    }

    fn set_query(mut self, query: AsRef<str>) -> Self {
        self.query = query.as_ref();
        self
    }

    fn set_max_batch_size(mut self, max_batch_size: i32) -> Self {
        self.max_batch_size = max_batch_size
        self
    }

    fn finish(mut self) -> PolarsResult<DataFrame> {
        let env = api::Environment::new()?;
        let connection = env.connect_with_connection_string(connector)?;

        let mut a = connection.prepare(query)?;
        let fields = read::infer_schema(&a)?;

        let buffer = read::buffer_from_metadata(&a, self.max_batch_size)?;

        let mut chunks = vec![];
        while let Some(batch) = cursor.fetch()? {
            let arrays = (0..batch.num_cols())
                .zip(fields.iter())
                .map(|(index, field)| {
                    let column_view = batch.column(index);
                    read::deserialize(column_view, field.data_type.clone())
                })
                .collect::<Vec<_>>();
            chunks.push(Chunk::new(arrays));
        }
    
        todo!()
    }
}
