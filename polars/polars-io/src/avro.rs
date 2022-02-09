use std::io::{Read, Seek};

use super::{finish_reader, ArrowChunk, ArrowReader, ArrowResult};
use crate::prelude::*;
use polars_core::prelude::*;

use arrow::io::avro::read;

#[must_use]
pub struct AvroReader<R> {
    reader: R,
    rechunk: bool,
    n_rows: Option<usize>,
}

impl<R: Read + Seek> AvroReader<R> {
    /// Get schema of the Avro File
    pub fn schema(&mut self) -> Result<Schema> {
        let (_, schema, _, _) = read::read_metadata(&mut self.reader)?;
        Ok(schema.into())
    }

    /// Get arrow schema of the Ipc File, this is faster than a polars schema.
    pub fn arrow_schema(&mut self) -> Result<ArrowSchema> {
        let (_, schema, _, _) = read::read_metadata(&mut self.reader)?;
        Ok(schema)
    }

    /// Stop reading when `n` rows are read.
    pub fn with_n_rows(mut self, num_rows: Option<usize>) -> Self {
        self.n_rows = num_rows;
        self
    }
}

impl<R> ArrowReader for read::Reader<R>
where
    R: Read + Seek,
{
    fn next_record_batch(&mut self) -> ArrowResult<Option<ArrowChunk>> {
        self.next().map_or(Ok(None), |v| v.map(Some))
    }
}

impl<R> SerReader<R> for AvroReader<R>
where
    R: Read + Seek,
{
    fn new(reader: R) -> Self {
        AvroReader {
            reader,
            rechunk: true,
            n_rows: None,
        }
    }

    fn set_rechunk(mut self, rechunk: bool) -> Self {
        self.rechunk = rechunk;
        self
    }

    fn finish(mut self) -> Result<DataFrame> {
        let rechunk = self.rechunk;
        let (avro_schema, schema, codec, file_marker) = read::read_metadata(&mut self.reader)?;

        let avro_reader = read::Reader::new(
            read::Decompressor::new(
                read::BlockStreamIterator::new(&mut self.reader, file_marker),
                codec,
            ),
            avro_schema,
            schema.clone().fields,
        );

        finish_reader(avro_reader, rechunk, None, None, None, &schema, None)
    }
}

#[cfg(test)]
mod test {
    use crate::avro::AvroReader;
    use crate::SerReader;
    use arrow::array::Array;
    use polars_core::df;
    use polars_core::prelude::*;
    use std::io::Cursor;

    fn write_avro(buf: &mut Cursor<Vec<u8>>) {
        use arrow::array::{Float64Array, Int64Array, Utf8Array};
        use arrow::datatypes::{Field, Schema};
        use arrow::io::avro::write;

        let i64_array = Int64Array::from(&[Some(1), Some(2)]);
        let f64_array = Float64Array::from(&[Some(0.1), Some(0.2)]);
        let utf8_array = Utf8Array::<i32>::from(&[Some("a"), Some("b")]);
        let i64_field = Field::new("i64", i64_array.data_type().clone(), true);
        let f64_field = Field::new("f64", f64_array.data_type().clone(), true);
        let utf8_field = Field::new("utf8", utf8_array.data_type().clone(), true);
        let schema = Schema::from(vec![i64_field, f64_field, utf8_field]);
        let arrays = vec![
            &i64_array as &dyn Array,
            &f64_array as &dyn Array,
            &utf8_array as &dyn Array,
        ];
        let avro_fields = write::to_avro_schema(&schema).unwrap();

        let mut serializers = arrays
            .iter()
            .zip(avro_fields.iter())
            .map(|(array, field)| write::new_serializer(*array, &field.schema))
            .collect::<Vec<_>>();
        let mut block = write::Block::new(arrays[0].len(), vec![]);

        write::serialize(&mut serializers, &mut block);

        let mut compressed_block = write::CompressedBlock::default();

        let _was_compressed = write::compress(&mut block, &mut compressed_block, None).unwrap();

        write::write_metadata(buf, avro_fields.clone(), None).unwrap();

        write::write_block(buf, &compressed_block).unwrap();
    }

    #[test]
    fn write_and_read_avro_naive() {
        let mut buf: Cursor<Vec<u8>> = Cursor::new(Vec::new());
        write_avro(&mut buf);
        buf.set_position(0);

        let df = AvroReader::new(buf).finish();
        assert!(df.is_ok());
        let df = df.unwrap();

        let expected = df!(
            "i64" => &[1, 2],
            "f64" => &[0.1, 0.2],
            "utf8" => &["a", "b"]
        )
        .unwrap();
        assert_eq!(df.shape(), expected.shape());
        assert!(df.frame_equal(&expected));
    }
}
