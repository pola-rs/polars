use crate::error::Result;
use crate::series::{
    primitive::ChunkedArray,
    series::{Series, SeriesRef},
};
use arrow::array::PrimitiveArray;
use arrow::{
    datatypes,
    datatypes::{DataType, Field, Schema},
};
use std::any::Any;
use std::io::Read;
use std::path::Path;
use std::sync::Arc;

type CSVReader<R: Read> = arrow::csv::Reader<R>;

struct DataFrame {
    schema: Arc<Schema>,
    columns: Vec<Series>,
}

impl DataFrame {
    fn fields(&self) -> &Vec<Field> {
        self.schema.fields()
    }

    fn new_from_csv<R: Read>(reader: &mut CSVReader<R>) -> Result<Self> {
        let mut columns = reader
            .schema()
            .fields()
            .iter()
            .map(|field| match field.data_type() {
                DataType::Int32 => Series::Int32(
                    ChunkedArray::<datatypes::Int32Type>::new_from_chunks(field.name(), vec![]),
                ),
                DataType::Int64 => Series::Int64(
                    ChunkedArray::<datatypes::Int64Type>::new_from_chunks(field.name(), vec![]),
                ),
                _ => unimplemented!(),
            })
            .collect::<Vec<_>>();

        while let Some(batch) = reader.next()? {
            batch
                .columns()
                .into_iter()
                .zip(&mut columns)
                .for_each(|(arr, ser)| {
                    ser.append_array(arr.clone());
                })
        }

        Ok(DataFrame {
            schema: reader.schema(),
            columns,
        })
    }
}

mod test {
    use super::*;
    use arrow::csv;
    use std::fs::File;

    #[test]
    fn read_csv() {
        let file = File::open("data/iris.csv").unwrap();
        let builder = csv::ReaderBuilder::new().infer_schema(None);
        let mut reader = builder.build(file).unwrap();

        println!("{:?}", reader.schema());
        let df = DataFrame::new_from_csv(&mut reader);
    }
}
