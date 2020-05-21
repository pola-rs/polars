use crate::{
    datatypes,
    datatypes::ArrowDataType,
    error::Result,
    series::{chunked_array::ChunkedArray, series::Series},
};
use arrow::datatypes::{Field, Schema};
use std::io::Read;
use std::sync::Arc;

type CSVReader<R> = arrow::csv::Reader<R>;

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
                ArrowDataType::Int32 => Series::Int32(
                    ChunkedArray::<datatypes::Int32Type>::new_from_chunks(field.name(), vec![]),
                ),
                ArrowDataType::Int64 => Series::Int64(
                    ChunkedArray::<datatypes::Int64Type>::new_from_chunks(field.name(), vec![]),
                ),
                ArrowDataType::Float32 => Series::Float32(
                    ChunkedArray::<datatypes::Float32Type>::new_from_chunks(field.name(), vec![]),
                ),
                ArrowDataType::Float64 => Series::Float64(
                    ChunkedArray::<datatypes::Float64Type>::new_from_chunks(field.name(), vec![]),
                ),
                ArrowDataType::Utf8 => Series::Utf8(
                    ChunkedArray::<datatypes::Utf8Type>::new_from_chunks(field.name(), vec![]),
                ),
                _ => unimplemented!(),
            })
            .collect::<Vec<_>>();

        while let Some(batch) = reader.next()? {
            batch
                .columns()
                .into_iter()
                .zip(&mut columns)
                .map(|(arr, ser)| ser.append_array(arr.clone()))
                .collect::<Result<Vec<_>>>()?;
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
