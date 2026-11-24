use polars_arrow::array::*;
use polars_arrow::datatypes::*;
use polars_arrow::io::avro::read;
use polars_arrow::io::avro::read::read_metadata;
use polars_arrow::record_batch::RecordBatchT;
use polars_error::PolarsResult;

pub(super) fn read_avro(
    mut avro: &[u8],
    projection: Option<Vec<bool>>,
) -> PolarsResult<(RecordBatchT<Box<dyn Array>>, ArrowSchema)> {
    let file = &mut avro;

    let metadata = read_metadata(file)?;
    let schema = read::infer_schema(&metadata.record)?;

    let mut reader = read::Reader::new(file, metadata, schema.clone(), projection.clone());

    let schema = if let Some(projection) = projection {
        schema
            .into_iter_values()
            .zip(projection.iter())
            .filter_map(|x| if *x.1 { Some(x.0) } else { None })
            .collect()
    } else {
        schema
    };

    reader.next().unwrap().map(|x| (x, schema))
}
