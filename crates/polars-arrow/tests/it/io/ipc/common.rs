use std::fs::File;
use std::io::Read;

use ahash::AHashMap;
use flate2::read::GzDecoder;
use polars_arrow::array::Array;
use polars_arrow::chunk::Chunk;
use polars_arrow::datatypes::Schema;
use polars_arrow::error::Result;
use polars_arrow::io::ipc::read::{read_stream_metadata, StreamReader};
use polars_arrow::io::ipc::IpcField;
use polars_arrow::io::json_integration::{read, ArrowJson};

type IpcRead = (Schema, Vec<IpcField>, Vec<Chunk<Box<dyn Array>>>);

/// Read gzipped JSON file
pub fn read_gzip_json(version: &str, file_name: &str) -> Result<IpcRead> {
    let testdata = crate::test_util::arrow_test_data();
    let file = File::open(format!(
        "{testdata}/arrow-ipc-stream/integration/{version}/{file_name}.json.gz"
    ))
    .unwrap();
    let mut gz = GzDecoder::new(&file);
    let mut s = String::new();
    gz.read_to_string(&mut s).unwrap();
    // convert to Arrow JSON
    let arrow_json: ArrowJson = serde_json::from_str(&s)?;

    let schema = serde_json::to_value(arrow_json.schema).unwrap();

    let (schema, ipc_fields) = read::deserialize_schema(&schema)?;

    // read dictionaries
    let mut dictionaries = AHashMap::new();
    if let Some(dicts) = arrow_json.dictionaries {
        for json_dict in dicts {
            // TODO: convert to a concrete Arrow type
            dictionaries.insert(json_dict.id, json_dict);
        }
    }

    let batches = arrow_json
        .batches
        .iter()
        .map(|batch| read::deserialize_chunk(&schema, &ipc_fields, batch, &dictionaries))
        .collect::<Result<Vec<_>>>()?;

    Ok((schema, ipc_fields, batches))
}

pub fn read_arrow_stream(
    version: &str,
    file_name: &str,
    projection: Option<Vec<usize>>,
) -> IpcRead {
    let testdata = crate::test_util::arrow_test_data();
    let mut file = File::open(format!(
        "{testdata}/arrow-ipc-stream/integration/{version}/{file_name}.stream"
    ))
    .unwrap();

    let metadata = read_stream_metadata(&mut file).unwrap();
    let reader = StreamReader::new(file, metadata, projection);

    let schema = reader.metadata().schema.clone();
    let ipc_fields = reader.metadata().ipc_schema.fields.clone();

    (
        schema,
        ipc_fields,
        reader
            .map(|x| x.map(|x| x.unwrap()))
            .collect::<Result<_>>()
            .unwrap(),
    )
}
