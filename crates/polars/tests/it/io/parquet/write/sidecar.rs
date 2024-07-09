use polars_parquet::parquet::error::ParquetError;
use polars_parquet::parquet::metadata::SchemaDescriptor;
use polars_parquet::parquet::schema::types::{ParquetType, PhysicalType};
use polars_parquet::parquet::write::{write_metadata_sidecar, FileWriter, Version, WriteOptions};

#[test]
fn basic() -> Result<(), ParquetError> {
    let schema = SchemaDescriptor::new(
        "schema".to_string(),
        vec![ParquetType::from_physical(
            "c1".to_string(),
            PhysicalType::Int32,
        )],
    );

    let mut metadatas = vec![];
    for i in 0..10 {
        // say we will write 10 files
        let relative_path = format!("part-{i}.parquet");
        let writer = std::io::Cursor::new(vec![]);
        let mut writer = FileWriter::new(
            writer,
            schema.clone(),
            WriteOptions {
                write_statistics: true,
                version: Version::V2,
            },
            None,
        );
        writer.end(None)?;
        let (_, mut metadata) = writer.into_inner_and_metadata();

        // once done, we write their relative paths:
        metadata.row_groups.iter_mut().for_each(|row_group| {
            row_group
                .columns
                .iter_mut()
                .for_each(|column| column.file_path = Some(relative_path.clone()))
        });
        metadatas.push(metadata);
    }

    // merge their row groups
    let first = metadatas.pop().unwrap();
    let sidecar = metadatas.into_iter().fold(first, |mut acc, metadata| {
        acc.row_groups.extend(metadata.row_groups);
        acc
    });

    // and write the metadata on a separate file
    let mut writer = std::io::Cursor::new(vec![]);
    write_metadata_sidecar(&mut writer, &sidecar)?;

    Ok(())
}
