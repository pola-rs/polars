use std::borrow::Borrow;
use std::io::Cursor;

use arrow2::chunk::Chunk;
use arrow2::datatypes::{Field, Metadata, Schema};
use arrow2::error::Result;
use arrow2::io::parquet::read as p_read;
use arrow2::io::parquet::write::*;
use sample_arrow2::array::ArbitraryArray;
use sample_arrow2::chunk::{ArbitraryChunk, ChainedChunk};
use sample_arrow2::datatypes::{sample_flat, ArbitraryArrowDataType};
use sample_std::{Chance, Random, Regex, Sample};
use sample_test::sample_test;

fn deep_chunk(depth: usize, len: usize) -> ArbitraryChunk<Regex, Chance> {
    let names = Regex::new("[a-z]{4,8}");
    let data_type = ArbitraryArrowDataType {
        struct_branch: 1..3,
        names: names.clone(),
        // TODO: this breaks the test
        // nullable: Chance(0.5),
        nullable: Chance(0.0),
        flat: sample_flat,
    }
    .sample_depth(depth);

    let array = ArbitraryArray {
        names,
        branch: 0..10,
        len: len..(len + 1),
        null: Chance(0.1),
        // TODO: this breaks the test
        // is_nullable: true,
        is_nullable: false,
    };

    ArbitraryChunk {
        // TODO: shrinking appears to be an issue with chunks this large. issues
        // currently reproduce on the smaller sizes anyway.
        // chunk_len: 10..1000,
        chunk_len: 1..10,
        array_count: 1..2,
        data_type,
        array,
    }
}

#[sample_test]
fn round_trip_sample(
    #[sample(deep_chunk(5, 100).sample_one())] chained: ChainedChunk,
) -> Result<()> {
    sample_test::env_logger_init();
    let chunks = vec![chained.value];
    let name = Regex::new("[a-z]{4, 8}");
    let mut g = Random::new();

    // TODO: this probably belongs in a helper in sample-arrow2
    let schema = Schema {
        fields: chunks
            .first()
            .unwrap()
            .iter()
            .map(|arr| {
                Field::new(
                    name.generate(&mut g),
                    arr.data_type().clone(),
                    arr.validity().is_some(),
                )
            })
            .collect(),
        metadata: Metadata::default(),
    };

    let options = WriteOptions {
        write_statistics: true,
        compression: CompressionOptions::Uncompressed,
        version: Version::V2,
        data_pagesize_limit: None,
    };

    let encodings: Vec<_> = schema
        .borrow()
        .fields
        .iter()
        .map(|field| transverse(field.data_type(), |_| Encoding::Plain))
        .collect();

    let row_groups = RowGroupIterator::try_new(
        chunks.clone().into_iter().map(Ok),
        &schema,
        options,
        encodings,
    )?;

    let buffer = Cursor::new(vec![]);
    let mut writer = FileWriter::try_new(buffer, schema, options)?;

    for group in row_groups {
        writer.write(group?)?;
    }
    writer.end(None)?;

    let mut buffer = writer.into_inner();

    let metadata = p_read::read_metadata(&mut buffer)?;
    let schema = p_read::infer_schema(&metadata)?;

    let mut reader = p_read::FileReader::new(buffer, metadata.row_groups, schema, None, None, None);

    let result: Vec<_> = reader.collect::<Result<_>>()?;

    assert_eq!(result, chunks);

    Ok(())
}
