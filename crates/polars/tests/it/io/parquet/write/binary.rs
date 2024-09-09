use polars_parquet::parquet::encoding::hybrid_rle::encode;
use polars_parquet::parquet::encoding::Encoding;
use polars_parquet::parquet::error::ParquetResult;
use polars_parquet::parquet::metadata::Descriptor;
use polars_parquet::parquet::page::{DataPage, DataPageHeader, DataPageHeaderV1, Page};
use polars_parquet::parquet::statistics::BinaryStatistics;
use polars_parquet::parquet::types::ord_binary;
use polars_parquet::parquet::write::WriteOptions;
use polars_parquet::parquet::CowBuffer;

fn unzip_option(array: &[Option<Vec<u8>>]) -> ParquetResult<(Vec<u8>, Vec<u8>)> {
    // leave the first 4 bytes announcing the length of the def level
    // this will be overwritten at the end, once the length is known.
    // This is unknown at this point because of the uleb128 encoding,
    // whose length is variable.
    let mut validity = std::io::Cursor::new(vec![0; 4]);
    validity.set_position(4);

    let mut values = vec![];
    let iter = array.iter().map(|value| {
        if let Some(item) = value {
            values.extend_from_slice(&(item.len() as i32).to_le_bytes());
            values.extend_from_slice(item.as_ref());
            true
        } else {
            false
        }
    });
    encode::<bool, _, _>(&mut validity, iter, 1)?;

    // write the length, now that it is known
    let mut validity = validity.into_inner();
    let length = validity.len() - 4;
    // todo: pay this small debt (loop?)
    let length = length.to_le_bytes();
    validity[0] = length[0];
    validity[1] = length[1];
    validity[2] = length[2];
    validity[3] = length[3];

    Ok((values, validity))
}

pub fn array_to_page_v1(
    array: &[Option<Vec<u8>>],
    options: &WriteOptions,
    descriptor: &Descriptor,
) -> ParquetResult<Page> {
    let (values, mut buffer) = unzip_option(array)?;

    buffer.extend_from_slice(&values);

    let statistics = if options.write_statistics {
        let statistics = &BinaryStatistics {
            primitive_type: descriptor.primitive_type.clone(),
            null_count: Some((array.len() - array.iter().flatten().count()) as i64),
            distinct_count: None,
            max_value: array
                .iter()
                .flatten()
                .max_by(|x, y| ord_binary(x, y))
                .cloned(),
            min_value: array
                .iter()
                .flatten()
                .min_by(|x, y| ord_binary(x, y))
                .cloned(),
        };
        Some(statistics.serialize())
    } else {
        None
    };

    let header = DataPageHeaderV1 {
        num_values: array.len() as i32,
        encoding: Encoding::Plain.into(),
        definition_level_encoding: Encoding::Rle.into(),
        repetition_level_encoding: Encoding::Rle.into(),
        statistics,
    };

    Ok(Page::Data(DataPage::new(
        DataPageHeader::V1(header),
        CowBuffer::Owned(buffer),
        descriptor.clone(),
        array.len(),
    )))
}
