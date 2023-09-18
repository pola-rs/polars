use parquet2::{
    encoding::Encoding,
    page::DataPage,
    schema::types::PrimitiveType,
    statistics::{serialize_statistics, BinaryStatistics, ParquetStatistics, Statistics},
};

use super::super::binary::{encode_delta, ord_binary};
use super::super::utils;
use super::super::WriteOptions;
use crate::{
    array::{Array, Utf8Array},
    error::{Error, Result},
    io::parquet::read::schema::is_nullable,
    offset::Offset,
};

pub(crate) fn encode_plain<O: Offset>(
    array: &Utf8Array<O>,
    is_optional: bool,
    buffer: &mut Vec<u8>,
) {
    if is_optional {
        array.iter().for_each(|x| {
            if let Some(x) = x {
                // BYTE_ARRAY: first 4 bytes denote length in littleendian.
                let len = (x.len() as u32).to_le_bytes();
                buffer.extend_from_slice(&len);
                buffer.extend_from_slice(x.as_bytes());
            }
        })
    } else {
        array.values_iter().for_each(|x| {
            // BYTE_ARRAY: first 4 bytes denote length in littleendian.
            let len = (x.len() as u32).to_le_bytes();
            buffer.extend_from_slice(&len);
            buffer.extend_from_slice(x.as_bytes());
        })
    }
}

pub fn array_to_page<O: Offset>(
    array: &Utf8Array<O>,
    options: WriteOptions,
    type_: PrimitiveType,
    encoding: Encoding,
) -> Result<DataPage> {
    let validity = array.validity();
    let is_optional = is_nullable(&type_.field_info);

    let mut buffer = vec![];
    utils::write_def_levels(
        &mut buffer,
        is_optional,
        validity,
        array.len(),
        options.version,
    )?;

    let definition_levels_byte_length = buffer.len();

    match encoding {
        Encoding::Plain => encode_plain(array, is_optional, &mut buffer),
        Encoding::DeltaLengthByteArray => encode_delta(
            array.values(),
            array.offsets().buffer(),
            array.validity(),
            is_optional,
            &mut buffer,
        ),
        _ => {
            return Err(Error::InvalidArgumentError(format!(
                "Datatype {:?} cannot be encoded by {:?} encoding",
                array.data_type(),
                encoding
            )))
        }
    }

    let statistics = if options.write_statistics {
        Some(build_statistics(array, type_.clone()))
    } else {
        None
    };

    utils::build_plain_page(
        buffer,
        array.len(),
        array.len(),
        array.null_count(),
        0,
        definition_levels_byte_length,
        statistics,
        type_,
        options,
        encoding,
    )
}

pub(crate) fn build_statistics<O: Offset>(
    array: &Utf8Array<O>,
    primitive_type: PrimitiveType,
) -> ParquetStatistics {
    let statistics = &BinaryStatistics {
        primitive_type,
        null_count: Some(array.null_count() as i64),
        distinct_count: None,
        max_value: array
            .iter()
            .flatten()
            .map(|x| x.as_bytes())
            .max_by(|x, y| ord_binary(x, y))
            .map(|x| x.to_vec()),
        min_value: array
            .iter()
            .flatten()
            .map(|x| x.as_bytes())
            .min_by(|x, y| ord_binary(x, y))
            .map(|x| x.to_vec()),
    } as &dyn Statistics;
    serialize_statistics(statistics)
}
