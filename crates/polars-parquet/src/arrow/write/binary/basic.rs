use arrow::array::{Array, BinaryArray, ValueSize};
use arrow::bitmap::Bitmap;
use arrow::offset::Offset;
use polars_error::PolarsResult;

use super::super::{utils, WriteOptions};
use crate::arrow::read::schema::is_nullable;
use crate::parquet::encoding::{delta_bitpacked, Encoding};
use crate::parquet::schema::types::PrimitiveType;
use crate::parquet::statistics::{BinaryStatistics, ParquetStatistics};
use crate::write::utils::invalid_encoding;
use crate::write::{EncodeNullability, Page, StatisticsOptions};

pub(crate) fn encode_non_null_values<'a, I: Iterator<Item = &'a [u8]>>(
    iter: I,
    buffer: &mut Vec<u8>,
) {
    iter.for_each(|x| {
        // BYTE_ARRAY: first 4 bytes denote length in littleendian.
        let len = (x.len() as u32).to_le_bytes();
        buffer.extend_from_slice(&len);
        buffer.extend_from_slice(x);
    })
}

pub(crate) fn encode_plain<O: Offset>(
    array: &BinaryArray<O>,
    options: EncodeNullability,
    buffer: &mut Vec<u8>,
) {
    if options.is_optional() && array.validity().is_some() {
        let len_before = buffer.len();
        let capacity = array.get_values_size()
            + (array.len() - array.null_count()) * std::mem::size_of::<u32>();
        buffer.reserve(capacity);
        encode_non_null_values(array.non_null_values_iter(), buffer);
        // Ensure we allocated properly.
        debug_assert_eq!(buffer.len() - len_before, capacity);
    } else {
        let len_before = buffer.len();
        let capacity = array.get_values_size() + array.len() * std::mem::size_of::<u32>();
        buffer.reserve(capacity);
        encode_non_null_values(array.values_iter(), buffer);
        // Ensure we allocated properly.
        debug_assert_eq!(buffer.len() - len_before, capacity);
    }
}

pub fn array_to_page<O: Offset>(
    array: &BinaryArray<O>,
    options: WriteOptions,
    type_: PrimitiveType,
    encoding: Encoding,
) -> PolarsResult<Page> {
    let validity = array.validity();
    let is_optional = is_nullable(&type_.field_info);
    let encode_options = EncodeNullability::new(is_optional);

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
        Encoding::Plain => encode_plain(array, encode_options, &mut buffer),
        Encoding::DeltaLengthByteArray => encode_delta(
            array.values(),
            array.offsets().buffer(),
            array.validity(),
            encode_options,
            &mut buffer,
        ),
        _ => return Err(invalid_encoding(encoding, array.data_type())),
    }

    let statistics = if options.has_statistics() {
        Some(build_statistics(array, type_.clone(), &options.statistics))
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
    .map(Page::Data)
}

pub(crate) fn build_statistics<O: Offset>(
    array: &BinaryArray<O>,
    primitive_type: PrimitiveType,
    options: &StatisticsOptions,
) -> ParquetStatistics {
    use polars_compute::min_max::MinMaxKernel;

    BinaryStatistics {
        primitive_type,
        null_count: options.null_count.then_some(array.null_count() as i64),
        distinct_count: None,
        max_value: options
            .max_value
            .then(|| array.max_propagate_nan_kernel().map(<[u8]>::to_vec))
            .flatten(),
        min_value: options
            .min_value
            .then(|| array.min_propagate_nan_kernel().map(<[u8]>::to_vec))
            .flatten(),
    }
    .serialize()
}

pub(crate) fn encode_delta<O: Offset>(
    values: &[u8],
    offsets: &[O],
    validity: Option<&Bitmap>,
    options: EncodeNullability,
    buffer: &mut Vec<u8>,
) {
    if options.is_optional() && validity.is_some() {
        if let Some(validity) = validity {
            let lengths = offsets
                .windows(2)
                .map(|w| (w[1] - w[0]).to_usize() as i64)
                .zip(validity.iter())
                .flat_map(|(x, is_valid)| if is_valid { Some(x) } else { None });
            let length = offsets.len() - 1 - validity.unset_bits();
            let lengths = utils::ExactSizedIter::new(lengths, length);

            delta_bitpacked::encode(lengths, buffer, 1);
        } else {
            let lengths = offsets.windows(2).map(|w| (w[1] - w[0]).to_usize() as i64);
            delta_bitpacked::encode(lengths, buffer, 1);
        }
    } else {
        let lengths = offsets.windows(2).map(|w| (w[1] - w[0]).to_usize() as i64);
        delta_bitpacked::encode(lengths, buffer, 1);
    }

    buffer.extend_from_slice(
        &values[offsets.first().unwrap().to_usize()..offsets.last().unwrap().to_usize()],
    )
}

/// Returns the ordering of two binary values. This corresponds to pyarrows' ordering
/// of statistics.
#[inline(always)]
pub(crate) fn ord_binary<'a>(a: &'a [u8], b: &'a [u8]) -> std::cmp::Ordering {
    a.cmp(b)
}
