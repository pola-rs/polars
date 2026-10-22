mod dremel;

use dremel::{Level, Levels};
use polars_arrow::array::Array;
use polars_error::PolarsResult;

use super::Nested;
use crate::parquet::encoding::hybrid_rle::encode;
use crate::parquet::read::levels::get_bit_width;
use crate::parquet::write::Version;

/// Returns the number of null values of a page, see
/// [`DataPage::new`](crate::parquet::page::DataPage::new). Every non-null value of the leaf
/// `array` is a value of the page.
pub fn num_nulls(num_values: usize, array: &dyn Array) -> usize {
    num_values - (array.len() - array.null_count())
}

fn write_levels_v1<F: FnOnce(&mut Vec<u8>) -> PolarsResult<()>>(
    buffer: &mut Vec<u8>,
    encode: F,
) -> PolarsResult<()> {
    buffer.extend_from_slice(&[0; 4]);
    let start = buffer.len();

    encode(buffer)?;

    let end = buffer.len();
    let length = end - start;

    // write the first 4 bytes as length
    let length = (length as i32).to_le_bytes();
    (0..4).for_each(|i| buffer[start - 4 + i] = length[i]);
    Ok(())
}

/// Writes `levels` with the hybrid RLE encoding to `buffer`.
fn write_levels<T: Level>(
    buffer: &mut Vec<u8>,
    levels: &[T],
    max_level: usize,
    version: Version,
) -> PolarsResult<()> {
    if max_level == 0 {
        return Ok(());
    }
    let num_bits = get_bit_width(max_level as i16);
    let levels = levels.iter().map(|&l| l.into());

    match version {
        Version::V1 => write_levels_v1(buffer, |buffer: &mut Vec<u8>| {
            encode::<u32, _, _>(buffer, levels, num_bits)?;
            Ok(())
        }),
        Version::V2 => Ok(encode::<u32, _, _>(buffer, levels, num_bits)?),
    }
}

fn max_def_level(nested: &[Nested]) -> usize {
    nested
        .iter()
        .map(|nested| match nested {
            Nested::Primitive(nested) => nested.is_optional as usize,
            Nested::List(nested) => 1 + (nested.is_optional as usize),
            Nested::LargeList(nested) => 1 + (nested.is_optional as usize),
            Nested::Struct(nested) => nested.is_optional as usize,
            Nested::FixedSizeList(nested) => 1 + nested.is_optional as usize,
        })
        .sum()
}

fn max_rep_level(nested: &[Nested]) -> usize {
    nested
        .iter()
        .map(|nested| match nested {
            Nested::FixedSizeList(_) | Nested::LargeList(_) | Nested::List(_) => 1,
            Nested::Primitive(_) | Nested::Struct(_) => 0,
        })
        .sum()
}

/// Write `repetition_levels` and `definition_levels` to buffer.
///
/// Returns the byte lengths of the repetition and definition levels and the number of values.
pub fn write_rep_and_def(
    page_version: Version,
    nested: &[Nested],
    buffer: &mut Vec<u8>,
) -> PolarsResult<(usize, usize, usize)> {
    let max_rep = max_rep_level(nested);
    let max_def = max_def_level(nested);
    // Every list adds a definition-level, so `max_rep <= max_def`.
    if max_def <= u8::MAX as usize {
        write_levels_typed::<u8>(page_version, nested, max_rep, max_def, buffer)
    } else {
        write_levels_typed::<u16>(page_version, nested, max_rep, max_def, buffer)
    }
}

fn write_levels_typed<T: Level>(
    page_version: Version,
    nested: &[Nested],
    max_rep: usize,
    max_def: usize,
    buffer: &mut Vec<u8>,
) -> PolarsResult<(usize, usize, usize)> {
    let levels = Levels::<T>::new(nested, max_rep > 0);

    write_levels(buffer, &levels.rep, max_rep, page_version)?;
    let repetition_levels_byte_length = buffer.len();

    write_levels(buffer, &levels.def, max_def, page_version)?;
    let definition_levels_byte_length = buffer.len() - repetition_levels_byte_length;

    Ok((
        repetition_levels_byte_length,
        definition_levels_byte_length,
        levels.num_values(),
    ))
}
