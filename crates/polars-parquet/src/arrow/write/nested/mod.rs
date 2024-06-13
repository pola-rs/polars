mod dremel;

pub use dremel::num_values;
use polars_error::PolarsResult;

use super::Nested;
use crate::parquet::encoding::hybrid_rle::encode;
use crate::parquet::read::levels::get_bit_width;
use crate::parquet::write::Version;

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

/// writes the rep levels to a `Vec<u8>`.
fn write_rep_levels(buffer: &mut Vec<u8>, nested: &[Nested], version: Version) -> PolarsResult<()> {
    let max_level = max_rep_level(nested) as i16;
    if max_level == 0 {
        return Ok(());
    }
    let num_bits = get_bit_width(max_level);

    let levels = dremel::BufferedDremelIter::new(nested).map(|d| u32::from(d.rep));

    match version {
        Version::V1 => {
            write_levels_v1(buffer, |buffer: &mut Vec<u8>| {
                encode::<u32, _, _>(buffer, levels, num_bits)?;
                Ok(())
            })?;
        },
        Version::V2 => {
            encode::<u32, _, _>(buffer, levels, num_bits)?;
        },
    }

    Ok(())
}

/// writes the def levels to a `Vec<u8>`.
fn write_def_levels(buffer: &mut Vec<u8>, nested: &[Nested], version: Version) -> PolarsResult<()> {
    let max_level = max_def_level(nested) as i16;
    if max_level == 0 {
        return Ok(());
    }
    let num_bits = get_bit_width(max_level);

    let levels = dremel::BufferedDremelIter::new(nested).map(|d| u32::from(d.def));

    match version {
        Version::V1 => write_levels_v1(buffer, move |buffer: &mut Vec<u8>| {
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
pub fn write_rep_and_def(
    page_version: Version,
    nested: &[Nested],
    buffer: &mut Vec<u8>,
) -> PolarsResult<(usize, usize)> {
    write_rep_levels(buffer, nested, page_version)?;
    let repetition_levels_byte_length = buffer.len();

    write_def_levels(buffer, nested, page_version)?;
    let definition_levels_byte_length = buffer.len() - repetition_levels_byte_length;

    Ok((repetition_levels_byte_length, definition_levels_byte_length))
}
