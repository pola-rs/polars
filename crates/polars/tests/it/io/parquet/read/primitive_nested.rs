use polars_parquet::parquet::encoding::bitpacked::{Unpackable, Unpacked};
use polars_parquet::parquet::encoding::hybrid_rle::HybridRleDecoder;
use polars_parquet::parquet::encoding::{bitpacked, uleb128, Encoding};
use polars_parquet::parquet::error::{ParquetError, ParquetResult};
use polars_parquet::parquet::page::{split_buffer, DataPage, EncodedSplitBuffer};
use polars_parquet::parquet::read::levels::get_bit_width;
use polars_parquet::parquet::types::NativeType;

use super::dictionary::PrimitivePageDict;
use super::{hybrid_rle_iter, Array};

fn read_buffer<T: NativeType>(values: &[u8]) -> impl Iterator<Item = T> + '_ {
    let chunks = values.chunks_exact(std::mem::size_of::<T>());
    chunks.map(|chunk| {
        // unwrap is infalible due to the chunk size.
        let chunk: T::Bytes = match chunk.try_into() {
            Ok(v) => v,
            Err(_) => panic!(),
        };
        T::from_le_bytes(chunk)
    })
}

// todo: generalize i64 -> T
fn compose_array<I: Iterator<Item = u32>, F: Iterator<Item = u32>, G: Iterator<Item = i64>>(
    rep_levels: I,
    def_levels: F,
    max_rep: u32,
    max_def: u32,
    mut values: G,
) -> Result<Array, ParquetError> {
    let mut outer = vec![];
    let mut inner = vec![];

    assert_eq!(max_rep, 1);
    assert_eq!(max_def, 3);
    let mut prev_def = 0;
    rep_levels
        .into_iter()
        .zip(def_levels.into_iter())
        .try_for_each(|(rep, def)| {
            match rep {
                1 => {},
                0 => {
                    if prev_def > 1 {
                        let old = std::mem::take(&mut inner);
                        outer.push(Some(Array::Int64(old)));
                    }
                },
                _ => unreachable!(),
            }
            match def {
                3 => inner.push(Some(values.next().unwrap())),
                2 => inner.push(None),
                1 => outer.push(Some(Array::Int64(vec![]))),
                0 => outer.push(None),
                _ => unreachable!(),
            }
            prev_def = def;
            Ok::<(), ParquetError>(())
        })?;
    outer.push(Some(Array::Int64(inner)));
    Ok(Array::List(outer))
}

fn read_array_impl<I: Iterator<Item = i64>>(
    rep_levels: &[u8],
    def_levels: &[u8],
    values: I,
    length: usize,
    rep_level_encoding: (&Encoding, i16),
    def_level_encoding: (&Encoding, i16),
) -> Result<Array, ParquetError> {
    let max_rep_level = rep_level_encoding.1 as u32;
    let max_def_level = def_level_encoding.1 as u32;

    match (
        (rep_level_encoding.0, max_rep_level == 0),
        (def_level_encoding.0, max_def_level == 0),
    ) {
        ((Encoding::Rle, true), (Encoding::Rle, true)) => compose_array(
            std::iter::repeat(0).take(length),
            std::iter::repeat(0).take(length),
            max_rep_level,
            max_def_level,
            values,
        ),
        ((Encoding::Rle, false), (Encoding::Rle, true)) => {
            let num_bits = get_bit_width(rep_level_encoding.1);
            let rep_levels = HybridRleDecoder::new(rep_levels, num_bits, length);
            compose_array(
                hybrid_rle_iter(rep_levels)?,
                std::iter::repeat(0).take(length),
                max_rep_level,
                max_def_level,
                values,
            )
        },
        ((Encoding::Rle, true), (Encoding::Rle, false)) => {
            let num_bits = get_bit_width(def_level_encoding.1);
            let def_levels = HybridRleDecoder::new(def_levels, num_bits, length);
            compose_array(
                std::iter::repeat(0).take(length),
                hybrid_rle_iter(def_levels)?,
                max_rep_level,
                max_def_level,
                values,
            )
        },
        ((Encoding::Rle, false), (Encoding::Rle, false)) => {
            let rep_levels =
                HybridRleDecoder::new(rep_levels, get_bit_width(rep_level_encoding.1), length);
            let def_levels =
                HybridRleDecoder::new(def_levels, get_bit_width(def_level_encoding.1), length);
            compose_array(
                hybrid_rle_iter(rep_levels)?,
                hybrid_rle_iter(def_levels)?,
                max_rep_level,
                max_def_level,
                values,
            )
        },
        _ => todo!(),
    }
}

fn read_array(
    rep_levels: &[u8],
    def_levels: &[u8],
    values: &[u8],
    length: u32,
    rep_level_encoding: (&Encoding, i16),
    def_level_encoding: (&Encoding, i16),
) -> Result<Array, ParquetError> {
    let values = read_buffer::<i64>(values);
    read_array_impl::<_>(
        rep_levels,
        def_levels,
        values,
        length as usize,
        rep_level_encoding,
        def_level_encoding,
    )
}

pub fn page_to_array<T: NativeType>(
    page: &DataPage,
    dict: Option<&PrimitivePageDict<T>>,
) -> Result<Array, ParquetError> {
    let EncodedSplitBuffer {
        rep: rep_levels,
        def: def_levels,
        values,
    } = split_buffer(page)?;

    match (&page.encoding(), dict) {
        (Encoding::Plain, None) => read_array(
            rep_levels,
            def_levels,
            values,
            page.num_values() as u32,
            (
                &page.repetition_level_encoding(),
                page.descriptor.max_rep_level,
            ),
            (
                &page.definition_level_encoding(),
                page.descriptor.max_def_level,
            ),
        ),
        _ => todo!(),
    }
}

pub struct DecoderIter<'a, T: Unpackable> {
    pub(crate) decoder: bitpacked::Decoder<'a, T>,
    pub(crate) buffered: T::Unpacked,
    pub(crate) unpacked_start: usize,
    pub(crate) unpacked_end: usize,
}

impl<'a, T: Unpackable> Iterator for DecoderIter<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.unpacked_start >= self.unpacked_end {
            let length;
            (self.buffered, length) = self.decoder.chunked().next_inexact()?;
            debug_assert!(length > 0);
            self.unpacked_start = 1;
            self.unpacked_end = length;
            return Some(self.buffered[0]);
        }

        let v = self.buffered[self.unpacked_start];
        self.unpacked_start += 1;
        Some(v)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.decoder.len() + self.unpacked_end - self.unpacked_start;
        (len, Some(len))
    }
}

impl<'a, T: Unpackable> ExactSizeIterator for DecoderIter<'a, T> {}

impl<'a, T: Unpackable> DecoderIter<'a, T> {
    pub fn new(packed: &'a [u8], num_bits: usize, length: usize) -> ParquetResult<Self> {
        assert!(num_bits > 0);
        Ok(Self {
            decoder: bitpacked::Decoder::try_new(packed, num_bits, length)?,
            buffered: T::Unpacked::zero(),
            unpacked_start: 0,
            unpacked_end: 0,
        })
    }
}

fn read_dict_array(
    rep_levels: &[u8],
    def_levels: &[u8],
    values: &[u8],
    length: u32,
    dict: &PrimitivePageDict<i64>,
    rep_level_encoding: (&Encoding, i16),
    def_level_encoding: (&Encoding, i16),
) -> Result<Array, ParquetError> {
    let dict_values = dict.values();

    let bit_width = values[0];
    let values = &values[1..];

    let (_, consumed) = uleb128::decode(values);
    let values = &values[consumed..];

    let indices = DecoderIter::<u32>::new(values, bit_width as usize, length as usize)?;

    let values = indices.map(|id| dict_values[id as usize]);

    read_array_impl::<_>(
        rep_levels,
        def_levels,
        values,
        length as usize,
        rep_level_encoding,
        def_level_encoding,
    )
}

pub fn page_dict_to_array(
    page: &DataPage,
    dict: Option<&PrimitivePageDict<i64>>,
) -> Result<Array, ParquetError> {
    assert_eq!(page.descriptor.max_rep_level, 1);

    let EncodedSplitBuffer {
        rep: rep_levels,
        def: def_levels,
        values,
    } = split_buffer(page)?;

    match (page.encoding(), dict) {
        (Encoding::PlainDictionary, Some(dict)) => read_dict_array(
            rep_levels,
            def_levels,
            values,
            page.num_values() as u32,
            dict,
            (
                &page.repetition_level_encoding(),
                page.descriptor.max_rep_level,
            ),
            (
                &page.definition_level_encoding(),
                page.descriptor.max_def_level,
            ),
        ),
        (_, None) => Err(ParquetError::OutOfSpec(
            "A dictionary-encoded page MUST be preceded by a dictionary page".to_string(),
        )),
        _ => todo!(),
    }
}
