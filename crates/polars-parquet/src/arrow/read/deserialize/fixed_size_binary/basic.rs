use arrow::array::FixedSizeBinaryArray;
use arrow::bitmap::MutableBitmap;
use arrow::datatypes::ArrowDataType;
use arrow::pushable::Pushable;
use polars_error::PolarsResult;
use polars_utils::iter::FallibleIterator;

use super::super::utils::{
    dict_indices_decoder, extend_from_decoder, not_implemented, DecodedState, Decoder,
};
use super::super::PagesIter;
use super::utils::FixedSizeBinary;
use crate::parquet::encoding::hybrid_rle::HybridRleDecoder;
use crate::parquet::encoding::Encoding;
use crate::parquet::error::{ParquetError, ParquetResult};
use crate::parquet::page::{split_buffer, DataPage, DictPage};
use crate::read::deserialize::utils::filter::Filter;
use crate::read::deserialize::utils::{self, BasicDecodeIterator, PageValidity};

#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
enum StateTranslation<'a> {
    Unit(std::slice::ChunksExact<'a, u8>),
    Dictionary(HybridRleDecoder<'a>, &'a [u8]),
}

impl<'pages, 'mmap: 'pages> utils::StateTranslation<'pages, 'mmap, BinaryDecoder> for StateTranslation<'pages> {
    fn new(
        decoder: &BinaryDecoder,
        page: &'pages DataPage<'mmap>,
        dict: Option<&'pages <BinaryDecoder as Decoder<'pages, 'mmap>>::Dict>,
        _page_validity: Option<&PageValidity<'pages>>,
        _filter: Option<&Filter<'pages>>,
    ) -> PolarsResult<Self> {
        match (page.encoding(), dict) {
            (Encoding::Plain, _) => {
                let values = split_buffer(page)?.values;
                if values.len() % decoder.size != 0 {
                    return Err(ParquetError::oos(format!(
                        "Fixed size binary data length {} is not divisible by size {}",
                        values.len(),
                        decoder.size
                    ))
                    .into());
                }
                let values = values.chunks_exact(decoder.size);
                Ok(Self::Unit(values))
            },
            (Encoding::PlainDictionary | Encoding::RleDictionary, Some(dict)) => {
                let values = dict_indices_decoder(page)?;
                Ok(Self::Dictionary(values, dict))
            },
            _ => Err(not_implemented(page)),
        }
    }

    fn len_when_not_nullable(&self) -> usize {
        match self {
            Self::Unit(v) => v.len(),
            Self::Dictionary(v, _) => v.len(),
        }
    }

    fn skip_in_place(&mut self, n: usize) -> ParquetResult<()> {
        if n == 0 {
            return Ok(());
        }

        match self {
            Self::Unit(v) => _ = v.nth(n - 1),
            Self::Dictionary(v, _) => v.skip_in_place(n)?,
        }

        Ok(())
    }

    fn extend_from_state(
        &mut self,
        decoder: &BinaryDecoder,
        decoded: &mut <BinaryDecoder as Decoder<'pages, 'mmap>>::DecodedState,
        page_validity: &mut Option<PageValidity<'pages>>,
        additional: usize,
    ) -> ParquetResult<()> {
        let (values, validity) = decoded;

        use StateTranslation as T;
        match (self, page_validity) {
            (T::Unit(page_values), None) => {
                // @TODO: This can be done through a extend
                for x in page_values.by_ref().take(additional) {
                    values.push(x)
                }
            },
            (T::Unit(page_values), Some(page_validity)) => extend_from_decoder(
                validity,
                page_validity,
                Some(additional),
                values,
                page_values,
            )?,
            (T::Dictionary(page_values, dict), None) => {
                // @TODO: Use Gatherer
                for x in page_values
                    .by_ref()
                    .map(|index| {
                        let index = index as usize;
                        &dict[index * decoder.size..(index + 1) * decoder.size]
                    })
                    .take(additional)
                {
                    values.push(x)
                }
                page_values.get_result()?;
            },
            (T::Dictionary(page_values, dict), Some(page_validity)) => {
                extend_from_decoder(
                    validity,
                    page_validity,
                    Some(additional),
                    values,
                    page_values.by_ref().map(|index| {
                        let index = index as usize;
                        &dict[index * decoder.size..(index + 1) * decoder.size]
                    }),
                )?;
                page_values.get_result()?;
            },
        }

        Ok(())
    }
}

struct BinaryDecoder {
    size: usize,
}

impl DecodedState for (FixedSizeBinary, MutableBitmap) {
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl<'pages, 'mmap: 'pages> Decoder<'pages, 'mmap> for BinaryDecoder {
    type Translation = StateTranslation<'pages>;
    type Dict = &'pages [u8];
    type DecodedState = (FixedSizeBinary, MutableBitmap);

    fn with_capacity(&self, capacity: usize) -> Self::DecodedState {
        (
            FixedSizeBinary::with_capacity(capacity, self.size),
            MutableBitmap::with_capacity(capacity),
        )
    }

    fn deserialize_dict(&self, page: &'pages DictPage) -> Self::Dict {
        page.buffer.as_ref()
    }
}

pub fn finish(
    data_type: &ArrowDataType,
    (values, validity): (FixedSizeBinary, MutableBitmap),
) -> PolarsResult<FixedSizeBinaryArray> {
    Ok(FixedSizeBinaryArray::new(
        data_type.clone(),
        values.values.into(),
        validity.into(),
    ))
}

pub struct FixedSizeBinaryDecodeIter;

impl FixedSizeBinaryDecodeIter {
    pub fn new<'pages, 'mmap: 'pages, I: PagesIter<'mmap>>(
        iter: I,
        data_type: ArrowDataType,
        num_rows: usize,
        chunk_size: Option<usize>,
    ) -> BasicDecodeIterator<
        'pages,
        'mmap,
        FixedSizeBinaryArray,
        I,
        BinaryDecoder,
        fn(
            &ArrowDataType,
            <BinaryDecoder as utils::Decoder<'pages, 'mmap>>::DecodedState,
        ) -> PolarsResult<FixedSizeBinaryArray>,
    > {
        let size = FixedSizeBinaryArray::get_size(&data_type);
        BasicDecodeIterator::new(
            iter,
            data_type,
            chunk_size,
            num_rows,
            BinaryDecoder { size },
            finish,
        )
    }
}
