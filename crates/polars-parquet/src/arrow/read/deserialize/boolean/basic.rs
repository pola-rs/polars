use arrow::array::BooleanArray;
use arrow::bitmap::utils::BitmapIter;
use arrow::bitmap::MutableBitmap;
use arrow::datatypes::ArrowDataType;
use polars_error::PolarsResult;

use super::super::utils::{extend_from_decoder, DecodedState, Decoder};
use super::super::{utils, PagesIter};
use crate::parquet::encoding::hybrid_rle::gatherer::HybridRleGatherer;
use crate::parquet::encoding::hybrid_rle::HybridRleDecoder;
use crate::parquet::encoding::Encoding;
use crate::parquet::error::ParquetResult;
use crate::parquet::page::{split_buffer, DataPage, DictPage};
use crate::read::deserialize::utils::filter::Filter;
use crate::read::deserialize::utils::{BasicDecodeIterator, BatchableCollector, PageValidity};

#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
enum StateTranslation<'pages> {
    Unit(BitmapIter<'pages>),
    Rle(HybridRleDecoder<'pages>),
}

impl<'pages, 'mmap: 'pages> utils::StateTranslation<'pages, 'mmap, BooleanDecoder> for StateTranslation<'pages> {
    fn new(
        _decoder: &BooleanDecoder,
        page: &'pages DataPage<'mmap>,
        _dict: Option<&'pages <BooleanDecoder as Decoder<'pages, 'mmap>>::Dict>,
        page_validity: Option<&PageValidity<'pages>>,
        _filter: Option<&Filter<'pages>>,
    ) -> PolarsResult<Self> {
        let values = split_buffer(page)?.values;

        match page.encoding() {
            Encoding::Plain => {
                let num_values = if page_validity.is_some() {
                    // @NOTE: We overestimate the amount of values here, but in the V1
                    // specification we don't really have a way to know the number of valid items.
                    // Without traversing the list.
                    values.len() * u8::BITS as usize
                } else {
                    page.num_values()
                };
                Ok(Self::Unit(BitmapIter::new(values, 0, num_values)))
            },
            Encoding::Rle => {
                // @NOTE: For a nullable list, we might very well overestimate the amount of
                // values, but we never collect those items. We don't really have a way to now the
                // number of valid items in the V1 specification.

                // For RLE boolean values the length in bytes is pre-pended.
                // https://github.com/apache/parquet-format/blob/e517ac4dbe08d518eb5c2e58576d4c711973db94/Encodings.md#run-length-encoding--bit-packing-hybrid-rle--3
                let (_len_in_bytes, values) = values.split_at(4);
                Ok(Self::Rle(HybridRleDecoder::new(
                    values,
                    1,
                    page.num_values(),
                )))
            },
            _ => Err(utils::not_implemented(page)),
        }
    }

    fn len_when_not_nullable(&self) -> usize {
        match self {
            Self::Unit(v) => v.len(),
            Self::Rle(v) => v.len(),
        }
    }

    fn skip_in_place(&mut self, n: usize) -> ParquetResult<()> {
        if n == 0 {
            return Ok(());
        }

        // @TODO: Add a skip_in_place on BitmapIter
        match self {
            Self::Unit(t) => _ = t.nth(n - 1),
            Self::Rle(t) => t.skip_in_place(n)?,
        }

        Ok(())
    }

    fn extend_from_state(
        &mut self,
        _decoder: &BooleanDecoder,
        decoded: &mut <BooleanDecoder as Decoder>::DecodedState,
        page_validity: &mut Option<PageValidity<'pages>>,
        additional: usize,
    ) -> ParquetResult<()> {
        let (values, validity) = decoded;

        match (self, page_validity) {
            (Self::Unit(page), None) => page.collect_n_into(values, additional),
            (Self::Unit(page_values), Some(page_validity)) => extend_from_decoder(
                validity,
                page_validity,
                Some(additional),
                values,
                page_values,
            )?,
            (Self::Rle(page_values), None) => {
                page_values.gather_n_into(values, additional, &BitmapGatherer)?
            },
            (Self::Rle(page_values), Some(page_validity)) => utils::extend_from_decoder(
                validity,
                page_validity,
                Some(additional),
                values,
                BitmapCollector(page_values),
            )?,
        }

        Ok(())
    }
}

struct BitmapGatherer;
impl HybridRleGatherer<u32> for BitmapGatherer {
    type Target = MutableBitmap;

    fn target_reserve(&self, target: &mut Self::Target, n: usize) {
        target.reserve(n);
    }

    fn target_num_elements(&self, target: &Self::Target) -> usize {
        target.len()
    }

    fn hybridrle_to_target(&self, value: u32) -> ParquetResult<u32> {
        Ok(value)
    }

    fn gather_one(&self, target: &mut Self::Target, value: u32) -> ParquetResult<()> {
        target.push(value != 0);
        Ok(())
    }

    fn gather_repeated(
        &self,
        target: &mut Self::Target,
        value: u32,
        n: usize,
    ) -> ParquetResult<()> {
        target.extend_constant(n, value != 0);
        Ok(())
    }

    // @TODO: The slice impl here can speed some stuff up
}
struct BitmapCollector<'a, 'b>(&'b mut HybridRleDecoder<'a>);
impl<'a, 'b> BatchableCollector<u32, MutableBitmap> for BitmapCollector<'a, 'b> {
    fn reserve(target: &mut MutableBitmap, n: usize) {
        target.reserve(n);
    }

    fn push_n(&mut self, target: &mut MutableBitmap, n: usize) -> ParquetResult<()> {
        self.0.gather_n_into(target, n, &BitmapGatherer)
    }

    fn push_n_nulls(&mut self, target: &mut MutableBitmap, n: usize) -> ParquetResult<()> {
        target.extend_constant(n, false);
        Ok(())
    }
}

impl DecodedState for (MutableBitmap, MutableBitmap) {
    fn len(&self) -> usize {
        self.0.len()
    }
}

struct BooleanDecoder;

impl<'pages, 'mmap: 'pages> Decoder<'pages, 'mmap> for BooleanDecoder {
    type Translation = StateTranslation<'pages>;
    type Dict = ();
    type DecodedState = (MutableBitmap, MutableBitmap);

    fn with_capacity(&self, capacity: usize) -> Self::DecodedState {
        (
            MutableBitmap::with_capacity(capacity),
            MutableBitmap::with_capacity(capacity),
        )
    }

    fn deserialize_dict(&self, _: &DictPage) -> Self::Dict {}
}

fn finish(
    data_type: &ArrowDataType,
    (values, validity): (MutableBitmap, MutableBitmap),
) -> PolarsResult<BooleanArray> {
    Ok(BooleanArray::new(
        data_type.clone(),
        values.into(),
        validity.into(),
    ))
}

pub struct BooleanDecodeIter;

impl BooleanDecodeIter {
    pub fn new<'pages, 'mmap: 'pages, I: PagesIter<'mmap>>(
        iter: I,
        data_type: ArrowDataType,
        chunk_size: Option<usize>,
        num_rows: usize,
    ) -> BasicDecodeIterator<
        'pages,
        'mmap,
        BooleanArray,
        I,
        BooleanDecoder,
        fn(
            &ArrowDataType,
            <BooleanDecoder as utils::Decoder<'pages, 'mmap>>::DecodedState,
        ) -> PolarsResult<BooleanArray>,
    > {
        BasicDecodeIterator::new(
            iter,
            data_type,
            chunk_size,
            num_rows,
            BooleanDecoder,
            finish,
        )
    }
}
