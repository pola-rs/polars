pub(crate) mod array_chunks;
pub(crate) mod dict_encoded;
pub(crate) mod filter;

use std::ops::Range;

use arrow::array::{DictionaryArray, DictionaryKey, PrimitiveArray};
use arrow::bitmap::{Bitmap, MutableBitmap};
use arrow::datatypes::ArrowDataType;
use arrow::pushable::Pushable;

use self::filter::Filter;
use super::BasicDecompressor;
use crate::parquet::encoding::hybrid_rle::{self, HybridRleChunk, HybridRleDecoder};
use crate::parquet::error::{ParquetError, ParquetResult};
use crate::parquet::page::{split_buffer, DataPage, DictPage};
use crate::parquet::schema::Repetition;

#[derive(Debug)]
pub(crate) struct State<'a, D: Decoder> {
    pub(crate) dict: Option<&'a D::Dict>,
    pub(crate) is_optional: bool,
    pub(crate) page_validity: Option<Bitmap>,
    pub(crate) translation: D::Translation<'a>,
}

pub(crate) trait StateTranslation<'a, D: Decoder>: Sized {
    type PlainDecoder;

    fn new(
        decoder: &D,
        page: &'a DataPage,
        dict: Option<&'a D::Dict>,
        page_validity: Option<&Bitmap>,
    ) -> ParquetResult<Self>;
}

impl<'a, D: Decoder> State<'a, D> {
    pub fn new(decoder: &D, page: &'a DataPage, dict: Option<&'a D::Dict>) -> ParquetResult<Self> {
        let is_optional =
            page.descriptor.primitive_type.field_info.repetition == Repetition::Optional;

        let mut page_validity = None;

        // Make the page_validity None if there are no nulls in the page
        if is_optional && !page.null_count().is_some_and(|nc| nc == 0) {
            let pv = page_validity_decoder(page)?;
            let pv = decode_page_validity(pv, None)?;

            if pv.unset_bits() > 0 {
                page_validity = Some(pv);
            }
        }

        let translation = D::Translation::new(decoder, page, dict, page_validity.as_ref())?;

        Ok(Self {
            dict,
            is_optional,
            page_validity,
            translation,
        })
    }

    pub fn new_nested(
        decoder: &D,
        page: &'a DataPage,
        dict: Option<&'a D::Dict>,
        mut page_validity: Option<Bitmap>,
    ) -> ParquetResult<Self> {
        let translation = D::Translation::new(decoder, page, dict, None)?;

        let is_optional =
            page.descriptor.primitive_type.field_info.repetition == Repetition::Optional;

        if page_validity
            .as_ref()
            .is_some_and(|bm| bm.unset_bits() == 0)
        {
            page_validity = None;
        }

        Ok(Self {
            dict,
            translation,
            is_optional,
            page_validity,
        })
    }

    pub fn decode(
        self,
        decoder: &mut D,
        decoded: &mut D::DecodedState,
        filter: Option<Filter>,
    ) -> ParquetResult<()> {
        decoder.extend_filtered_with_state(self, decoded, filter)
    }
}

pub fn not_implemented(page: &DataPage) -> ParquetError {
    let is_optional = page.descriptor.primitive_type.field_info.repetition == Repetition::Optional;
    let required = if is_optional { "optional" } else { "required" };
    ParquetError::not_supported(format!(
        "Decoding {:?} \"{:?}\"-encoded {required} parquet pages not yet supported",
        page.descriptor.primitive_type.physical_type,
        page.encoding(),
    ))
}

pub(crate) type PageValidity<'a> = HybridRleDecoder<'a>;
pub(crate) fn page_validity_decoder(page: &DataPage) -> ParquetResult<PageValidity> {
    let validity = split_buffer(page)?.def;
    let decoder = hybrid_rle::HybridRleDecoder::new(validity, 1, page.num_values());
    Ok(decoder)
}

pub(crate) fn unspecialized_decode<T: Default>(
    mut num_rows: usize,

    mut decode_one: impl FnMut() -> ParquetResult<T>,

    mut filter: Option<Filter>,
    page_validity: Option<Bitmap>,

    is_optional: bool,

    validity: &mut MutableBitmap,
    target: &mut impl Pushable<T>,
) -> ParquetResult<()> {
    match &filter {
        None => {},
        Some(Filter::Range(range)) => {
            match page_validity.as_ref() {
                None => {
                    for _ in 0..range.start {
                        decode_one()?;
                    }
                },
                Some(pv) => {
                    for _ in 0..pv.clone().sliced(0, range.start).set_bits() {
                        decode_one()?;
                    }
                },
            }

            num_rows = range.len();
            filter = None;
        },
        Some(Filter::Mask(mask)) => {
            if mask.unset_bits() == 0 {
                num_rows = mask.len();
                filter = None;
            }
        },
    };

    match (filter, page_validity) {
        (None, None) => {
            target.reserve(num_rows);
            for _ in 0..num_rows {
                target.push(decode_one()?);
            }

            if is_optional {
                validity.extend_constant(num_rows, true);
            }
        },
        (None, Some(page_validity)) => {
            target.reserve(page_validity.len());
            for is_valid in page_validity.iter() {
                let v = if is_valid {
                    decode_one()?
                } else {
                    T::default()
                };
                target.push(v);
            }

            validity.extend_from_bitmap(&page_validity);
        },
        (Some(Filter::Range(_)), _) => unreachable!(),
        (Some(Filter::Mask(mask)), None) => {
            let num_rows = mask.set_bits();
            target.reserve(num_rows);

            let mut iter = mask.iter();
            while iter.num_remaining() > 0 {
                let num_ones = iter.take_leading_ones();

                if num_ones > 0 {
                    for _ in 0..num_rows {
                        target.push(decode_one()?);
                    }
                }

                let num_zeros = iter.take_leading_zeros();
                for _ in 0..num_zeros {
                    decode_one()?;
                }
            }

            if is_optional {
                validity.extend_constant(num_rows, true);
            }
        },
        (Some(Filter::Mask(mask)), Some(page_validity)) => {
            assert_eq!(mask.len(), page_validity.len());

            let num_rows = mask.set_bits();
            target.reserve(num_rows);

            let mut mask_iter = mask.fast_iter_u56();
            let mut validity_iter = page_validity.fast_iter_u56();

            let mut iter = |mut f: u64, mut v: u64| {
                while f != 0 {
                    let offset = f.trailing_ones();

                    if (v >> offset) & 1 != 0 {
                        target.push(decode_one()?);
                    } else {
                        target.push(T::default());
                    }

                    let skip = (v & (1u64 << offset).wrapping_sub(1)).count_ones() as usize;
                    for _ in 0..skip {
                        decode_one()?;
                    }

                    v >>= offset + 1;
                    f >>= offset + 1;
                }

                for _ in 0..v.count_ones() as usize {
                    decode_one()?;
                }

                ParquetResult::Ok(())
            };

            for (f, v) in mask_iter.by_ref().zip(validity_iter.by_ref()) {
                iter(f, v)?;
            }

            let (f, fl) = mask_iter.remainder();
            let (v, vl) = validity_iter.remainder();

            assert_eq!(fl, vl);

            iter(f, v)?;

            validity.extend_from_bitmap(&page_validity);
        },
    }

    Ok(())
}

/// An item with a known size
pub(super) trait ExactSize {
    /// The number of items in the container
    fn len(&self) -> usize;
}

/// A decoder that knows how to map `State` -> Array
pub(super) trait Decoder: Sized {
    /// The state that this decoder derives from a [`DataPage`]. This is bound to the page.
    type Translation<'a>: StateTranslation<'a, Self>;
    /// The dictionary representation that the decoder uses
    type Dict: ExactSize;
    /// The target state that this Decoder decodes into.
    type DecodedState: ExactSize;

    type Output;

    /// Initializes a new [`Self::DecodedState`].
    fn with_capacity(&self, capacity: usize) -> Self::DecodedState;

    /// Deserializes a [`DictPage`] into [`Self::Dict`].
    fn deserialize_dict(&mut self, page: DictPage) -> ParquetResult<Self::Dict>;

    fn extend_filtered_with_state(
        &mut self,
        state: State<'_, Self>,
        decoded: &mut Self::DecodedState,
        filter: Option<Filter>,
    ) -> ParquetResult<()>;

    fn apply_dictionary(
        &mut self,
        _decoded: &mut Self::DecodedState,
        _dict: &Self::Dict,
    ) -> ParquetResult<()> {
        Ok(())
    }

    fn finalize(
        &self,
        dtype: ArrowDataType,
        dict: Option<Self::Dict>,
        decoded: Self::DecodedState,
    ) -> ParquetResult<Self::Output>;
}

pub trait DictDecodable: Decoder {
    fn finalize_dict_array<K: DictionaryKey>(
        &self,
        dtype: ArrowDataType,
        dict: Self::Dict,
        keys: PrimitiveArray<K>,
    ) -> ParquetResult<DictionaryArray<K>>;
}

pub struct PageDecoder<D: Decoder> {
    pub iter: BasicDecompressor,
    pub dtype: ArrowDataType,
    pub dict: Option<D::Dict>,
    pub decoder: D,
}

impl<D: Decoder> PageDecoder<D> {
    pub fn new(
        mut iter: BasicDecompressor,
        dtype: ArrowDataType,
        mut decoder: D,
    ) -> ParquetResult<Self> {
        let dict_page = iter.read_dict_page()?;
        let dict = dict_page.map(|d| decoder.deserialize_dict(d)).transpose()?;

        Ok(Self {
            iter,
            dtype,
            dict,
            decoder,
        })
    }

    pub fn collect_n(mut self, mut filter: Option<Filter>) -> ParquetResult<D::Output> {
        let mut num_rows_remaining = Filter::opt_num_rows(&filter, self.iter.total_num_values());

        let mut target = self.decoder.with_capacity(num_rows_remaining);

        if let Some(dict) = self.dict.as_ref() {
            self.decoder.apply_dictionary(&mut target, dict)?;
        }

        while num_rows_remaining > 0 {
            let Some(page) = self.iter.next() else {
                break;
            };
            let page = page?;

            let state_filter;
            (state_filter, filter) = Filter::opt_split_at(&filter, page.num_values());

            // Skip the whole page if we don't need any rows from it
            if state_filter.as_ref().is_some_and(|f| f.num_rows() == 0) {
                continue;
            }

            let page = page.decompress(&mut self.iter)?;

            let state = State::new(&self.decoder, &page, self.dict.as_ref())?;

            let start_length = target.len();
            state.decode(&mut self.decoder, &mut target, state_filter)?;
            let end_length = target.len();

            num_rows_remaining -= end_length - start_length;

            self.iter.reuse_page_buffer(page);
        }

        self.decoder.finalize(self.dtype, self.dict, target)
    }
}

#[inline]
pub(super) fn dict_indices_decoder(
    page: &DataPage,
    null_count: usize,
) -> ParquetResult<hybrid_rle::HybridRleDecoder> {
    let indices_buffer = split_buffer(page)?.values;

    // SPEC: Data page format: the bit width used to encode the entry ids stored as 1 byte (max bit width = 32),
    // SPEC: followed by the values encoded using RLE/Bit packed described above (with the given bit width).
    let bit_width = indices_buffer[0];
    let indices_buffer = &indices_buffer[1..];

    Ok(hybrid_rle::HybridRleDecoder::new(
        indices_buffer,
        bit_width as u32,
        page.num_values() - null_count,
    ))
}

/// Freeze a [`MutableBitmap`] into a `Option<Bitmap>`.
///
/// This will turn the several instances where `None` (representing "all valid") suffices.
pub fn freeze_validity(validity: MutableBitmap) -> Option<Bitmap> {
    if validity.is_empty() {
        return None;
    }

    let validity = validity.freeze();

    if validity.unset_bits() == 0 {
        return None;
    }

    Some(validity)
}

pub(crate) fn filter_from_range(rng: Range<usize>) -> Bitmap {
    let mut bm = MutableBitmap::with_capacity(rng.end);

    bm.extend_constant(rng.start, false);
    bm.extend_constant(rng.len(), true);

    bm.freeze()
}

pub(crate) fn decode_hybrid_rle_into_bitmap(
    mut page_validity: HybridRleDecoder<'_>,
    limit: Option<usize>,
    bitmap: &mut MutableBitmap,
) -> ParquetResult<()> {
    assert!(page_validity.num_bits() <= 1);

    let mut limit = limit.unwrap_or(page_validity.len());
    bitmap.reserve(limit);

    while let Some(chunk) = page_validity.next_chunk()? {
        if limit == 0 {
            break;
        }

        match chunk {
            HybridRleChunk::Rle(value, size) => {
                let size = size.min(limit);
                bitmap.extend_constant(size, value != 0);
                limit -= size;
            },
            HybridRleChunk::Bitpacked(decoder) => {
                let len = decoder.len().min(limit);
                bitmap.extend_from_slice(decoder.as_slice(), 0, len);
                limit -= len;
            },
        }
    }

    Ok(())
}

pub(crate) fn decode_page_validity(
    page_validity: HybridRleDecoder<'_>,
    limit: Option<usize>,
) -> ParquetResult<Bitmap> {
    let mut bm = MutableBitmap::new();
    decode_hybrid_rle_into_bitmap(page_validity, limit, &mut bm)?;
    Ok(bm.freeze())
}
