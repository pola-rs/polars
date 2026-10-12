use polars_arrow::array::{PrimitiveArray, Splitable};
use polars_arrow::bitmap::{Bitmap, BitmapBuilder};
use polars_arrow::datatypes::ArrowDataType;
use polars_arrow::types::NativeType;
use polars_utils::vec::with_cast_mut_vec;

use super::super::utils;
use super::plain::{decode_masked_optional, decode_masked_required, decode_optional};
use crate::parquet::encoding::{Encoding, hybrid_rle};
use crate::parquet::error::{ParquetError, ParquetResult};
use crate::parquet::page::{DataPage, DictPage, split_buffer};
use crate::read::Filter;
use crate::read::deserialize::dictionary_encoded::{
    self, append_validity, constrain_page_validity,
};
use crate::read::deserialize::utils::array_chunks::ArrayChunks;
use crate::read::deserialize::utils::{Decoded, dict_indices_decoder, freeze_validity};
use crate::read::expr::{ParquetScalar, SpecializedParquetColumnExpr};

type I128Bytes = <i128 as NativeType>::AlignedBytes;

#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub(crate) enum StateTranslation<'a> {
    Plain(&'a [u8], usize),
    Dictionary(hybrid_rle::HybridRleDecoder<'a>),
}

impl<'a> utils::StateTranslation<'a, FixedLenDecimalDecoder> for StateTranslation<'a> {
    type PlainDecoder = &'a [u8];

    fn new(
        decoder: &FixedLenDecimalDecoder,
        page: &'a DataPage,
        dict: Option<&'a <FixedLenDecimalDecoder as utils::Decoder>::Dict>,
        page_validity: Option<&Bitmap>,
    ) -> ParquetResult<Self> {
        match (page.encoding(), dict) {
            (Encoding::Plain, _) => {
                let mut values = split_buffer(page)?.values;
                if values.len() % decoder.size != 0 {
                    return Err(ParquetError::oos(format!(
                        "Fixed size binary data length {} is not divisible by size {}",
                        values.len(),
                        decoder.size
                    )));
                }
                // Trailing padding is not data.
                let max_len = page.num_values() * decoder.size;
                if values.len() > max_len {
                    values = &values[..max_len];
                }
                Ok(Self::Plain(values, decoder.size))
            },
            (Encoding::PlainDictionary | Encoding::RleDictionary, Some(_)) => {
                let values =
                    dict_indices_decoder(page, page_validity.map_or(0, |bm| bm.unset_bits()))?;
                Ok(Self::Dictionary(values))
            },
            _ => Err(utils::not_implemented(page)),
        }
    }

    fn num_rows(&self) -> usize {
        match self {
            Self::Plain(v, n) => v.len() / n,
            Self::Dictionary(i) => i.len(),
        }
    }
}

/// Decodes `FIXED_LEN_BYTE_ARRAY(size)` decimals (big-endian two's complement, `size <= 16`)
/// straight into `i128`. The dictionary is converted once.
#[derive(Debug)]
pub(crate) struct FixedLenDecimalDecoder {
    size: usize,
    intermediate: Vec<i128>,
}

impl FixedLenDecimalDecoder {
    pub(crate) fn new(size: usize) -> Self {
        assert!((1..=16).contains(&size));
        Self {
            size,
            intermediate: Vec::new(),
        }
    }
}

#[inline(always)]
fn be_to_i128<const N: usize>(bytes: &[u8; N]) -> i128 {
    let mut buf = [0u8; 16];
    buf[..N].copy_from_slice(bytes);
    // Arithmetic shift to sign-extend.
    i128::from_be_bytes(buf) >> (8 * (16 - N))
}

/// Calls `$f::<N>($args)` with `N` equal to the runtime value `$size`, `1 <= N <= 16`.
macro_rules! dispatch_size {
    ($size:expr, $f:ident($($arg:expr),*)) => {
        match $size {
            1 => $f::<1>($($arg),*),
            2 => $f::<2>($($arg),*),
            3 => $f::<3>($($arg),*),
            4 => $f::<4>($($arg),*),
            5 => $f::<5>($($arg),*),
            6 => $f::<6>($($arg),*),
            7 => $f::<7>($($arg),*),
            8 => $f::<8>($($arg),*),
            9 => $f::<9>($($arg),*),
            10 => $f::<10>($($arg),*),
            11 => $f::<11>($($arg),*),
            12 => $f::<12>($($arg),*),
            13 => $f::<13>($($arg),*),
            14 => $f::<14>($($arg),*),
            15 => $f::<15>($($arg),*),
            16 => $f::<16>($($arg),*),
            n => panic!("Decimal128 value has {n} bytes, expected 1 to 16"),
        }
    };
}

/// Sign-extends a big-endian two's complement integer of at most 16 bytes.
pub(crate) fn convert_i128(value: &[u8]) -> i128 {
    fn one<const N: usize>(value: &[u8]) -> i128 {
        be_to_i128::<N>(value.try_into().unwrap())
    }
    if value.is_empty() {
        return 0;
    }
    dispatch_size!(value.len(), one(value))
}

/// Appends the big-endian `size`-byte values in `values` to `target` as `i128`.
fn convert(size: usize, values: &[u8], target: &mut Vec<i128>) {
    fn all<const N: usize>(values: &[u8], target: &mut Vec<i128>) {
        let (chunks, rest) = values.as_chunks::<N>();
        debug_assert!(rest.is_empty());
        target.extend(chunks.iter().map(be_to_i128::<N>));
    }
    target.reserve(values.len() / size);
    dispatch_size!(size, all(values, target))
}

/// The bytes of values `start..end`, or an error when the page is too short.
fn value_bytes(size: usize, values: &[u8], start: usize, end: usize) -> ParquetResult<&[u8]> {
    values
        .get(start * size..end * size)
        .ok_or_else(|| ParquetError::oos("Fixed size binary page has fewer values than expected"))
}

impl utils::Decoder for FixedLenDecimalDecoder {
    type Translation<'a> = StateTranslation<'a>;
    type Dict = PrimitiveArray<i128>;
    type DecodedState = (Vec<i128>, BitmapBuilder);
    type Output = PrimitiveArray<i128>;

    fn with_capacity(&self, capacity: usize) -> Self::DecodedState {
        (
            Vec::<i128>::with_capacity(capacity),
            BitmapBuilder::with_capacity(capacity),
        )
    }

    fn deserialize_dict(&mut self, page: DictPage) -> ParquetResult<Self::Dict> {
        let values = page.buffer.as_ref();
        if values.len() % self.size != 0 {
            return Err(ParquetError::oos(format!(
                "Fixed size binary dictionary length {} is not divisible by size {}",
                values.len(),
                self.size
            )));
        }
        let mut target = Vec::new();
        convert(self.size, values, &mut target);
        Ok(PrimitiveArray::new(
            i128::PRIMITIVE.into(),
            target.into(),
            None,
        ))
    }

    fn evaluate_predicate(
        &mut self,
        state: &utils::State<'_, Self>,
        _predicate: Option<&SpecializedParquetColumnExpr>,
        pred_true_mask: &mut BitmapBuilder,
        dict_mask: Option<&Bitmap>,
    ) -> ParquetResult<bool> {
        if state.page_validity.is_some() {
            return Ok(false);
        }

        if let StateTranslation::Dictionary(values) = &state.translation {
            dictionary_encoded::predicate::decode(
                values.clone(),
                dict_mask.unwrap(),
                pred_true_mask,
            )?;
            return Ok(true);
        }

        Ok(false)
    }

    fn finalize(
        &self,
        dtype: ArrowDataType,
        _dict: Option<Self::Dict>,
        (values, validity): Self::DecodedState,
    ) -> ParquetResult<Self::Output> {
        let validity = freeze_validity(validity);
        Ok(PrimitiveArray::try_new(dtype, values.into(), validity).unwrap())
    }

    fn extend_decoded(
        &self,
        decoded: &mut Self::DecodedState,
        additional: &dyn polars_arrow::array::Array,
        is_optional: bool,
    ) -> ParquetResult<()> {
        let additional = additional
            .as_any()
            .downcast_ref::<PrimitiveArray<i128>>()
            .unwrap();
        decoded.0.extend(additional.values().iter().copied());
        match additional.validity() {
            Some(v) => decoded.1.extend_from_bitmap(v),
            None if is_optional => decoded.1.extend_constant(additional.len(), true),
            None => {},
        }

        Ok(())
    }

    fn extend_constant(
        &mut self,
        decoded: &mut Self::DecodedState,
        length: usize,
        value: &ParquetScalar,
    ) -> ParquetResult<()> {
        if value.is_null() {
            decoded.extend_nulls(length);
            return Ok(());
        }

        let value = value.as_binary().unwrap();
        assert_eq!(value.len(), self.size);
        decoded
            .0
            .extend(std::iter::repeat_n(convert_i128(value), length));
        decoded.1.extend_constant(length, true);

        Ok(())
    }

    fn extend_filtered_with_state(
        &mut self,
        state: utils::State<'_, Self>,
        decoded: &mut Self::DecodedState,
        filter: Option<Filter>,
        _chunks: &mut Vec<Self::Output>,
    ) -> ParquetResult<()> {
        match state.translation {
            StateTranslation::Plain(values, _) => {
                let size = self.size;
                let num_values = values.len() / size;
                let page_validity = state.page_validity.as_ref();

                if state.is_optional {
                    append_validity(page_validity, filter.as_ref(), &mut decoded.1, num_values);
                }
                let page_validity =
                    constrain_page_validity(num_values, page_validity, filter.as_ref());

                // Without nulls, unfiltered and range-filtered values are converted straight
                // into the output. Otherwise only the values between the first and last selected
                // row are converted, and the aligned-bytes kernels select them and place them
                // between the nulls.
                let (start, num_rows, page_validity, filter) = match (filter, page_validity) {
                    (None, None) => {
                        convert(size, values, &mut decoded.0);
                        return Ok(());
                    },
                    (Some(Filter::Range(rng)), None) => {
                        let values = value_bytes(size, values, rng.start, rng.end)?;
                        convert(size, values, &mut decoded.0);
                        return Ok(());
                    },
                    (Some(Filter::Mask(mut mask)), None) => {
                        let start = mask.take_leading_zeros();
                        mask.take_trailing_zeros();
                        let values = value_bytes(size, values, start, start + mask.len())?;
                        self.intermediate.clear();
                        convert(size, values, &mut self.intermediate);
                        let values = ArrayChunks::<'_, I128Bytes>::new(bytemuck::cast_slice(
                            &self.intermediate,
                        ))
                        .unwrap();
                        return with_cast_mut_vec::<i128, I128Bytes, _, _>(
                            &mut decoded.0,
                            |target| decode_masked_required(values, mask, target),
                        );
                    },
                    (None, Some(pv)) => (0, pv.len(), pv, None),
                    (Some(Filter::Range(rng)), Some(pv)) => (rng.start, rng.len(), pv, None),
                    (Some(Filter::Mask(mut mask)), Some(pv)) => {
                        let start = mask.take_leading_zeros();
                        mask.take_trailing_zeros();
                        (start, mask.len(), pv, Some(mask))
                    },
                    (Some(Filter::Predicate(_)), _) => unreachable!(),
                };

                let (prevalidity, mut page_validity) = page_validity.split_at(start);
                page_validity.slice(0, num_rows);
                let values_start = prevalidity.set_bits();
                let values_end = values_start + page_validity.set_bits();
                let values = value_bytes(size, values, values_start, values_end)?;

                self.intermediate.clear();
                convert(size, values, &mut self.intermediate);
                let values =
                    ArrayChunks::<'_, I128Bytes>::new(bytemuck::cast_slice(&self.intermediate))
                        .unwrap();

                with_cast_mut_vec::<i128, I128Bytes, _, _>(&mut decoded.0, |target| match filter {
                    None => decode_optional(values, page_validity, target),
                    Some(mask) => decode_masked_optional(values, page_validity, mask, target),
                })
            },
            StateTranslation::Dictionary(ref indexes) => dictionary_encoded::decode_dict(
                indexes.clone(),
                state.dict.unwrap().values().as_slice(),
                state.is_optional,
                state.page_validity.as_ref(),
                filter,
                &mut decoded.1,
                &mut decoded.0,
            ),
        }
    }
}
