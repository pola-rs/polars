//! Decoding a dictionary-encoded column chunk of any type as codes into its dictionary.
//!
//! [`CodesDecoder`] wraps the decoder the chunk would be read flat with: it deserializes the
//! dictionary page with it (so the values are as the flat read gives them, with the casts,
//! decimal conversions and timestamp units of that decoder) and decodes the data pages as indexes
//! into it, as the [`CategoricalDecoder`] does for strings.
//!
//! [`CategoricalDecoder`]: super::categorical::CategoricalDecoder

use polars_arrow::array::{Array, DictionaryArray, FixedSizeBinaryArray, PrimitiveArray};
use polars_arrow::bitmap::{Bitmap, BitmapBuilder};
use polars_arrow::datatypes::{ArrowDataType, Field, IntegerType, TimeUnit};
use polars_error::PolarsResult;
use polars_utils::IdxSize;
use polars_utils::float16::pf16;
use polars_utils::vec::with_cast_mut_vec;

use super::simple::{int96_to_i64_ms, int96_to_i64_s, int96_to_i64_us};
use super::utils::{
    self, Decoder, DictPredicate, StateTranslation, dict_indices_decoder, freeze_validity,
};
use super::{PredicateFilter, binary, categorical, fixed_size_binary, primitive};
use crate::parquet::encoding::Encoding;
use crate::parquet::encoding::hybrid_rle::HybridRleDecoder;
use crate::parquet::error::ParquetResult;
use crate::parquet::page::{DataPage, DictPage};
use crate::parquet::schema::types::{PhysicalType, PrimitiveType};
use crate::parquet::types::int96_to_i64_ns;
use crate::read::ParquetError;
use crate::read::expr::{ParquetScalar, SpecializedParquetColumnExpr};

#[derive(Clone)]
pub struct CodesTranslation<'a>(HybridRleDecoder<'a>);

impl<'a, D: Decoder> StateTranslation<'a, CodesDecoder<D>> for CodesTranslation<'a> {
    type PlainDecoder = HybridRleDecoder<'a>;

    fn new(
        _decoder: &CodesDecoder<D>,
        page: &'a DataPage,
        _dict: Option<&'a D::Dict>,
        page_validity: Option<&Bitmap>,
    ) -> ParquetResult<Self> {
        match page.encoding() {
            Encoding::PlainDictionary | Encoding::RleDictionary => Ok(Self(dict_indices_decoder(
                page,
                page_validity.map_or(0, |bm| bm.unset_bits()),
            )?)),
            // The page encoding stats said every page is dictionary-encoded, and they lied.
            _ => Err(ParquetError::NotDictionaryEncoded),
        }
    }

    fn num_rows(&self) -> usize {
        self.0.len()
    }
}

/// Decodes the data pages of a fully dictionary-encoded column chunk as `u32` codes, with the
/// dictionary page deserialized by `inner`.
pub struct CodesDecoder<D: Decoder> {
    inner: D,
    dict_size: usize,
}

impl<D: Decoder> CodesDecoder<D> {
    fn new(inner: D) -> Self {
        Self {
            inner,
            dict_size: usize::MAX,
        }
    }
}

impl<D: Decoder> Decoder for CodesDecoder<D> {
    type Translation<'a> = CodesTranslation<'a>;
    type Dict = D::Dict;
    type DecodedState = (Vec<u32>, BitmapBuilder);
    type Output = DictionaryArray<u32>;

    fn with_capacity(&self, capacity: usize) -> Self::DecodedState {
        (
            Vec::with_capacity(capacity),
            BitmapBuilder::with_capacity(capacity),
        )
    }

    fn evaluate_dict_predicate(
        &self,
        dict: &Self::Dict,
        predicate: &PredicateFilter,
    ) -> PolarsResult<Bitmap> {
        self.inner.evaluate_dict_predicate(dict, predicate)
    }

    fn evaluate_predicate(
        &mut self,
        state: &utils::State<'_, Self>,
        predicate: Option<&SpecializedParquetColumnExpr>,
        pred_true_mask: &mut BitmapBuilder,
        dict_mask: Option<&Bitmap>,
    ) -> ParquetResult<bool> {
        // An equality filter would need the code of its constant.
        if matches!(predicate, Some(SpecializedParquetColumnExpr::Equal(v)) if !v.is_null()) {
            return Ok(false);
        }
        if state.page_validity.is_some() {
            // @Performance: implement validity aware
            return Ok(false);
        }
        let Some(dict_mask) = dict_mask else {
            return Ok(false);
        };
        super::dictionary_encoded::predicate::decode(
            state.translation.0.clone(),
            dict_mask,
            pred_true_mask,
        )?;
        Ok(true)
    }

    fn unspecialized_predicate_decode(
        &mut self,
        state: utils::State<'_, Self>,
        decoded: &mut Self::DecodedState,
        pred_true_mask: &mut BitmapBuilder,
        predicate: &PredicateFilter,
        dict: Option<Self::Dict>,
        dict_predicate: Option<&DictPredicate>,
        _dtype: &ArrowDataType,
    ) -> ParquetResult<()> {
        let is_optional = state.is_optional;
        let mut page = self.with_capacity(StateTranslation::<Self>::num_rows(&state.translation));
        self.extend_filtered_with_state(state, &mut page, None, &mut Vec::new())?;
        let (keys, validity) = page;
        let validity = is_optional.then(|| validity.freeze());
        let is_valid = |i: usize| validity.as_ref().is_none_or(|v| v.get_bit(i));

        let dict_mask = match dict_predicate {
            Some(DictPredicate::Mask(mask)) => Some(mask),
            _ => None,
        };

        // The predicate result for each valid row, if the dictionary's did not decide it.
        let row_mask = match (dict_mask, &dict) {
            (Some(_), _) => None,
            (None, Some(dict)) => {
                let idx = keys
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| is_valid(*i))
                    .map(|(_, k)| *k as IdxSize)
                    .collect::<Vec<_>>();
                // SAFETY: the decoder checked every key against the dictionary's length.
                let values = unsafe {
                    polars_compute::gather::take_unchecked(
                        dict,
                        &polars_arrow::datatypes::IdxArr::from_vec(idx),
                    )
                };
                Some(
                    predicate
                        .predicate
                        .try_evaluate(values.as_ref())
                        .map_err(|e| ParquetError::InvalidParameter(e.to_string()))?,
                )
            },
            (None, None) => None,
        };

        let null_is_true = validity.as_ref().is_some_and(|v| v.unset_bits() > 0)
            && predicate
                .predicate
                .evaluate_null()
                .map_err(|e| ParquetError::InvalidParameter(e.to_string()))?;
        let start_len = pred_true_mask.len();
        let mut num_valid = 0;
        for (i, key) in keys.iter().enumerate() {
            let keep = if !is_valid(i) {
                null_is_true
            } else if let Some(row_mask) = &row_mask {
                num_valid += 1;
                row_mask.get_bit(num_valid - 1)
            } else {
                // SAFETY: the mask has one bit per dictionary value, and keys are in bounds.
                unsafe { dict_mask.unwrap().get_bit_unchecked(*key as usize) }
            };
            pred_true_mask.push(keep);
            if keep && predicate.include_values {
                decoded.0.push(*key);
                if is_optional {
                    decoded.1.push(is_valid(i));
                }
            }
        }
        debug_assert_eq!(pred_true_mask.len() - start_len, keys.len());
        Ok(())
    }

    fn deserialize_dict(&mut self, page: DictPage) -> ParquetResult<Self::Dict> {
        let dict = self.inner.deserialize_dict(page)?;
        self.dict_size = dict.len();
        Ok(dict)
    }

    fn apply_dictionary(
        &mut self,
        _decoded: &mut Self::DecodedState,
        _dict: &Self::Dict,
    ) -> ParquetResult<()> {
        Ok(())
    }

    fn extend_decoded(
        &self,
        decoded: &mut Self::DecodedState,
        additional: &dyn Array,
        is_optional: bool,
    ) -> ParquetResult<()> {
        let additional = additional
            .as_any()
            .downcast_ref::<DictionaryArray<u32>>()
            .unwrap();
        decoded.0.extend(additional.keys().values().iter().copied());
        match additional.validity() {
            Some(v) => decoded.1.extend_from_bitmap(v),
            None if is_optional => decoded.1.extend_constant(additional.len(), true),
            None => {},
        }
        Ok(())
    }

    fn extend_filtered_with_state(
        &mut self,
        state: utils::State<'_, Self>,
        decoded: &mut Self::DecodedState,
        filter: Option<super::Filter>,
        _chunks: &mut Vec<Self::Output>,
    ) -> ParquetResult<()> {
        with_cast_mut_vec::<u32, <u32 as polars_arrow::types::NativeType>::AlignedBytes, _, _>(
            &mut decoded.0,
            |codes| {
                super::dictionary_encoded::decode_dict_dispatch(
                    state.translation.0,
                    u32::try_from(self.dict_size).ok().unwrap(),
                    state.is_optional,
                    state.page_validity.as_ref(),
                    filter,
                    &mut decoded.1,
                    codes,
                )
            },
        )
    }

    fn extend_constant(
        &mut self,
        decoded: &mut Self::DecodedState,
        length: usize,
        value: &ParquetScalar,
    ) -> ParquetResult<()> {
        // `evaluate_predicate` does not take equality filters on a constant.
        if !value.is_null() {
            return Err(ParquetError::not_supported(
                "dictionary codes with a non-null equality filter",
            ));
        }
        decoded.0.resize(decoded.0.len() + length, 0);
        decoded.1.extend_constant(length, false);
        Ok(())
    }

    fn finalize(
        &self,
        _dtype: ArrowDataType,
        dict: Option<Self::Dict>,
        (codes, validity): Self::DecodedState,
    ) -> ParquetResult<DictionaryArray<u32>> {
        let dict =
            dict.ok_or_else(|| ParquetError::oos("dictionary-encoded page without a dictionary"))?;
        let validity = freeze_validity(validity);
        let keys = PrimitiveArray::new(ArrowDataType::UInt32, codes.into(), validity);
        let dtype =
            ArrowDataType::Dictionary(IntegerType::UInt32, Box::new(dict.dtype().clone()), false);
        // SAFETY: the decoder checked every code against the dictionary's length.
        Ok(unsafe { DictionaryArray::try_new_unchecked(dtype, keys, dict.to_boxed()) }.unwrap())
    }
}

/// Fixed-width values narrower than this are not worth coding: a `u32` code per row is bigger
/// than the value, and mapping such values is cheap anyway.
const MIN_CODED_WIDTH: usize = 4;

/// Whether [`decode_dictionary_codes`] decodes a column chunk of this parquet type to a column
/// of `dtype` (every top-level type but booleans, nested ones, Polars' own Categoricals and
/// fixed-width values under [`MIN_CODED_WIDTH`] bytes: 8 and 16-bit integers, `Float16`).
pub fn dictionary_codes_supported(type_: &PrimitiveType, dtype: &ArrowDataType) -> bool {
    use ArrowDataType::*;
    let dtype = match dtype {
        // An extension type is read as its storage by the flat decoders, but its field's dtype
        // does not match the codes' values.
        Extension(_) => return false,
        Decimal32(p, s) | Decimal64(p, s) => &Decimal(*p, *s),
        other => other,
    };
    let logical = dtype.to_storage();
    match (&type_.physical_type, logical) {
        (PhysicalType::ByteArray, Utf8View | LargeUtf8 | Utf8 | BinaryView | LargeBinary) => true,
        (PhysicalType::Int32, UInt32 | Int32 | Date32 | Time32(_) | Decimal(..)) => true,
        (
            PhysicalType::Int64,
            UInt32 | UInt64 | Int64 | Time64(_) | Duration(_) | Timestamp(..) | Decimal(..),
        ) => true,
        (PhysicalType::Int96, Timestamp(..)) => true,
        (PhysicalType::Float, Float32) | (PhysicalType::Double, Float64) => true,
        (PhysicalType::FixedLenByteArray(n), FixedSizeBinary(m)) => n == m && *m >= MIN_CODED_WIDTH,
        (PhysicalType::FixedLenByteArray(16), UInt128 | Int128) => true,
        (PhysicalType::FixedLenByteArray(n), Decimal(..)) => (1..=16).contains(n),
        _ => false,
    }
}

/// Decodes a flat column chunk as codes into its dictionary page, which must be there with every
/// data page dictionary-encoded, and [`dictionary_codes_supported`]. The values of the
/// dictionaries are of the field's dtype, as a flat read gives them; the String and Binary ones
/// share the dictionary page's buffers.
pub fn decode_dictionary_codes(
    field: &Field,
    type_: &PrimitiveType,
    pages: crate::parquet::read::BasicDecompressor,
    filter: Option<super::Filter>,
) -> ParquetResult<(Vec<DictionaryArray<u32>>, Bitmap)> {
    use ArrowDataType::*;
    let dtype = match &field.dtype {
        Decimal32(p, s) | Decimal64(p, s) => Decimal(*p, *s),
        other => other.clone(),
    };
    let name = field.name.as_str();
    let physical = &type_.physical_type;

    macro_rules! run {
        ($decoder:expr) => {
            run(name, pages, &dtype, $decoder, filter)
        };
    }

    match (physical, dtype.to_storage()) {
        (PhysicalType::ByteArray, Utf8View | LargeUtf8 | Utf8) => {
            categorical::decode_byte_dictionary_codes(name, pages, filter, true)
        },
        (PhysicalType::ByteArray, BinaryView) => {
            categorical::decode_byte_dictionary_codes(name, pages, filter, false)
        },
        (PhysicalType::ByteArray, LargeBinary) => run!(binary::BinaryDecoder),
        (PhysicalType::Int32, UInt32) => run!(primitive::IntDecoder::<i32, u32, _>::cast_as()),
        (PhysicalType::Int64, UInt32) => run!(primitive::IntDecoder::<i64, u32, _>::cast_as()),
        (PhysicalType::Int32, Int32 | Date32 | Time32(_)) => {
            run!(primitive::IntDecoder::<i32, _, _>::unit())
        },
        (PhysicalType::Int64, Int64 | Time64(_) | Duration(_)) => {
            run!(primitive::IntDecoder::<i64, _, _>::unit())
        },
        (PhysicalType::Int64, UInt64) => run!(primitive::IntDecoder::<i64, u64, _>::cast_as()),
        (PhysicalType::Int32, Decimal(..)) => {
            run!(primitive::IntDecoder::<i32, i128, _>::cast_into())
        },
        (PhysicalType::Int64, Decimal(..)) => {
            run!(primitive::IntDecoder::<i64, i128, _>::cast_into())
        },
        (PhysicalType::FixedLenByteArray(n), Decimal(..)) => {
            run!(primitive::FixedLenDecimalDecoder::new(*n))
        },
        (PhysicalType::Float, Float32) => run!(primitive::FloatDecoder::<f32, _, _>::unit()),
        (PhysicalType::Double, Float64) => run!(primitive::FloatDecoder::<f64, _, _>::unit()),
        (PhysicalType::FixedLenByteArray(n), FixedSizeBinary(_) | UInt128 | Int128) => {
            run!(fixed_size_binary::BinaryDecoder { size: *n })
        },
        (PhysicalType::Int96, Timestamp(unit, _)) => match unit {
            TimeUnit::Nanosecond => run!(primitive::FloatDecoder::closure(|x: [u32; 3]| {
                int96_to_i64_ns(x).unwrap_or(i64::MAX)
            })),
            TimeUnit::Microsecond => {
                run!(primitive::FloatDecoder::closure(|x: [u32; 3]| {
                    int96_to_i64_us(x)
                }))
            },
            TimeUnit::Millisecond => {
                run!(primitive::FloatDecoder::closure(|x: [u32; 3]| {
                    int96_to_i64_ms(x)
                }))
            },
            TimeUnit::Second => {
                run!(primitive::FloatDecoder::closure(|x: [u32; 3]| {
                    int96_to_i64_s(x)
                }))
            },
        },
        (PhysicalType::Int64, Timestamp(unit, _)) => {
            let (factor, is_multiplier) =
                super::simple::unify_timestamp_unit(&type_.logical_type, *unit);
            match (factor, is_multiplier) {
                (1, _) => run!(primitive::IntDecoder::<i64, _, _>::unit()),
                (a, true) => run!(primitive::IntDecoder::closure(move |x: i64| x * a)),
                (a, false) => run!(primitive::IntDecoder::closure(move |x: i64| x / a)),
            }
        },
        _ => Err(ParquetError::not_supported(format!(
            "dictionary codes for parquet type {physical:?} read as {dtype:?}"
        ))),
    }
}

fn run<D: Decoder>(
    name: &str,
    pages: crate::parquet::read::BasicDecompressor,
    dtype: &ArrowDataType,
    decoder: D,
    filter: Option<super::Filter>,
) -> ParquetResult<(Vec<DictionaryArray<u32>>, Bitmap)> {
    let (_, arrays, pred_true_mask) =
        utils::PageDecoder::new(name, pages, dtype.clone(), CodesDecoder::new(decoder), None)?
            .collect(filter)?;
    // Every chunk holds the same dictionary: give it the field's dtype, once.
    let Some(first) = arrays.first() else {
        return Ok((arrays, pred_true_mask));
    };
    let values = retype(first.values().clone(), dtype)?;
    let dict_dtype = ArrowDataType::Dictionary(IntegerType::UInt32, Box::new(dtype.clone()), false);
    let arrays = arrays
        .into_iter()
        .map(|a| {
            // SAFETY: the codes are unchanged, and so is the dictionary's length.
            Ok(unsafe {
                DictionaryArray::try_new_unchecked(
                    dict_dtype.clone(),
                    a.keys().clone(),
                    values.clone(),
                )
            }?)
        })
        .collect::<ParquetResult<Vec<_>>>()?;
    Ok((arrays, pred_true_mask))
}

/// `values`, as decoded by the flat decoder, with the logical `dtype` of its column.
fn retype(values: Box<dyn Array>, dtype: &ArrowDataType) -> ParquetResult<Box<dyn Array>> {
    macro_rules! primitive {
        ($($t:ty),*) => {$(
            if let Some(a) = values.as_any().downcast_ref::<PrimitiveArray<$t>>() {
                return Ok(a.clone().to(dtype.clone()).to_boxed());
            }
        )*};
    }
    primitive!(i8, i16, i32, i64, u8, u16, u32, u64, i128, f32, f64, pf16);
    if let Some(a) = values.as_any().downcast_ref::<FixedSizeBinaryArray>() {
        let (_, bytes, validity) = a.clone().into_inner();
        return Ok(match dtype.to_storage() {
            ArrowDataType::UInt128 => PrimitiveArray::<u128>::try_new(
                dtype.clone(),
                bytes.try_transmute().expect("aligned by the decoder"),
                validity,
            )?
            .to_boxed(),
            ArrowDataType::Int128 => PrimitiveArray::<i128>::try_new(
                dtype.clone(),
                bytes.try_transmute().expect("aligned by the decoder"),
                validity,
            )?
            .to_boxed(),
            _ => values,
        });
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use polars_arrow::datatypes::ArrowDataType;

    use super::*;
    use crate::parquet::schema::Repetition;
    use crate::parquet::schema::types::{FieldInfo, PrimitiveType};

    fn pq(physical_type: PhysicalType) -> PrimitiveType {
        PrimitiveType {
            field_info: FieldInfo {
                name: "c".into(),
                repetition: Repetition::Optional,
                id: None,
            },
            logical_type: None,
            converted_type: None,
            physical_type,
        }
    }

    #[test]
    fn supported_pairs() {
        use ArrowDataType::*;
        let yes = [
            (PhysicalType::Int32, UInt32),
            (PhysicalType::Int32, Date32),
            (PhysicalType::Int32, Decimal(9, 2)),
            (PhysicalType::Int64, UInt64),
            (
                PhysicalType::Int64,
                Time64(polars_arrow::datatypes::TimeUnit::Nanosecond),
            ),
            (PhysicalType::Int64, Decimal(18, 2)),
            (
                PhysicalType::Int96,
                Timestamp(polars_arrow::datatypes::TimeUnit::Nanosecond, None),
            ),
            (PhysicalType::Double, Float64),
            (PhysicalType::FixedLenByteArray(16), Decimal(38, 2)),
            (PhysicalType::FixedLenByteArray(16), Int128),
            (PhysicalType::FixedLenByteArray(4), FixedSizeBinary(4)),
            (PhysicalType::ByteArray, LargeBinary),
            (PhysicalType::ByteArray, Utf8View),
        ];
        for (physical, dtype) in yes {
            assert!(
                dictionary_codes_supported(&pq(physical), &dtype),
                "{physical:?} {dtype:?}"
            );
        }
        let no = [
            (PhysicalType::Boolean, Boolean),
            (PhysicalType::Int32, Int8),
            (PhysicalType::Int32, UInt8),
            (PhysicalType::Int32, Int16),
            (PhysicalType::Int32, UInt16),
            (PhysicalType::FixedLenByteArray(2), Float16),
            (PhysicalType::FixedLenByteArray(3), FixedSizeBinary(3)),
            (PhysicalType::Int32, Int64),
            (PhysicalType::ByteArray, Decimal(38, 2)),
            (PhysicalType::FixedLenByteArray(17), Decimal(38, 2)),
            (PhysicalType::Int32, Date64),
            (PhysicalType::ByteArray, Int32),
        ];
        for (physical, dtype) in no {
            assert!(
                !dictionary_codes_supported(&pq(physical), &dtype),
                "{physical:?} {dtype:?}"
            );
        }
    }
}
