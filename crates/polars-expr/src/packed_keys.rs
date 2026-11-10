//! Keys of a few columns packed into one unsigned integer, so that the single-key
//! tables can hold them.
//!
//! Each column gets a bit field in the word. A column with a value range stores
//! `value - min` in as few bits as the range needs; any other column stores its raw
//! bits. A value outside its range does not fit its field. Such a key gets an id from
//! a map shared by all tables of one operation, and its word is the escape bit plus
//! that id. Equal keys get equal ids, so equal keys still get equal words.
use std::sync::Mutex;

use num_traits::{PrimInt, WrappingAdd, WrappingSub};
use polars_arrow::array::{Array, BooleanArray, PrimitiveArray};
use polars_arrow::bitmap::{Bitmap, MutableBitmap};
use polars_arrow::compute::utils::combine_validities_and_many;
use polars_arrow::types::NativeType;
use polars_buffer::Buffer;
use polars_core::prelude::row_encode::_get_rows_encoded_unordered;
use polars_core::prelude::*;
use polars_core::with_match_physical_integer_polars_type;
use polars_utils::IdxSize;
use polars_utils::index::idxsize_to_u64;
use polars_utils::total_ord::{canonical_f32, canonical_f64};

/// A column value or a packed word, as an unsigned integer.
pub(crate) trait Bits:
    PrimInt + WrappingAdd + WrappingSub + NativeType + Send + Sync + 'static
{
    fn from_u128(x: u128) -> Self;

    fn to_u128(self) -> u128;

    /// All ones if `b`, else zero.
    #[inline(always)]
    fn mask(b: bool) -> Self {
        Self::zero().wrapping_sub(&Self::from_u128(b as u128))
    }

    /// `x` zero-extended or truncated to this width.
    #[inline(always)]
    fn from_bits<C: Bits>(x: C) -> Self {
        Self::from_u128(x.to_u128())
    }
}

macro_rules! impl_bits {
    ($($t:ty),*) => {$(
        impl Bits for $t {
            #[inline(always)]
            fn from_u128(x: u128) -> Self {
                x as $t
            }

            #[inline(always)]
            fn to_u128(self) -> u128 {
                self as u128
            }
        }
    )*};
}
impl_bits!(u8, u16, u32, u64, u128);

#[derive(Clone, Debug, PartialEq)]
enum Field {
    /// The raw bits of the value. When nulls are keys, the bit above the value is
    /// set for a null and the value bits are zero.
    Bits,
    /// `value - min`, both as bits of the column width. When nulls are keys, a null
    /// is 0 and a value is `value - min + 1`.
    Range { min: u128, span: u128 },
}

#[derive(Clone, Debug, PartialEq)]
struct PackedCol {
    physical: DataType,
    field: Field,
    /// Bits of the value: the column width, or those `span` needs.
    value_bits: u32,
    shift: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PackedLayout {
    cols: Vec<PackedCol>,
    nulls_in_key: bool,
    word_bits: u32,
    /// Set when a column has a range, so a key can be out of range.
    escape_shift: Option<u32>,
}

/// Width in bits of a key column of this physical type, or `None` when it is not
/// packed.
fn physical_bits(physical: &DataType) -> Option<u32> {
    Some(match physical {
        DataType::Null => 0,
        DataType::Boolean => 1,
        DataType::Int8 | DataType::UInt8 => 8,
        DataType::Int16 | DataType::UInt16 | DataType::Float16 => 16,
        DataType::Int32 | DataType::UInt32 | DataType::Float32 => 32,
        DataType::Int64 | DataType::UInt64 | DataType::Float64 => 64,
        DataType::Int128 | DataType::UInt128 => 128,
        _ => return None,
    })
}

fn word_widths() -> impl Iterator<Item = u32> {
    [
        cfg!(feature = "dtype-u8").then_some(8),
        cfg!(feature = "dtype-u16").then_some(16),
        Some(32),
        Some(64),
        cfg!(feature = "dtype-u128").then_some(128),
    ]
    .into_iter()
    .flatten()
}

/// The field of a column with values in `min..=max`, or `None` when the range
/// doesn't make it smaller than its raw bits.
fn range_field(
    physical: &DataType,
    bits: u32,
    nulls_in_key: bool,
    min: i128,
    max: i128,
) -> Option<(Field, u32)> {
    if !physical.is_integer() || min > max {
        return None;
    }
    // The range may come from a wider type; only the part the type can hold counts.
    let (lo, hi) = if physical.is_unsigned_integer() {
        (
            0,
            if bits >= 128 {
                i128::MAX
            } else {
                (1i128 << bits) - 1
            },
        )
    } else if bits >= 128 {
        (i128::MIN, i128::MAX)
    } else {
        (-(1i128 << (bits - 1)), (1i128 << (bits - 1)) - 1)
    };
    let (min, max) = (min.max(lo), max.min(hi));
    if min > max {
        return None;
    }
    let span = max.checked_sub(min)? as u128;
    let codes = span.checked_add(nulls_in_key as u128)?;
    let value_bits = 128 - codes.leading_zeros();
    let full_bits = bits + nulls_in_key as u32;
    if value_bits >= full_bits {
        return None;
    }
    let col_mask = if bits >= 128 {
        u128::MAX
    } else {
        (1u128 << bits) - 1
    };
    Some((
        Field::Range {
            min: (min as u128) & col_mask,
            span,
        },
        value_bits,
    ))
}

impl PackedLayout {
    /// The layout of keys of these dtypes, or `None` when they don't fit one word or
    /// a single plain key is better.
    ///
    /// `ranges` holds an inclusive value range per column, or is empty. Values
    /// outside them are still correct, but slow.
    pub(crate) fn new<'a>(
        dtypes: impl IntoIterator<Item = &'a DataType>,
        nulls_in_key: bool,
        ranges: &[Option<(i128, i128)>],
    ) -> Option<Self> {
        let physicals: Vec<DataType> = dtypes.into_iter().map(|dt| dt.to_physical()).collect();
        if !(physicals.len() >= 2
            || matches!(physicals.as_slice(), [DataType::Boolean | DataType::Null]))
        {
            return None;
        }
        let full_width = Self::with_ranges(&physicals, nulls_in_key, &[]);
        if ranges.iter().all(|r| r.is_none()) {
            return full_width;
        }
        // Ranges need an escape bit and room for escape ids, so they may give a
        // wider word.
        let ranged = Self::with_ranges(&physicals, nulls_in_key, ranges);
        match (ranged, full_width) {
            (Some(r), Some(f)) if f.word_bits <= r.word_bits => Some(f),
            (r, f) => r.or(f),
        }
    }

    fn with_ranges(
        physicals: &[DataType],
        nulls_in_key: bool,
        ranges: &[Option<(i128, i128)>],
    ) -> Option<Self> {
        let mut cols = Vec::with_capacity(physicals.len());
        let mut shift = 0;
        for (i, physical) in physicals.iter().cloned().enumerate() {
            let bits = physical_bits(&physical)?;
            let range = ranges.get(i).copied().flatten();
            let (field, value_bits) = range
                .and_then(|(min, max)| range_field(&physical, bits, nulls_in_key, min, max))
                .unwrap_or((Field::Bits, bits));
            let null_bit = (nulls_in_key && field == Field::Bits && bits > 0) as u32;
            cols.push(PackedCol {
                physical,
                field,
                value_bits,
                shift,
            });
            shift += value_bits + null_bit;
        }

        let has_range = cols.iter().any(|c| matches!(c.field, Field::Range { .. }));
        // The escape bit is the top bit of the word, and the bits below it number the
        // escaped keys, so a word with ranges has at least 32 bits.
        let needed = shift + has_range as u32;
        let min_bits = if has_range { 32 } else { 0 };
        let word_bits = word_widths().find(|w| *w >= needed && *w >= min_bits)?;
        Some(Self {
            cols,
            nulls_in_key,
            word_bits,
            escape_shift: has_range.then_some(word_bits - 1),
        })
    }

    /// The dtype of the packed keys.
    pub(crate) fn word_dtype(&self) -> DataType {
        match self.word_bits {
            #[cfg(feature = "dtype-u8")]
            8 => DataType::UInt8,
            #[cfg(feature = "dtype-u16")]
            16 => DataType::UInt16,
            32 => DataType::UInt32,
            64 => DataType::UInt64,
            #[cfg(feature = "dtype-u128")]
            128 => DataType::UInt128,
            _ => unreachable!(),
        }
    }

    /// The word width and the bits of each column, e.g. `u32 [0..10) range of 1000 [10..42)`.
    fn describe(&self) -> String {
        let fields = self
            .cols
            .iter()
            .map(|col| {
                let end = col.shift + col.value_bits;
                match col.field {
                    Field::Bits => format!("[{}..{end})", col.shift),
                    Field::Range { span, .. } => {
                        format!("[{}..{end}) range of {}", col.shift, span + 1)
                    },
                }
            })
            .collect::<Vec<_>>();
        format!("u{} {}", self.word_bits, fields.join(" "))
    }
}

#[derive(Default)]
struct EscapeMap {
    /// The row encoding of each escaped key, and its id.
    ids: PlHashMap<Box<[u8]>, u64>,
    /// The escaped keys in id order, as their key columns.
    keys: Option<DataFrame>,
    len: u64,
    /// Estimated memory of the escaped keys, in bytes.
    bytes: usize,
}

/// Escaped keys only exist when the statistics do not match the data, so their
/// memory is capped instead of spilled.
const MAX_ESCAPED_BYTES: usize = 64 << 20;

/// A packed layout and the keys of the operation that did not fit it.
pub struct PackedKeys {
    layout: PackedLayout,
    escaped: Mutex<EscapeMap>,
    max_escaped_bytes: usize,
}

impl std::fmt::Debug for PackedKeys {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PackedKeys")
            .field("layout", &self.layout)
            .finish_non_exhaustive()
    }
}

/// The values of a physical integer or float column as unsigned bits, floats
/// canonicalized so that equal floats have equal bits.
macro_rules! with_column_bits {
    ($s:expr, |$values:ident, $C:ident| $body:expr) => {{
        let s: &Series = $s;
        match s.dtype() {
            #[cfg(feature = "dtype-f16")]
            DataType::Float16 => {
                type $C = u16;
                let arr = s.f16().unwrap().downcast_as_array();
                let bits: Vec<u16> = arr
                    .values()
                    .iter()
                    .map(|x| polars_utils::total_ord::canonical_f16(*x).to_bits())
                    .collect();
                let $values: &[u16] = &bits;
                $body
            },
            DataType::Float32 => {
                type $C = u32;
                let arr = s.f32().unwrap().downcast_as_array();
                let bits: Vec<u32> = arr
                    .values()
                    .iter()
                    .map(|x| canonical_f32(*x).to_bits())
                    .collect();
                let $values: &[u32] = &bits;
                $body
            },
            DataType::Float64 => {
                type $C = u64;
                let arr = s.f64().unwrap().downcast_as_array();
                let bits: Vec<u64> = arr
                    .values()
                    .iter()
                    .map(|x| canonical_f64(*x).to_bits())
                    .collect();
                let $values: &[u64] = &bits;
                $body
            },
            dt => with_match_physical_integer_polars_type!(dt, |$T| {
                let ca: &ChunkedArray<$T> = s.as_phys_any().downcast_ref().unwrap();
                let native = ca.downcast_as_array().values().as_slice();
                match size_of::<<$T as PolarsNumericType>::Native>() {
                    1 => {
                        type $C = u8;
                        let $values: &[u8] = bytemuck_cast(native);
                        $body
                    },
                    2 => {
                        type $C = u16;
                        let $values: &[u16] = bytemuck_cast(native);
                        $body
                    },
                    4 => {
                        type $C = u32;
                        let $values: &[u32] = bytemuck_cast(native);
                        $body
                    },
                    8 => {
                        type $C = u64;
                        let $values: &[u64] = bytemuck_cast(native);
                        $body
                    },
                    _ => {
                        type $C = u128;
                        let $values: &[u128] = bytemuck_cast(native);
                        $body
                    },
                }
            }),
        }
    }};
}

/// Reinterprets integers as unsigned integers of the same width.
fn bytemuck_cast<T: NativeType, U: Bits>(values: &[T]) -> &[U] {
    assert_eq!(size_of::<T>(), size_of::<U>());
    // SAFETY: both are plain integers of the same size and alignment.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), values.len()) }
}

/// ORs the field of each value into its word. Returns whether a valid value was out
/// of its range.
#[inline(always)]
fn encode_column<U: Bits, C: Bits>(
    out: &mut [U],
    values: &[C],
    validity: Option<&Bitmap>,
    col: &PackedCol,
    nulls_in_key: bool,
) -> bool {
    let shift = col.shift as usize;
    match col.field {
        Field::Bits => {
            match validity {
                Some(validity) if nulls_in_key => {
                    let null_shift = shift + col.value_bits as usize;
                    for ((o, x), valid) in out.iter_mut().zip(values).zip(validity.iter()) {
                        let x = *x & C::mask(valid);
                        *o = *o
                            | (U::from_bits(x) << shift)
                            | (U::from_u128(!valid as u128) << null_shift);
                    }
                },
                _ => {
                    for (o, x) in out.iter_mut().zip(values) {
                        *o = *o | (U::from_bits(*x) << shift);
                    }
                },
            }
            false
        },
        Field::Range { min, span } => {
            let min = C::from_u128(min);
            let span = C::from_u128(span);
            let mut out_of_range = false;
            match validity {
                None => {
                    let add = C::from_u128(nulls_in_key as u128);
                    for (o, x) in out.iter_mut().zip(values) {
                        let d = x.wrapping_sub(&min);
                        out_of_range |= d > span;
                        *o = *o | (U::from_bits(d.wrapping_add(&add)) << shift);
                    }
                },
                Some(validity) => {
                    for ((o, x), valid) in out.iter_mut().zip(values).zip(validity.iter()) {
                        let d = x.wrapping_sub(&min);
                        out_of_range |= (d > span) & valid;
                        let code = if nulls_in_key {
                            d.wrapping_add(&C::one()) & C::mask(valid)
                        } else {
                            d
                        };
                        *o = *o | (U::from_bits(code) << shift);
                    }
                },
            }
            out_of_range
        },
    }
}

/// Sets the rows whose valid value is out of the column's range.
fn mark_out_of_range<C: Bits>(
    values: &[C],
    validity: Option<&Bitmap>,
    min: u128,
    span: u128,
    marks: &mut MutableBitmap,
) {
    let min = C::from_u128(min);
    let span = C::from_u128(span);
    for (i, x) in values.iter().enumerate() {
        if x.wrapping_sub(&min) > span && validity.is_none_or(|v| v.get_bit(i)) {
            marks.set(i, true);
        }
    }
}

/// The values of each field of `words`, as bits of the column width.
fn decode_column<U: Bits, C: Bits>(
    words: &[U],
    col: &PackedCol,
    nulls_in_key: bool,
) -> (Vec<C>, Option<Bitmap>) {
    let shift = col.shift as usize;
    let value_mask = U::from_u128(if col.value_bits >= 128 {
        u128::MAX
    } else {
        (1u128 << col.value_bits) - 1
    });
    match col.field {
        Field::Bits => {
            let values = words
                .iter()
                .map(|w| C::from_bits((*w >> shift) & value_mask))
                .collect();
            let validity = nulls_in_key.then(|| {
                let null_shift = shift + col.value_bits as usize;
                Bitmap::from_trusted_len_iter(
                    words
                        .iter()
                        .map(|w| (*w >> null_shift) & U::one() == U::zero()),
                )
            });
            (values, validity)
        },
        Field::Range { min, .. } => {
            let min = C::from_u128(min);
            let add = C::from_u128(nulls_in_key as u128);
            let values = words
                .iter()
                .map(|w| {
                    C::from_bits((*w >> shift) & value_mask)
                        .wrapping_sub(&add)
                        .wrapping_add(&min)
                })
                .collect();
            let validity = nulls_in_key.then(|| {
                Bitmap::from_trusted_len_iter(
                    words
                        .iter()
                        .map(|w| (*w >> shift) & value_mask != U::zero()),
                )
            });
            (values, validity)
        },
    }
}

fn primitive_array<C: Bits, T: NativeType>(
    values: Vec<C>,
    validity: Option<Bitmap>,
    physical: &DataType,
) -> Box<dyn Array> {
    let values: Buffer<T> = Buffer::from(values).try_transmute().unwrap();
    Box::new(PrimitiveArray::<T>::new(
        physical.to_arrow(CompatLevel::newest()),
        values,
        validity,
    ))
}

impl PackedKeys {
    pub(crate) fn new(layout: PackedLayout) -> Self {
        Self {
            layout,
            escaped: Mutex::new(EscapeMap::default()),
            max_escaped_bytes: MAX_ESCAPED_BYTES,
        }
    }

    pub(crate) fn layout(&self) -> &PackedLayout {
        &self.layout
    }

    pub(crate) fn describe(&self) -> String {
        self.layout.describe()
    }

    /// The packed word of each key row, as a series of the word dtype. When
    /// `null_is_valid` is false, a key with a null is a null word.
    pub(crate) fn encode(&self, keys: &[Column], null_is_valid: bool) -> PolarsResult<Series> {
        assert_eq!(keys.len(), self.layout.cols.len());
        assert!(
            self.layout.nulls_in_key || !null_is_valid,
            "keys with null values were packed without null codes"
        );
        match self.layout.word_bits {
            #[cfg(feature = "dtype-u8")]
            8 => self.encode_impl::<u8>(keys, null_is_valid),
            #[cfg(feature = "dtype-u16")]
            16 => self.encode_impl::<u16>(keys, null_is_valid),
            32 => self.encode_impl::<u32>(keys, null_is_valid),
            64 => self.encode_impl::<u64>(keys, null_is_valid),
            #[cfg(feature = "dtype-u128")]
            128 => self.encode_impl::<u128>(keys, null_is_valid),
            _ => unreachable!(),
        }
    }

    fn encode_impl<U: Bits>(&self, keys: &[Column], null_is_valid: bool) -> PolarsResult<Series> {
        let height = keys.first().map_or(0, |c| c.len());
        let physicals: Vec<Series> = keys
            .iter()
            .zip(&self.layout.cols)
            .map(|(c, col)| {
                let s = c.as_materialized_series().to_physical_repr().rechunk();
                assert_eq!(
                    s.dtype(),
                    &col.physical,
                    "key dtype does not match its layout"
                );
                assert_eq!(s.len(), height);
                s
            })
            .collect();
        let validities: Vec<Option<Bitmap>> = physicals
            .iter()
            .map(|s| {
                let validity: Option<&Bitmap> = s.chunks().first().and_then(|a| a.validity());
                validity.filter(|v| v.unset_bits() > 0).cloned()
            })
            .collect();

        let nulls_in_key = self.layout.nulls_in_key;
        let mut out = vec![U::zero(); height];
        let mut any_out_of_range = false;
        for ((s, col), validity) in physicals.iter().zip(&self.layout.cols).zip(&validities) {
            let validity = validity.as_ref();
            match &col.physical {
                DataType::Null => {},
                DataType::Boolean => {
                    let values = s.bool().unwrap().downcast_as_array().values();
                    let values: Vec<u8> = values.iter().map(|b| b as u8).collect();
                    encode_column(&mut out, &values, validity, col, nulls_in_key);
                },
                _ => with_column_bits!(s, |values, C| {
                    any_out_of_range |=
                        encode_column::<U, C>(&mut out, values, validity, col, nulls_in_key);
                }),
            }
        }

        if any_out_of_range {
            self.escape(keys, &physicals, &validities, &mut out)?;
        }

        let validity = if null_is_valid {
            None
        } else {
            combine_validities_and_many(&validities.iter().map(|v| v.as_ref()).collect::<Vec<_>>())
        };
        let word_dtype = self.layout.word_dtype();
        let arr = PrimitiveArray::<U>::new(
            word_dtype.to_arrow(CompatLevel::newest()),
            out.into(),
            validity,
        );
        Ok(unsafe {
            Series::from_chunks_and_dtype_unchecked(
                PlSmallStr::EMPTY,
                vec![Box::new(arr)],
                &word_dtype,
            )
        })
    }

    /// Gives every key with a value out of its range the word of its escape id.
    #[cold]
    fn escape<U: Bits>(
        &self,
        keys: &[Column],
        physicals: &[Series],
        validities: &[Option<Bitmap>],
        out: &mut [U],
    ) -> PolarsResult<()> {
        let mut marks = MutableBitmap::from_len_zeroed(out.len());
        for ((s, col), validity) in physicals.iter().zip(&self.layout.cols).zip(validities) {
            if let Field::Range { min, span } = col.field {
                with_column_bits!(s, |values, C| {
                    mark_out_of_range::<C>(values, validity.as_ref(), min, span, &mut marks);
                });
            }
        }
        let marks: Bitmap = marks.freeze();
        let rows: Vec<IdxSize> = marks.true_idx_iter().map(|i| i as IdxSize).collect();
        let idx = IdxCa::from_vec(PlSmallStr::EMPTY, rows.clone());
        let escaped: Vec<Column> = keys
            .iter()
            .map(|c| unsafe { c.take_unchecked(&idx) })
            .collect();
        let encoded = _get_rows_encoded_unordered(&escaped, false).unwrap();

        let escape_bit = U::one() << self.layout.escape_shift.unwrap() as usize;
        let max_id = ((escape_bit - U::one()).to_u128() as u64).min(idxsize_to_u64(IdxSize::MAX));
        let mut map = self.escaped.lock().unwrap();
        let map = &mut *map;
        let mut new_rows = Vec::new();
        let mut full = false;
        for (i, (row, bytes)) in rows.iter().zip(encoded.values_iter()).enumerate() {
            let id = match map.ids.get(bytes) {
                Some(id) => *id,
                None => {
                    // The map entry, and the key columns of about the same size.
                    let size = 2 * bytes.len() + 32;
                    if map.len >= max_id || map.bytes + size > self.max_escaped_bytes {
                        full = true;
                        break;
                    }
                    let id = map.len;
                    map.ids.insert(Box::from(bytes), id);
                    map.len += 1;
                    map.bytes += size;
                    new_rows.push(i as IdxSize);
                    id
                },
            };
            out[*row as usize] = escape_bit | U::from_u128(id as u128);
        }
        if !new_rows.is_empty() {
            let new_rows = IdxCa::from_vec(PlSmallStr::EMPTY, new_rows);
            let cols = escaped
                .iter()
                .map(|c| unsafe { c.take_unchecked(&new_rows) })
                .collect();
            let new_keys = unsafe { DataFrame::new_unchecked(new_rows.len(), cols) };
            match &mut map.keys {
                Some(keys) => {
                    keys.vstack_mut_owned_unchecked(new_keys);
                    if keys.first_col_n_chunks() > 64 {
                        keys.rechunk_mut();
                    }
                },
                None => map.keys = Some(new_keys),
            }
        }
        polars_ensure!(
            !full,
            ComputeError: "too many join or group-by keys are outside the min/max of their column statistics; the statistics do not match the data"
        );
        Ok(())
    }

    /// The key columns of `schema` for packed `words`. The bits of a null word are
    /// decoded like any other.
    pub(crate) fn decode<U: Bits>(&self, schema: &Schema, words: &[U]) -> DataFrame {
        assert_eq!(schema.len(), self.layout.cols.len());
        let nulls_in_key = self.layout.nulls_in_key;
        let mut cols: Vec<Column> = schema
            .iter()
            .zip(&self.layout.cols)
            .map(|((name, dtype), col)| {
                assert_eq!(dtype.to_physical(), col.physical);
                let array: Box<dyn Array> = match &col.physical {
                    DataType::Null => {
                        return Column::full_null(name.clone(), words.len(), dtype);
                    },
                    DataType::Boolean => {
                        let (values, validity) = decode_column::<U, u8>(words, col, nulls_in_key);
                        Box::new(BooleanArray::new(
                            ArrowDataType::Boolean,
                            Bitmap::from_trusted_len_iter(values.into_iter().map(|b| b != 0)),
                            validity,
                        ))
                    },
                    #[cfg(feature = "dtype-f16")]
                    DataType::Float16 => {
                        let (v, m) = decode_column::<U, u16>(words, col, nulls_in_key);
                        primitive_array::<u16, polars_utils::float16::pf16>(v, m, &col.physical)
                    },
                    DataType::Float32 => {
                        let (v, m) = decode_column::<U, u32>(words, col, nulls_in_key);
                        primitive_array::<u32, f32>(v, m, &col.physical)
                    },
                    DataType::Float64 => {
                        let (v, m) = decode_column::<U, u64>(words, col, nulls_in_key);
                        primitive_array::<u64, f64>(v, m, &col.physical)
                    },
                    dt => with_match_physical_integer_polars_type!(dt, |$T| {
                        type N = <$T as PolarsNumericType>::Native;
                        match size_of::<N>() {
                            1 => { let (v, m) = decode_column::<U, u8>(words, col, nulls_in_key); primitive_array::<u8, N>(v, m, dt) },
                            2 => { let (v, m) = decode_column::<U, u16>(words, col, nulls_in_key); primitive_array::<u16, N>(v, m, dt) },
                            4 => { let (v, m) = decode_column::<U, u32>(words, col, nulls_in_key); primitive_array::<u32, N>(v, m, dt) },
                            8 => { let (v, m) = decode_column::<U, u64>(words, col, nulls_in_key); primitive_array::<u64, N>(v, m, dt) },
                            _ => { let (v, m) = decode_column::<U, u128>(words, col, nulls_in_key); primitive_array::<u128, N>(v, m, dt) },
                        }
                    }),
                };
                unsafe {
                    Series::from_chunks_and_dtype_unchecked(name.clone(), vec![array], &col.physical)
                        .from_physical_unchecked(dtype)
                        .unwrap()
                        .into_column()
                }
            })
            .collect();

        if let Some(escape_shift) = self.layout.escape_shift {
            let escape_bit = U::one() << escape_shift as usize;
            if words.iter().any(|w| *w & escape_bit != U::zero()) {
                self.patch_escaped(&mut cols, words, escape_bit);
            }
        }
        unsafe { DataFrame::new_unchecked(words.len(), cols) }
    }

    #[cold]
    fn patch_escaped<U: Bits>(&self, cols: &mut [Column], words: &[U], escape_bit: U) {
        let escaped = self.escaped.lock().unwrap().keys.clone().unwrap();
        let ids: IdxCa = words
            .iter()
            .map(|w| {
                (*w & escape_bit != U::zero()).then(|| (*w & !escape_bit).to_u128() as IdxSize)
            })
            .collect();
        let is_packed: BooleanChunked = ids.is_null();
        for (col, escaped_col) in cols.iter_mut().zip(escaped.columns()) {
            let from_map = unsafe { escaped_col.take_unchecked(&ids) };
            let patched = col.zip_with(&is_packed, &from_map).unwrap();
            *col = patched.with_name(col.name().clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use polars_utils::IdxSize;

    use super::*;
    use crate::groups::new_hash_grouper;
    use crate::hash_keys::KeyEncoder;
    use crate::idx_table::new_idx_table;

    fn word_dtype(
        dtypes: &[DataType],
        nulls_in_key: bool,
        ranges: &[Option<(i128, i128)>],
    ) -> Option<DataType> {
        PackedLayout::new(dtypes, nulls_in_key, ranges).map(|l| l.word_dtype())
    }

    #[test]
    #[cfg(all(feature = "dtype-u8", feature = "dtype-u16", feature = "dtype-u128"))]
    fn word_width() {
        use DataType::*;
        let bools = vec![Boolean; 8];
        assert_eq!(word_dtype(&bools, false, &[]), Some(UInt8));
        assert_eq!(word_dtype(&bools, true, &[]), Some(UInt16));
        assert_eq!(word_dtype(&[Int32, Int16], true, &[]), Some(UInt64));
        assert_eq!(word_dtype(&[Int64, Int64], false, &[]), Some(UInt128));
        assert_eq!(word_dtype(&[Int64, Int64], true, &[]), None);
        assert_eq!(word_dtype(&[Int64, String], false, &[]), None);
        // A single plain key is not packed.
        assert_eq!(word_dtype(&[Int64], false, &[]), None);
        assert_eq!(word_dtype(&[Boolean], true, &[]), Some(UInt8));
        // 0..=1000 plus the null code needs 10 bits, and a word with ranges has at
        // least 32 bits.
        let r = Some((0, 1000));
        assert_eq!(word_dtype(&[Int64, Int64], true, &[r, r]), Some(UInt32));
        let r = Some((-(1 << 40), 1 << 40));
        assert_eq!(word_dtype(&[Int64, Int64], true, &[r, r]), Some(UInt128));
        // A range as wide as the type doesn't help.
        let r = Some((i8::MIN as i128, i8::MAX as i128));
        assert_eq!(word_dtype(&[Int8, Int8], false, &[r, r]), Some(UInt16));
        // Narrow ranges need a 32-bit word, so the 16-bit full width is used.
        let r = Some((0, 3));
        assert_eq!(word_dtype(&[Int8, Int8], false, &[r, r]), Some(UInt16));
    }

    fn keys_df() -> DataFrame {
        df!(
            "a" => [Some(0i64), Some(1), Some(2), Some(1000), Some(-5), Some(2), Some(1000), None, Some(1), None, Some(i64::MIN), Some(i64::MAX)],
            "b" => [Some(1i32), Some(1), Some(2), Some(1), Some(3), Some(2), Some(1), Some(5), Some(1), Some(5), Some(0), None],
            "c" => [Some(true), Some(false), Some(true), Some(true), None, Some(true), Some(true), Some(false), Some(false), Some(false), Some(true), Some(true)]
        )
        .unwrap()
    }

    /// Keys whose values are mostly outside of these ranges.
    fn narrow_encoder(schema: Arc<Schema>, nulls_in_key: bool) -> KeyEncoder {
        let encoder = KeyEncoder::new(schema, nulls_in_key, &[Some((0, 2)), Some((1, 3)), None]);
        assert!(encoder.describe_packed().unwrap().contains("range"));
        encoder
    }

    fn group(encoder: &KeyEncoder, df: &DataFrame) -> (Vec<IdxSize>, DataFrame) {
        let mut grouper = new_hash_grouper(encoder);
        let mut group_idxs = Vec::new();
        let random_state = PlRandomState::default();
        // In two parts, so the second part finds the escaped keys of the first.
        let mid = df.height() / 2;
        for part in [df.slice(0, mid), df.slice(mid as i64, df.height() - mid)] {
            let keys = encoder
                .hash_keys(&part, random_state.clone(), true)
                .unwrap();
            let rows: Vec<IdxSize> = (0..part.height() as IdxSize).collect();
            unsafe { grouper.insert_keys_subset(&keys, &rows, Some(&mut group_idxs)) };
        }
        (
            group_idxs,
            grouper.get_keys_in_group_order(encoder.schema()),
        )
    }

    #[test]
    fn escaped_keys_group_like_full_width() {
        let df = keys_df();
        let schema = df.schema().clone();
        let (narrow_groups, narrow_keys) = group(&narrow_encoder(schema.clone(), true), &df);
        let (full_groups, full_keys) = group(&KeyEncoder::new(schema, true, &[]), &df);
        assert_eq!(narrow_groups, full_groups);
        assert_eq!(narrow_groups, [0, 1, 2, 3, 4, 2, 3, 5, 1, 5, 6, 7]);
        assert!(narrow_keys.equals_missing(&full_keys));
        let first_rows = IdxCa::from_vec(PlSmallStr::EMPTY, vec![0, 1, 2, 3, 4, 7, 10, 11]);
        assert!(narrow_keys.equals_missing(&df.take(&first_rows).unwrap()));
    }

    #[test]
    #[cfg(all(feature = "dtype-i8", feature = "dtype-u16", feature = "dtype-date"))]
    fn round_trip_all_types() {
        let mut df = df!(
            "bool" => [Some(true), None, Some(false), Some(true)],
            "i8" => [Some(-1i8), Some(i8::MIN), None, Some(-1)],
            "u16" => [Some(u16::MAX), Some(0), Some(1), Some(u16::MAX)],
            "f32" => [Some(-0.0f32), Some(f32::INFINITY), None, Some(0.0)],
            "date" => [Some(-3i32), None, Some(i32::MAX), Some(-3)]
        )
        .unwrap();
        df.with_column(Column::full_null("null".into(), 4, &DataType::Null))
            .unwrap();
        let date = df.column("date").unwrap().cast(&DataType::Date).unwrap();
        df.with_column(date).unwrap();
        let encoder = KeyEncoder::new(df.schema().clone(), true, &[]);
        assert!(encoder.describe_packed().is_some());
        let (groups, keys) = group(&encoder, &df);
        // -0.0 and 0.0 are one key.
        assert_eq!(groups, [0, 1, 2, 0]);
        let first_rows = IdxCa::from_vec(PlSmallStr::EMPTY, vec![0, 1, 2]);
        assert!(keys.equals_missing(&df.take(&first_rows).unwrap()));
    }

    fn join_matches(
        encoder: &KeyEncoder,
        build: &DataFrame,
        probe: &DataFrame,
    ) -> Vec<(IdxSize, IdxSize)> {
        let mut table = new_idx_table(encoder);
        let random_state = PlRandomState::default();
        let build_keys = encoder
            .hash_keys(build, random_state.clone(), false)
            .unwrap();
        let rows: Vec<IdxSize> = (0..build.height() as IdxSize).collect();
        unsafe { table.insert_keys_subset(&build_keys, &rows, false) };
        let (mut table_match, mut probe_match) = (Vec::new(), Vec::new());
        let probe_keys = encoder.hash_keys(probe, random_state, false).unwrap();
        let rows: Vec<IdxSize> = (0..probe.height() as IdxSize).collect();
        unsafe {
            table.probe_subset(
                &probe_keys,
                &rows,
                &mut table_match,
                &mut probe_match,
                false,
                false,
                IdxSize::MAX,
            )
        };
        let mut matches: Vec<_> = table_match.into_iter().zip(probe_match).collect();
        matches.sort_unstable();
        matches
    }

    #[test]
    fn escaped_keys_match_in_join() {
        let df = keys_df();
        let build = df.slice(0, 7);
        let probe = df.slice(3, 9);
        let schema = df.schema().clone();
        let narrow = join_matches(&narrow_encoder(schema.clone(), false), &build, &probe);
        let full = join_matches(&KeyEncoder::new(schema, false, &[]), &build, &probe);
        assert_eq!(narrow, full);
        // Rows 3 and 6 are the same out-of-range key, and null keys never match.
        assert!(narrow.contains(&(3, 0)) && narrow.contains(&(6, 3)) && narrow.contains(&(3, 3)));
        assert!(!narrow.iter().any(|(_, p)| [4, 6].contains(p)));
    }

    #[test]
    fn too_many_escaped_keys_error() {
        let df = keys_df();
        let layout = PackedLayout::new(
            df.schema().iter_values(),
            true,
            &[Some((0, 2)), Some((1, 3)), None],
        )
        .unwrap();
        let packed = PackedKeys {
            layout,
            escaped: Mutex::new(EscapeMap::default()),
            max_escaped_bytes: 100,
        };
        let err = packed.encode(df.columns(), true).unwrap_err();
        assert!(err.to_string().contains("statistics do not match the data"));
        let map = packed.escaped.lock().unwrap();
        assert!(map.len > 0);
        assert_eq!(map.len as usize, map.keys.as_ref().unwrap().height());
    }
}
