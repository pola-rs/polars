use polars_arrow::datatypes::ArrowDataType;

use crate::fixed::decimal;
use crate::row::{RowEncodingContext, RowEncodingOptions};

/// Which array type the encoded rows are returned as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowEncodingOutput {
    AlwaysBinaryOffset,
    AlwaysBinary,
    /// Bit-packed rows when every column is a flat fixed-width type and they fit in 128 bits:
    /// a boolean for at most 1 bit, else the smallest unsigned integer that fits.
    /// A single binary or string column without context that is unordered or ascending with
    /// nulls first is returned as is, so these rows may contain nulls. Otherwise `LargeBinary`.
    Optimal,
}

/// One input column of a row encoding.
#[derive(Clone, Debug)]
pub struct RowEncodingField {
    /// Physical arrow dtype of the column.
    pub dtype: ArrowDataType,
    pub opt: RowEncodingOptions,
    pub ctx: Option<RowEncodingContext>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum PackedKind {
    Null,
    Boolean,
    /// Integers and floats, stored as their order-preserving fixed-length encoding.
    Number,
    Decimal,
}

#[derive(Clone, Debug)]
pub(crate) struct PackedField {
    pub kind: PackedKind,
    pub value_bits: u32,
    pub has_null_bit: bool,
    /// Position of the lowest value bit; the null bit sits right above the value bits.
    pub shift: u32,
    pub descending: bool,
    pub nulls_last: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct Packing {
    pub fields: Vec<PackedField>,
    pub num_bits: u32,
}

#[derive(Clone, Debug)]
enum RowFormat {
    /// The byte format, see the crate documentation.
    Bytes,
    /// The only column, unchanged.
    PassThrough,
    Packed(Packing),
}

/// How columns are turned into rows. Rows must be decoded with the layout they were encoded
/// with.
#[derive(Clone, Debug)]
pub struct RowEncodingLayout {
    fields: Vec<RowEncodingField>,
    dtype: ArrowDataType,
    format: RowFormat,
}

impl RowEncodingLayout {
    pub fn new(fields: Vec<RowEncodingField>, output: RowEncodingOutput) -> Self {
        let (dtype, format) = match output {
            RowEncodingOutput::AlwaysBinaryOffset => (ArrowDataType::LargeBinary, RowFormat::Bytes),
            RowEncodingOutput::AlwaysBinary => (ArrowDataType::BinaryView, RowFormat::Bytes),
            RowEncodingOutput::Optimal => {
                if let Some(dtype) = pass_through_dtype(&fields) {
                    (dtype, RowFormat::PassThrough)
                } else if let Some(p) = pack(&fields) {
                    (packed_dtype(p.num_bits), RowFormat::Packed(p))
                } else {
                    (ArrowDataType::LargeBinary, RowFormat::Bytes)
                }
            },
        };
        Self {
            fields,
            dtype,
            format,
        }
    }

    pub fn fields(&self) -> &[RowEncodingField] {
        &self.fields
    }

    /// The dtype of the rows: `Boolean`, `UInt8` to `UInt128`, `BinaryView` or `LargeBinary`.
    pub fn dtype(&self) -> &ArrowDataType {
        &self.dtype
    }

    pub(crate) fn packing(&self) -> Option<&Packing> {
        match &self.format {
            RowFormat::Packed(packing) => Some(packing),
            _ => None,
        }
    }

    /// Whether the rows are bit-packed into a boolean or an unsigned integer.
    pub fn is_packed(&self) -> bool {
        matches!(self.format, RowFormat::Packed(_))
    }

    /// Whether the rows are the only column, unchanged.
    pub fn is_pass_through(&self) -> bool {
        matches!(self.format, RowFormat::PassThrough)
    }
}

fn packed_dtype(num_bits: u32) -> ArrowDataType {
    match num_bits {
        0..=1 => ArrowDataType::Boolean,
        2..=8 => ArrowDataType::UInt8,
        9..=16 => ArrowDataType::UInt16,
        17..=32 => ArrowDataType::UInt32,
        33..=64 => ArrowDataType::UInt64,
        65..=128 => ArrowDataType::UInt128,
        _ => unreachable!(),
    }
}

/// The dtype of the rows if the only column can be returned unchanged.
fn pass_through_dtype(fields: &[RowEncodingField]) -> Option<ArrowDataType> {
    let [field] = fields else {
        return None;
    };
    let ascending_nulls_first = !field.opt.is_ordered()
        || !field
            .opt
            .intersects(RowEncodingOptions::DESCENDING | RowEncodingOptions::NULLS_LAST);
    if field.ctx.is_some() || !ascending_nulls_first {
        return None;
    }
    match field.dtype {
        ArrowDataType::BinaryView | ArrowDataType::Utf8View => Some(ArrowDataType::BinaryView),
        ArrowDataType::LargeBinary => Some(ArrowDataType::LargeBinary),
        _ => None,
    }
}

/// The kind and number of value bits of a field, if it can be bit-packed.
fn packed_kind(field: &RowEncodingField) -> Option<(PackedKind, u32)> {
    use ArrowDataType as D;
    let number = |bytes: usize| Some((PackedKind::Number, 8 * bytes as u32));
    match &field.ctx {
        None => {},
        Some(RowEncodingContext::Decimal(precision)) if field.dtype == D::Int128 => {
            let num_bits = decimal::num_bits_from_precision(*precision);
            return if num_bits >= 127 {
                number(16)
            } else {
                Some((PackedKind::Decimal, num_bits as u32 + 1))
            };
        },
        // Lexically ordered categoricals are encoded as strings.
        Some(RowEncodingContext::Categorical(ctx)) if ctx.is_enum || !field.opt.is_ordered() => {},
        Some(_) => return None,
    }
    match &field.dtype {
        D::Null => Some((PackedKind::Null, 0)),
        D::Boolean => Some((PackedKind::Boolean, 1)),
        D::UInt8 | D::Int8 => number(1),
        D::UInt16 | D::Int16 | D::Float16 => number(2),
        D::UInt32 | D::Int32 | D::Float32 => number(4),
        D::UInt64 | D::Int64 | D::Float64 => number(8),
        D::UInt128 | D::Int128 => number(16),
        _ => None,
    }
}

/// The bit-packing of the fields, if they can all be packed into at most 128 bits.
fn pack(fields: &[RowEncodingField]) -> Option<Packing> {
    let mut packed = Vec::with_capacity(fields.len());
    let mut shift = 0;
    for field in fields.iter().rev() {
        let (kind, value_bits) = packed_kind(field)?;
        let has_null_bit = field.opt.encode_nulls() && !matches!(kind, PackedKind::Null);
        let ordered = field.opt.is_ordered();
        packed.push(PackedField {
            kind,
            value_bits,
            has_null_bit,
            shift,
            descending: ordered && field.opt.contains(RowEncodingOptions::DESCENDING),
            nulls_last: ordered && field.opt.contains(RowEncodingOptions::NULLS_LAST),
        });
        shift += value_bits + has_null_bit as u32;
    }
    if shift > 128 {
        return None;
    }
    packed.reverse();
    Some(Packing {
        fields: packed,
        num_bits: shift,
    })
}
