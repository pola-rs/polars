use std::sync::Arc;

use arrow::array::Array;
use arrow::bitmap::{Bitmap, MutableBitmap};
use arrow::types::AlignedBytes;

#[derive(Clone)]
pub enum ParquetScalar {
    Null,

    Boolean(bool),

    Int8(i8),
    Int16(i16),
    Int32(i32),
    Int64(i64),
    UInt8(u8),
    UInt16(u16),
    UInt32(u32),
    UInt64(u64),

    Float32(f32),
    Float64(f64),

    FixedSizeBinary(Box<[u8]>),

    String(Box<str>),
    Binary(Box<[u8]>),
}

impl ParquetScalar {
    pub(crate) fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }

    pub(crate) fn to_aligned_bytes<B: AlignedBytes>(&self) -> Option<B> {
        match self {
            Self::Int8(v) => <B::Unaligned>::try_from(&v.to_le_bytes())
                .ok()
                .map(B::from_unaligned),
            Self::Int16(v) => <B::Unaligned>::try_from(&v.to_le_bytes())
                .ok()
                .map(B::from_unaligned),
            Self::Int32(v) => <B::Unaligned>::try_from(&v.to_le_bytes())
                .ok()
                .map(B::from_unaligned),
            Self::Int64(v) => <B::Unaligned>::try_from(&v.to_le_bytes())
                .ok()
                .map(B::from_unaligned),
            Self::UInt8(v) => <B::Unaligned>::try_from(&v.to_le_bytes())
                .ok()
                .map(B::from_unaligned),
            Self::UInt16(v) => <B::Unaligned>::try_from(&v.to_le_bytes())
                .ok()
                .map(B::from_unaligned),
            Self::UInt32(v) => <B::Unaligned>::try_from(&v.to_le_bytes())
                .ok()
                .map(B::from_unaligned),
            Self::UInt64(v) => <B::Unaligned>::try_from(&v.to_le_bytes())
                .ok()
                .map(B::from_unaligned),
            Self::Float32(v) => <B::Unaligned>::try_from(&v.to_le_bytes())
                .ok()
                .map(B::from_unaligned),
            Self::Float64(v) => <B::Unaligned>::try_from(&v.to_le_bytes())
                .ok()
                .map(B::from_unaligned),
            _ => None,
        }
    }

    pub(crate) fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s.as_ref()),
            _ => None,
        }
    }

    pub(crate) fn as_binary(&self) -> Option<&[u8]> {
        match self {
            Self::Binary(s) => Some(s.as_ref()),
            _ => None,
        }
    }
}

pub enum ParquetScalarRange {
    Min(ParquetScalar),
    Max(ParquetScalar),
    Closed(ParquetScalar, ParquetScalar),
}

pub type ParquetColumnExprRef = Arc<dyn ParquetColumnExpr>;
pub trait ParquetColumnExpr: Send + Sync {
    fn evaluate(&self, values: &dyn Array) -> Bitmap {
        let mut bm = MutableBitmap::new();
        self.evaluate_mut(values, &mut bm);
        bm.freeze()
    }
    fn evaluate_mut(&self, values: &dyn Array, bm: &mut MutableBitmap);

    fn evaluate_null(&self) -> bool;

    fn to_equals_scalar(&self) -> Option<ParquetScalar>;
    fn to_range_scalar(&self) -> Option<ParquetScalarRange>;
}

impl ParquetScalarRange {
    pub fn into_closed_with_min_max(
        self,
        min: ParquetScalar,
        max: ParquetScalar,
    ) -> (ParquetScalar, ParquetScalar) {
        debug_assert_eq!(std::mem::discriminant(&min), std::mem::discriminant(&max));

        match self {
            Self::Min(min) => (min, max),
            Self::Max(max) => (min, max),
            Self::Closed(min, max) => (min, max),
        }
    }
}
