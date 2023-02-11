use super::*;
use crate::prelude::*;

pub type DecimalChunked = Logical<DecimalType, Int128Type>;

impl Int128Chunked {
    pub fn into_decimal(self, precision: usize, scale: usize) -> DecimalChunked {
        let mut dt = DecimalChunked::new_logical(self);
        dt.2 = Some(DataType::Decimal128(Some((precision, scale))));
        dt
    }
}

impl LogicalType for DecimalChunked {
    fn dtype(&self) -> &DataType {
        self.2.as_ref().unwrap()
    }

    fn cast(&self, dtype: &DataType) -> PolarsResult<Series> {
        // TODO: proper cast for various numeric types
        self.0.cast(dtype)
    }
}

impl DecimalChunked {
    pub fn precision(&self) -> usize {
        match self.2.as_ref().unwrap() {
            DataType::Decimal128(Some((precision, _))) => *precision,
            _ => unreachable!(),
        }
    }
    pub fn scale(&self) -> usize {
        match self.2.as_ref().unwrap() {
            DataType::Decimal128(Some((_, scale))) => *scale,
            _ => unreachable!(),
        }
    }
}
