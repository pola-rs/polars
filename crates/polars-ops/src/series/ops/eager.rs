use polars_core::prelude::DataType;
use polars_core::series::Series;
use polars_error::PolarsResult;

pub trait ShrinkType {
    fn shrink_type(&self) -> PolarsResult<Series>;
}

impl ShrinkType for Series {
    fn shrink_type(&self) -> PolarsResult<Series> {
        if !self.dtype().is_primitive_numeric() {
            return Ok(self.clone());
        }

        if self.dtype().is_float() {
            return self.cast(&DataType::Float32);
        }

        if self.dtype().is_unsigned_integer() {
            let max = self.max_reduce()?.value().extract::<u64>().unwrap_or(0_u64);

            if cfg!(feature = "dtype-u8") && max <= u8::MAX as u64 {
                self.cast(&DataType::UInt8)
            } else if cfg!(feature = "dtype-u16") && max <= u16::MAX as u64 {
                self.cast(&DataType::UInt16)
            } else if max <= u32::MAX as u64 {
                self.cast(&DataType::UInt32)
            } else {
                Ok(self.clone())
            }
        } else {
            let min = self.min_reduce()?.value().extract::<i64>().unwrap_or(0_i64);
            let max = self.max_reduce()?.value().extract::<i64>().unwrap_or(0_i64);

            if cfg!(feature = "dtype-i8") && min >= i8::MIN as i64 && max <= i8::MAX as i64 {
                self.cast(&DataType::Int8)
            } else if cfg!(feature = "dtype-i16")
                && min >= i16::MIN as i64
                && max <= i16::MAX as i64
            {
                self.cast(&DataType::Int16)
            } else if min >= i32::MIN as i64 && max <= i32::MAX as i64 {
                self.cast(&DataType::Int32)
            } else {
                Ok(self.clone())
            }
        }
    }
}
