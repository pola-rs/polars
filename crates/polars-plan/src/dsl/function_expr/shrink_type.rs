use super::*;

pub(super) fn shrink(c: Column) -> PolarsResult<Column> {
    if !c.dtype().is_primitive_numeric() {
        return Ok(c);
    }

    if c.dtype().is_float() {
        return c.cast(&DataType::Float32);
    }

    if c.dtype().is_unsigned_integer() {
        let max = c.max_reduce()?.value().extract::<u64>().unwrap_or(0_u64);

        if cfg!(feature = "dtype-u8") && max <= u8::MAX as u64 {
            c.cast(&DataType::UInt8)
        } else if cfg!(feature = "dtype-u16") && max <= u16::MAX as u64 {
            c.cast(&DataType::UInt16)
        } else if max <= u32::MAX as u64 {
            c.cast(&DataType::UInt32)
        } else {
            Ok(c)
        }
    } else {
        let min = c.min_reduce()?.value().extract::<i64>().unwrap_or(0_i64);
        let max = c.max_reduce()?.value().extract::<i64>().unwrap_or(0_i64);

        if cfg!(feature = "dtype-i8") && min >= i8::MIN as i64 && max <= i8::MAX as i64 {
            c.cast(&DataType::Int8)
        } else if cfg!(feature = "dtype-i16") && min >= i16::MIN as i64 && max <= i16::MAX as i64 {
            c.cast(&DataType::Int16)
        } else if min >= i32::MIN as i64 && max <= i32::MAX as i64 {
            c.cast(&DataType::Int32)
        } else {
            Ok(c)
        }
    }
}
