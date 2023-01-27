use super::*;

pub(super) fn shrink(s: Series) -> PolarsResult<Series> {
    if s.dtype().is_numeric() {
        if s.dtype().is_float() {
            s.cast(&DataType::Float32)
        } else if s.dtype().is_unsigned() {
            let max = s.max_as_series().get(0).unwrap().extract::<u64>().unwrap();
            if max <= u8::MAX as u64 {
                s.cast(&DataType::UInt8)
            } else if max <= u16::MAX as u64 {
                s.cast(&DataType::UInt16)
            } else if max <= u32::MAX as u64 {
                s.cast(&DataType::UInt32)
            } else {
                Ok(s)
            }
        } else {
            let min = s.min_as_series().get(0).unwrap().extract::<i64>().unwrap();
            let max = s.max_as_series().get(0).unwrap().extract::<i64>().unwrap();

            if min >= i8::MIN as i64 && max <= i8::MAX as i64 {
                s.cast(&DataType::Int8)
            } else if min >= i16::MIN as i64 && max <= i16::MAX as i64 {
                s.cast(&DataType::Int16)
            } else if min >= i32::MIN as i64 && max <= i32::MAX as i64 {
                s.cast(&DataType::Int32)
            } else {
                Ok(s)
            }
        }
    } else {
        Ok(s)
    }
}
