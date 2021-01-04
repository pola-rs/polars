use crate::chunked_array::kernels::{is_nan, is_not_nan};
use crate::{
    prelude::*,
    utils::{integer_decode_f32, integer_decode_f64},
};
use num::Float;

pub trait ChunkIntegerDecode {
    fn integer_decode(&self) -> (UInt64Chunked, Int16Chunked, Int8Chunked);
}
pub trait IntegerDecode {
    fn integer_decode(&self) -> (u64, i16, i8);
}

impl IntegerDecode for f64 {
    fn integer_decode(&self) -> (u64, i16, i8) {
        integer_decode_f64(*self)
    }
}

impl IntegerDecode for f32 {
    fn integer_decode(&self) -> (u64, i16, i8) {
        integer_decode_f32(*self)
    }
}

fn process_float<T>(
    val: T,
    u64_builder: &mut PrimitiveChunkedBuilder<UInt64Type>,
    i16_builder: &mut PrimitiveChunkedBuilder<Int16Type>,
    i8_builder: &mut PrimitiveChunkedBuilder<Int8Type>,
) where
    T: IntegerDecode,
{
    let (mantissa, exponent, sign) = val.integer_decode();
    u64_builder.append_value(mantissa);
    i16_builder.append_value(exponent);
    i8_builder.append_value(sign);
}

impl<T> ChunkIntegerDecode for ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: IntegerDecode,
{
    fn integer_decode(&self) -> (UInt64Chunked, Int16Chunked, Int8Chunked) {
        let name = self.name();
        let name_len = name.len();
        let mut u64_name = String::with_capacity(name_len + 3);
        u64_name.push_str(name);
        u64_name.push_str("u64");
        let mut i16_name = String::with_capacity(name_len + 3);
        i16_name.push_str(name);
        i16_name.push_str("i16");
        let mut i8_name = String::with_capacity(name_len + 3);
        i8_name.push_str(name);
        i8_name.push_str("i16");

        let mut u64_builder = PrimitiveChunkedBuilder::<UInt64Type>::new(&u64_name, self.len());
        let mut i16_builder = PrimitiveChunkedBuilder::<Int16Type>::new(&i16_name, self.len());
        let mut i8_builder = PrimitiveChunkedBuilder::<Int8Type>::new(&i8_name, self.len());

        match self.null_count() {
            0 => self.into_no_null_iter().for_each(|v| {
                process_float(v, &mut u64_builder, &mut i16_builder, &mut i8_builder)
            }),
            _ => self.into_iter().for_each(|opt_v| {
                if let Some(v) = opt_v {
                    process_float(v, &mut u64_builder, &mut i16_builder, &mut i8_builder)
                }
            }),
        }
        (
            u64_builder.finish(),
            i16_builder.finish(),
            i8_builder.finish(),
        )
    }
}

pub trait IsNan {
    fn is_nan(&self) -> BooleanChunked;
    fn is_not_nan(&self) -> BooleanChunked;
}

impl<T> IsNan for ChunkedArray<T>
where
    T: PolarsFloatType,
    T::Native: Float,
{
    fn is_nan(&self) -> BooleanChunked {
        self.apply_kernel_cast(is_nan)
    }
    fn is_not_nan(&self) -> BooleanChunked {
        self.apply_kernel_cast(is_not_nan)
    }
}
