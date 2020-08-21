use std::mem;
use std::ops::{Deref, DerefMut};

/// Used to split the mantissa and exponent of floating point numbers
/// https://stackoverflow.com/questions/39638363/how-can-i-use-a-hashmap-with-f64-as-key-in-rust
pub(crate) fn integer_decode(val: f64) -> (u64, i16, i8) {
    let bits: u64 = unsafe { mem::transmute(val) };
    let sign: i8 = if bits >> 63 == 0 { 1 } else { -1 };
    let mut exponent: i16 = ((bits >> 52) & 0x7ff) as i16;
    let mantissa = if exponent == 0 {
        (bits & 0xfffffffffffff) << 1
    } else {
        (bits & 0xfffffffffffff) | 0x10000000000000
    };

    exponent -= 1023 + 52;
    (mantissa, exponent, sign)
}

pub(crate) fn floating_encode_f64(mantissa: u64, exponent: i16, sign: i8) -> f64 {
    sign as f64 * mantissa as f64 * (2.0f64).powf(exponent as f64)
}

#[macro_export]
macro_rules! exec_concurrent {
    ($block_a:block, $block_b:block) => {{
        thread::scope(|s| {
            let handle_left = s.spawn(|_| $block_a);
            let handle_right = s.spawn(|_| $block_b);
            let return_left = handle_left.join().expect("thread panicked");
            let return_right = handle_right.join().expect("thread panicked");
            (return_left, return_right)
        })
        .expect("could not join threads or thread panicked")
    }};
}

/// Just a wrapper structure. Useful for certain impl specializations
pub struct Xob<T> {
    inner: T,
}

impl<T> Xob<T> {
    pub fn new(inner: T) -> Self {
        Xob { inner }
    }

    pub fn into_inner(self) -> T {
        self.inner
    }
}

impl<T> Deref for Xob<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T> DerefMut for Xob<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

pub fn get_iter_capacity<T, I: Iterator<Item = T>>(iter: &I) -> usize {
    match iter.size_hint() {
        (_lower, Some(upper)) => upper,
        (0, None) => 1024,
        (lower, None) => lower,
    }
}

#[macro_export]
macro_rules! match_arrow_data_type_apply_macro {
    ($obj:expr, $macro:ident, $macro_utf8:ident) => {{
        match $obj {
            ArrowDataType::Utf8 => $macro_utf8!(),
            ArrowDataType::Boolean => $macro!(BooleanType),
            ArrowDataType::UInt8 => $macro!(UInt8Type),
            ArrowDataType::UInt16 => $macro!(UInt16Type),
            ArrowDataType::UInt32 => $macro!(UInt32Type),
            ArrowDataType::UInt64 => $macro!(UInt64Type),
            ArrowDataType::Int8 => $macro!(Int8Type),
            ArrowDataType::Int16 => $macro!(Int16Type),
            ArrowDataType::Int32 => $macro!(Int32Type),
            ArrowDataType::Int64 => $macro!(Int64Type),
            ArrowDataType::Float32 => $macro!(Float32Type),
            ArrowDataType::Float64 => $macro!(Float64Type),
            ArrowDataType::Date32(DateUnit::Day) => $macro!(Date32Type),
            ArrowDataType::Date64(DateUnit::Millisecond) => $macro!(Date64Type),
            ArrowDataType::Time32(TimeUnit::Millisecond) => $macro!(Time32MillisecondType),
            ArrowDataType::Time32(TimeUnit::Second) => $macro!(Time32SecondType),
            ArrowDataType::Time64(TimeUnit::Nanosecond) => $macro!(Time64NanosecondType),
            ArrowDataType::Time64(TimeUnit::Microsecond) => $macro!(Time64MicrosecondType),
            ArrowDataType::Interval(IntervalUnit::DayTime) => $macro!(IntervalDayTimeType),
            ArrowDataType::Interval(IntervalUnit::YearMonth) => $macro!(IntervalYearMonthType),
            ArrowDataType::Duration(TimeUnit::Nanosecond) => $macro!(DurationNanosecondType),
            ArrowDataType::Duration(TimeUnit::Microsecond) => $macro!(DurationMicrosecondType),
            ArrowDataType::Duration(TimeUnit::Millisecond) => $macro!(DurationMillisecondType),
            ArrowDataType::Duration(TimeUnit::Second) => $macro!(DurationSecondType),
            ArrowDataType::Timestamp(TimeUnit::Nanosecond, _) => $macro!(TimestampNanosecondType),
            ArrowDataType::Timestamp(TimeUnit::Microsecond, _) => $macro!(TimestampMicrosecondType),
            ArrowDataType::Timestamp(TimeUnit::Millisecond, _) => $macro!(Time32MillisecondType),
            ArrowDataType::Timestamp(TimeUnit::Second, _) => $macro!(TimestampSecondType),
            _ => unimplemented!(),
        }
    }};
}
