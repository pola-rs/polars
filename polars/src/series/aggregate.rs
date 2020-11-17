use crate::prelude::*;
use num::{NumCast, ToPrimitive, Zero};
use std::ops::Div;

// TODO: implement types
macro_rules! apply_agg_fn {
    ($self:ident, $agg:ident) => {
        match $self {
            Series::Bool(a) => a
                .$agg()
                .map(|v| T::from(v).expect("could not cast bool to T")),
            Series::UInt8(a) => a
                .$agg()
                .map(|v| T::from(v).expect("could not cast u8 to T")),
            Series::UInt16(a) => a
                .$agg()
                .map(|v| T::from(v).expect("could not cast u16 to T")),
            Series::UInt32(a) => a
                .$agg()
                .map(|v| T::from(v).expect("could not cast u32 to T")),
            Series::UInt64(a) => a
                .$agg()
                .map(|v| T::from(v).expect("could not cast u64 to T")),
            Series::Int8(a) => a
                .$agg()
                .map(|v| T::from(v).expect("could not cast i8 to T")),
            Series::Int16(a) => a
                .$agg()
                .map(|v| T::from(v).expect("could not cast i16 to T")),
            Series::Int32(a) => a
                .$agg()
                .map(|v| T::from(v).expect("could not cast i32 to T")),
            Series::Int64(a) => a
                .$agg()
                .map(|v| T::from(v).expect("could not cast i64 to T")),
            Series::Float32(a) => a
                .$agg()
                .map(|v| T::from(v).expect("could not cast f32 to T")),
            Series::Float64(a) => a
                .$agg()
                .map(|v| T::from(v).expect("could not cast f64 to T")),
            Series::Date32(a) => a
                .$agg()
                .map(|v| T::from(v).expect("could not cast Date64 to T")),
            Series::Date64(a) => a
                .$agg()
                .map(|v| T::from(v).expect("could not cast Date64 to T")),
            Series::Time32Millisecond(a) => a
                .$agg()
                .map(|v| T::from(v).expect("could not cast Time32Millisecond to T")),
            Series::Time32Second(a) => a
                .$agg()
                .map(|v| T::from(v).expect("could not cast Time32Second to T")),
            Series::Time64Nanosecond(a) => a
                .$agg()
                .map(|v| T::from(v).expect("could not cast Time64Nanosecond to T")),
            Series::Time64Microsecond(a) => a
                .$agg()
                .map(|v| T::from(v).expect("could not cast Time64Microsecond to T")),
            Series::DurationNanosecond(a) => a
                .$agg()
                .map(|v| T::from(v).expect("could not cast DurationNanosecond to T")),
            Series::DurationMicrosecond(a) => a
                .$agg()
                .map(|v| T::from(v).expect("could not cast DurationMicrosecond to T")),
            Series::DurationMillisecond(a) => a
                .$agg()
                .map(|v| T::from(v).expect("could not cast DurationMillisecond to T")),
            Series::DurationSecond(a) => a
                .$agg()
                .map(|v| T::from(v).expect("could not cast DurationSecond to T")),
            Series::TimestampNanosecond(a) => a
                .$agg()
                .map(|v| T::from(v).expect("could not cast TimestampNanosecond to T")),
            Series::TimestampMicrosecond(a) => a
                .$agg()
                .map(|v| T::from(v).expect("could not cast TimestampMicrosecond to T")),
            Series::TimestampMillisecond(a) => a
                .$agg()
                .map(|v| T::from(v).expect("could not cast TimestampMillisecond to T")),
            Series::TimestampSecond(a) => a
                .$agg()
                .map(|v| T::from(v).expect("could not cast TimestampSecond to T")),
            Series::IntervalDayTime(a) => a
                .$agg()
                .map(|v| T::from(v).expect("could not cast IntervalDayTime to T")),
            Series::IntervalYearMonth(a) => a
                .$agg()
                .map(|v| T::from(v).expect("could not cast IntervalYearMonth to T")),
            Series::Utf8(_a) => None,
            Series::List(_a) => None,
            Series::Object(_a) => None,
        }
    };
}

impl Series {
    /// Returns `None` if the array is empty or only contains null values.
    /// ```
    /// # use polars::prelude::*;
    /// let s = Series::new("days", [1, 2, 3].as_ref());
    /// assert_eq!(s.sum(), Some(6));
    /// ```
    pub fn sum<T>(&self) -> Option<T>
    where
        T: NumCast + Zero + ToPrimitive,
    {
        apply_agg_fn!(self, sum)
    }

    /// Returns the minimum value in the array, according to the natural order.
    /// Returns an option because the array is nullable.
    /// ```
    /// # use polars::prelude::*;
    /// let s = Series::new("days", [1, 2, 3].as_ref());
    /// assert_eq!(s.min(), Some(1));
    /// ```
    pub fn min<T>(&self) -> Option<T>
    where
        T: NumCast + Zero + ToPrimitive,
    {
        apply_agg_fn!(self, min)
    }

    /// Returns the maximum value in the array, according to the natural order.
    /// Returns an option because the array is nullable.
    /// ```
    /// # use polars::prelude::*;
    /// let s = Series::new("days", [1, 2, 3].as_ref());
    /// assert_eq!(s.max(), Some(3));
    /// ```
    pub fn max<T>(&self) -> Option<T>
    where
        T: NumCast + Zero + ToPrimitive,
    {
        apply_agg_fn!(self, max)
    }

    pub fn mean<T>(&self) -> Option<T>
    where
        T: NumCast + Zero + ToPrimitive + Div<Output = T>,
    {
        apply_agg_fn!(self, sum).map(|v| v / T::from(self.len()).unwrap())
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    fn test_agg_bool() {
        let s = Series::new("", vec![true, false, true].as_slice());
        assert_eq!(s.max::<u8>(), Some(1));
        assert_eq!(s.min::<u8>(), Some(0));
    }
}
