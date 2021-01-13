//! Comparison operations on Series.

use super::Series;
use crate::apply_method_numeric_series;
use crate::prelude::*;
use crate::series::arithmetic::coerce_lhs_rhs;

fn fill_bool(val: bool, len: usize) -> BooleanChunked {
    std::iter::repeat(val).take(len).collect()
}

macro_rules! impl_compare {
    ($self:expr, $rhs:expr, $method:ident) => {{
        match $self.dtype() {
            DataType::Boolean => $self.bool().unwrap().$method($rhs.bool().unwrap()),
            DataType::Utf8 => $self.utf8().unwrap().$method($rhs.utf8().unwrap()),
            DataType::UInt8 => $self.u8().unwrap().$method($rhs.u8().unwrap()),
            DataType::UInt16 => $self.u16().unwrap().$method($rhs.u16().unwrap()),
            DataType::UInt32 => $self.u32().unwrap().$method($rhs.u32().unwrap()),
            DataType::UInt64 => $self.u64().unwrap().$method($rhs.u64().unwrap()),
            DataType::Int8 => $self.i8().unwrap().$method($rhs.i8().unwrap()),
            DataType::Int16 => $self.i16().unwrap().$method($rhs.i16().unwrap()),
            DataType::Int32 => $self.i32().unwrap().$method($rhs.i32().unwrap()),
            DataType::Int64 => $self.i64().unwrap().$method($rhs.i64().unwrap()),
            DataType::Float32 => $self.f32().unwrap().$method($rhs.f32().unwrap()),
            DataType::Float64 => $self.f64().unwrap().$method($rhs.f64().unwrap()),
            DataType::Date32 => $self.date32().unwrap().$method($rhs.date32().unwrap()),
            DataType::Date64 => $self.date64().unwrap().$method($rhs.date64().unwrap()),
            DataType::Time64(TimeUnit::Nanosecond) => $self
                .time64_nanosecond()
                .unwrap()
                .$method($rhs.time64_nanosecond().unwrap()),
            DataType::Duration(TimeUnit::Nanosecond) => $self
                .duration_nanosecond()
                .unwrap()
                .$method($rhs.duration_nanosecond().unwrap()),
            DataType::Duration(TimeUnit::Millisecond) => $self
                .duration_millisecond()
                .unwrap()
                .$method($rhs.duration_millisecond().unwrap()),
            DataType::List(_) => $self.list().unwrap().$method($rhs.list().unwrap()),
            _ => unimplemented!(),
        }
    }};
}

impl ChunkCompare<&Series> for Series {
    fn eq_missing(&self, rhs: &Series) -> BooleanChunked {
        let (lhs, rhs) = coerce_lhs_rhs(self, rhs).expect("cannot coerce datatypes");
        impl_compare!(lhs.as_ref(), rhs.as_ref(), eq_missing)
    }

    /// Create a boolean mask by checking for equality.
    fn eq(&self, rhs: &Series) -> BooleanChunked {
        let (lhs, rhs) = coerce_lhs_rhs(self, rhs).expect("cannot coerce datatypes");
        impl_compare!(lhs.as_ref(), rhs.as_ref(), eq)
    }

    /// Create a boolean mask by checking for inequality.
    fn neq(&self, rhs: &Series) -> BooleanChunked {
        let (lhs, rhs) = coerce_lhs_rhs(self, rhs).expect("cannot coerce datatypes");
        impl_compare!(lhs.as_ref(), rhs.as_ref(), neq)
    }

    /// Create a boolean mask by checking if lhs > rhs.
    fn gt(&self, rhs: &Series) -> BooleanChunked {
        let (lhs, rhs) = coerce_lhs_rhs(self, rhs).expect("cannot coerce datatypes");
        impl_compare!(lhs.as_ref(), rhs.as_ref(), gt)
    }

    /// Create a boolean mask by checking if lhs >= rhs.
    fn gt_eq(&self, rhs: &Series) -> BooleanChunked {
        let (lhs, rhs) = coerce_lhs_rhs(self, rhs).expect("cannot coerce datatypes");
        impl_compare!(lhs.as_ref(), rhs.as_ref(), gt_eq)
    }

    /// Create a boolean mask by checking if lhs < rhs.
    fn lt(&self, rhs: &Series) -> BooleanChunked {
        let (lhs, rhs) = coerce_lhs_rhs(self, rhs).expect("cannot coerce datatypes");
        impl_compare!(lhs.as_ref(), rhs.as_ref(), lt)
    }

    /// Create a boolean mask by checking if lhs <= rhs.
    fn lt_eq(&self, rhs: &Series) -> BooleanChunked {
        let (lhs, rhs) = coerce_lhs_rhs(self, rhs).expect("cannot coerce datatypes");
        impl_compare!(lhs.as_ref(), rhs.as_ref(), lt_eq)
    }
}

impl<Rhs> ChunkCompare<Rhs> for Series
where
    Rhs: NumComp,
{
    fn eq_missing(&self, rhs: Rhs) -> BooleanChunked {
        self.eq(rhs)
    }

    fn eq(&self, rhs: Rhs) -> BooleanChunked {
        apply_method_numeric_series!(self, eq, rhs)
    }

    fn neq(&self, rhs: Rhs) -> BooleanChunked {
        apply_method_numeric_series!(self, neq, rhs)
    }

    fn gt(&self, rhs: Rhs) -> BooleanChunked {
        apply_method_numeric_series!(self, gt, rhs)
    }

    fn gt_eq(&self, rhs: Rhs) -> BooleanChunked {
        apply_method_numeric_series!(self, gt_eq, rhs)
    }

    fn lt(&self, rhs: Rhs) -> BooleanChunked {
        apply_method_numeric_series!(self, lt, rhs)
    }

    fn lt_eq(&self, rhs: Rhs) -> BooleanChunked {
        apply_method_numeric_series!(self, lt_eq, rhs)
    }
}

impl ChunkCompare<&str> for Series {
    fn eq_missing(&self, rhs: &str) -> BooleanChunked {
        self.eq(rhs)
    }

    fn eq(&self, rhs: &str) -> BooleanChunked {
        if let Ok(a) = self.utf8() {
            a.eq(rhs)
        } else {
            std::iter::repeat(false).take(self.len()).collect()
        }
    }

    fn neq(&self, rhs: &str) -> BooleanChunked {
        if let Ok(a) = self.utf8() {
            a.neq(rhs)
        } else {
            std::iter::repeat(false).take(self.len()).collect()
        }
    }

    fn gt(&self, rhs: &str) -> BooleanChunked {
        if let Ok(a) = self.utf8() {
            a.gt(rhs)
        } else {
            std::iter::repeat(false).take(self.len()).collect()
        }
    }

    fn gt_eq(&self, rhs: &str) -> BooleanChunked {
        if let Ok(a) = self.utf8() {
            a.gt_eq(rhs)
        } else {
            std::iter::repeat(false).take(self.len()).collect()
        }
    }

    fn lt(&self, rhs: &str) -> BooleanChunked {
        if let Ok(a) = self.utf8() {
            a.lt(rhs)
        } else {
            std::iter::repeat(false).take(self.len()).collect()
        }
    }

    fn lt_eq(&self, rhs: &str) -> BooleanChunked {
        if let Ok(a) = self.utf8() {
            a.lt_eq(rhs)
        } else {
            std::iter::repeat(false).take(self.len()).collect()
        }
    }
}
