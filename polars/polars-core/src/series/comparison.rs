//! Comparison operations on Series.

use super::Series;
use crate::apply_method_numeric_series;
use crate::prelude::*;
use crate::series::arithmetic::coerce_lhs_rhs;

macro_rules! impl_compare {
    ($self:expr, $rhs:expr, $method:ident) => {{
        let (lhs, rhs) = coerce_lhs_rhs($self, $rhs).expect("cannot coerce datatypes");
        let lhs = lhs.as_ref();
        let rhs = rhs.as_ref();
        match lhs.dtype() {
            DataType::Boolean => lhs.bool().unwrap().$method(rhs.bool().unwrap()),
            DataType::Utf8 => lhs.utf8().unwrap().$method(rhs.utf8().unwrap()),
            DataType::UInt8 => lhs.u8().unwrap().$method(rhs.u8().unwrap()),
            DataType::UInt16 => lhs.u16().unwrap().$method(rhs.u16().unwrap()),
            DataType::UInt32 => lhs.u32().unwrap().$method(rhs.u32().unwrap()),
            DataType::UInt64 => lhs.u64().unwrap().$method(rhs.u64().unwrap()),
            DataType::Int8 => lhs.i8().unwrap().$method(rhs.i8().unwrap()),
            DataType::Int16 => lhs.i16().unwrap().$method(rhs.i16().unwrap()),
            DataType::Int32 => lhs.i32().unwrap().$method(rhs.i32().unwrap()),
            DataType::Int64 => lhs.i64().unwrap().$method(rhs.i64().unwrap()),
            DataType::Float32 => lhs.f32().unwrap().$method(rhs.f32().unwrap()),
            DataType::Float64 => lhs.f64().unwrap().$method(rhs.f64().unwrap()),
            DataType::Date32 => lhs.date32().unwrap().$method(rhs.date32().unwrap()),
            DataType::Date64 => lhs.date64().unwrap().$method(rhs.date64().unwrap()),
            DataType::Time64(TimeUnit::Nanosecond) => lhs
                .time64_nanosecond()
                .unwrap()
                .$method(rhs.time64_nanosecond().unwrap()),
            DataType::Duration(TimeUnit::Nanosecond) => lhs
                .duration_nanosecond()
                .unwrap()
                .$method(rhs.duration_nanosecond().unwrap()),
            DataType::Duration(TimeUnit::Millisecond) => lhs
                .duration_millisecond()
                .unwrap()
                .$method(rhs.duration_millisecond().unwrap()),
            DataType::List(_) => lhs.list().unwrap().$method(rhs.list().unwrap()),
            _ => unimplemented!(),
        }
    }};
}

fn compare_cat_to_str_value<Compare>(
    cat: &Series,
    value: &str,
    name: &str,
    compare: Compare,
    fill_value: bool,
) -> BooleanChunked
where
    Compare: Fn(&Series, u32) -> BooleanChunked,
{
    let cat = cat.categorical().expect("should be categorical");
    let cat_map = cat.get_categorical_map().unwrap();
    match cat_map.find(value) {
        None => BooleanChunked::full(name, fill_value, cat.len()),
        Some(cat_idx) => {
            let cat = cat.cast_with_dtype(&DataType::UInt32).unwrap();
            compare(&cat, cat_idx)
        }
    }
}

fn compare_cat_to_str_series<Compare>(
    cat: &Series,
    string: &Series,
    name: &str,
    compare: Compare,
    fill_value: bool,
) -> BooleanChunked
where
    Compare: Fn(&Series, u32) -> BooleanChunked,
{
    match string.utf8().expect("should be utf8 column").get(0) {
        None => cat.is_null(),
        Some(value) => compare_cat_to_str_value(cat, value, name, compare, fill_value),
    }
}

impl ChunkCompare<&Series> for Series {
    fn eq_missing(&self, rhs: &Series) -> BooleanChunked {
        impl_compare!(self, rhs, eq_missing)
    }

    /// Create a boolean mask by checking for equality.
    fn eq(&self, rhs: &Series) -> BooleanChunked {
        use DataType::*;
        match (self.dtype(), rhs.dtype(), self.len(), rhs.len()) {
            (Categorical, Utf8, _, 1) => {
                return compare_cat_to_str_series(
                    self,
                    rhs,
                    self.name(),
                    |s, idx| s.eq(idx),
                    false,
                );
            }
            (Utf8, Categorical, 1, _) => {
                return compare_cat_to_str_series(
                    rhs,
                    self,
                    self.name(),
                    |s, idx| s.eq(idx),
                    false,
                );
            }
            _ => {
                impl_compare!(self, rhs, eq)
            }
        }
    }

    /// Create a boolean mask by checking for inequality.
    fn neq(&self, rhs: &Series) -> BooleanChunked {
        use DataType::*;
        match (self.dtype(), rhs.dtype(), self.len(), rhs.len()) {
            (Categorical, Utf8, _, 1) => {
                return compare_cat_to_str_series(
                    self,
                    rhs,
                    self.name(),
                    |s, idx| s.neq(idx),
                    true,
                );
            }
            (Utf8, Categorical, 1, _) => {
                return compare_cat_to_str_series(
                    rhs,
                    self,
                    self.name(),
                    |s, idx| s.neq(idx),
                    true,
                );
            }
            _ => {
                impl_compare!(self, rhs, neq)
            }
        }
    }

    /// Create a boolean mask by checking if self > rhs.
    fn gt(&self, rhs: &Series) -> BooleanChunked {
        impl_compare!(self, rhs, gt)
    }

    /// Create a boolean mask by checking if self >= rhs.
    fn gt_eq(&self, rhs: &Series) -> BooleanChunked {
        impl_compare!(self, rhs, gt_eq)
    }

    /// Create a boolean mask by checking if self < rhs.
    fn lt(&self, rhs: &Series) -> BooleanChunked {
        impl_compare!(self, rhs, lt)
    }

    /// Create a boolean mask by checking if self <= rhs.
    fn lt_eq(&self, rhs: &Series) -> BooleanChunked {
        impl_compare!(self, rhs, lt_eq)
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
        use DataType::*;
        match self.dtype() {
            Utf8 => self.utf8().unwrap().eq(rhs),
            Categorical => {
                compare_cat_to_str_value(self, rhs, self.name(), |lhs, idx| lhs.eq(idx), false)
            }
            _ => BooleanChunked::full(self.name(), false, self.len()),
        }
    }

    fn neq(&self, rhs: &str) -> BooleanChunked {
        use DataType::*;
        match self.dtype() {
            Utf8 => self.utf8().unwrap().neq(rhs),
            Categorical => {
                compare_cat_to_str_value(self, rhs, self.name(), |lhs, idx| lhs.neq(idx), true)
            }
            _ => BooleanChunked::full(self.name(), false, self.len()),
        }
    }

    fn gt(&self, rhs: &str) -> BooleanChunked {
        if let Ok(a) = self.utf8() {
            a.gt(rhs)
        } else {
            BooleanChunked::full(self.name(), false, self.len())
        }
    }

    fn gt_eq(&self, rhs: &str) -> BooleanChunked {
        if let Ok(a) = self.utf8() {
            a.gt_eq(rhs)
        } else {
            BooleanChunked::full(self.name(), false, self.len())
        }
    }

    fn lt(&self, rhs: &str) -> BooleanChunked {
        if let Ok(a) = self.utf8() {
            a.lt(rhs)
        } else {
            BooleanChunked::full(self.name(), false, self.len())
        }
    }

    fn lt_eq(&self, rhs: &str) -> BooleanChunked {
        if let Ok(a) = self.utf8() {
            a.lt_eq(rhs)
        } else {
            BooleanChunked::full(self.name(), false, self.len())
        }
    }
}
