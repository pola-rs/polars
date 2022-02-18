//! Comparison operations on Series.

use super::Series;
use crate::apply_method_numeric_series;
use crate::prelude::*;
use crate::series::arithmetic::coerce_lhs_rhs;
use std::ops::Deref;

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
            #[cfg(feature = "dtype-date")]
            DataType::Date => lhs.date().unwrap().$method(rhs.date().unwrap().deref()),
            #[cfg(feature = "dtype-time")]
            DataType::Time => lhs.time().unwrap().$method(rhs.time().unwrap().deref()),
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime(_, _) => lhs
                .datetime()
                .unwrap()
                .$method(rhs.datetime().unwrap().deref()),
            #[cfg(feature = "dtype-duration")]
            DataType::Duration(_) => lhs
                .duration()
                .unwrap()
                .$method(rhs.duration().unwrap().deref()),
            DataType::List(_) => lhs.list().unwrap().$method(rhs.list().unwrap()),
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical(_) => lhs
                .categorical()
                .unwrap()
                .logical()
                .$method(rhs.categorical().unwrap().logical()),

            _ => unimplemented!(),
        }
    }};
}

#[cfg(feature = "dtype-categorical")]
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
    let cat_map = cat.get_rev_map();
    match cat_map.find(value) {
        None => BooleanChunked::full(name, fill_value, cat.len()),
        Some(cat_idx) => {
            let cat = cat.cast(&DataType::UInt32).unwrap();
            compare(&cat, cat_idx)
        }
    }
}

#[cfg(feature = "dtype-categorical")]
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
        #[cfg(feature = "dtype-categorical")]
        use DataType::*;
        match (self.dtype(), rhs.dtype(), self.len(), rhs.len()) {
            #[cfg(feature = "dtype-categorical")]
            (Categorical(_), Utf8, _, 1) => {
                return compare_cat_to_str_series(
                    self,
                    rhs,
                    self.name(),
                    |s, idx| s.eq_missing(idx),
                    false,
                );
            }
            #[cfg(feature = "dtype-categorical")]
            (Utf8, Categorical(_), 1, _) => {
                return compare_cat_to_str_series(
                    rhs,
                    self,
                    self.name(),
                    |s, idx| s.eq_missing(idx),
                    false,
                );
            }
            _ => {
                impl_compare!(self, rhs, eq_missing)
            }
        }
    }

    /// Create a boolean mask by checking for equality.
    fn equal(&self, rhs: &Series) -> BooleanChunked {
        #[cfg(feature = "dtype-categorical")]
        use DataType::*;
        let mut out = match (self.dtype(), rhs.dtype(), self.len(), rhs.len()) {
            #[cfg(feature = "dtype-categorical")]
            (Categorical(_), Utf8, _, 1) => {
                return compare_cat_to_str_series(
                    self,
                    rhs,
                    self.name(),
                    |s, idx| s.equal(idx),
                    false,
                );
            }
            #[cfg(feature = "dtype-categorical")]
            (Utf8, Categorical(_), 1, _) => {
                return compare_cat_to_str_series(
                    rhs,
                    self,
                    self.name(),
                    |s, idx| s.equal(idx),
                    false,
                );
            }
            _ => {
                impl_compare!(self, rhs, equal)
            }
        };
        out.rename(self.name());
        out
    }

    /// Create a boolean mask by checking for inequality.
    fn not_equal(&self, rhs: &Series) -> BooleanChunked {
        #[cfg(feature = "dtype-categorical")]
        use DataType::*;
        let mut out = match (self.dtype(), rhs.dtype(), self.len(), rhs.len()) {
            #[cfg(feature = "dtype-categorical")]
            (Categorical(_), Utf8, _, 1) => {
                return compare_cat_to_str_series(
                    self,
                    rhs,
                    self.name(),
                    |s, idx| s.not_equal(idx),
                    true,
                );
            }
            #[cfg(feature = "dtype-categorical")]
            (Utf8, Categorical(_), 1, _) => {
                return compare_cat_to_str_series(
                    rhs,
                    self,
                    self.name(),
                    |s, idx| s.not_equal(idx),
                    true,
                );
            }
            _ => {
                impl_compare!(self, rhs, not_equal)
            }
        };
        out.rename(self.name());
        out
    }

    /// Create a boolean mask by checking if self > rhs.
    fn gt(&self, rhs: &Series) -> BooleanChunked {
        let mut out = impl_compare!(self, rhs, gt);
        out.rename(self.name());
        out
    }

    /// Create a boolean mask by checking if self >= rhs.
    fn gt_eq(&self, rhs: &Series) -> BooleanChunked {
        let mut out = impl_compare!(self, rhs, gt_eq);
        out.rename(self.name());
        out
    }

    /// Create a boolean mask by checking if self < rhs.
    fn lt(&self, rhs: &Series) -> BooleanChunked {
        let mut out = impl_compare!(self, rhs, lt);
        out.rename(self.name());
        out
    }

    /// Create a boolean mask by checking if self <= rhs.
    fn lt_eq(&self, rhs: &Series) -> BooleanChunked {
        let mut out = impl_compare!(self, rhs, lt_eq);
        out.rename(self.name());
        out
    }
}

impl<Rhs> ChunkCompare<Rhs> for Series
where
    Rhs: NumericNative,
{
    fn eq_missing(&self, rhs: Rhs) -> BooleanChunked {
        self.equal(rhs)
    }

    fn equal(&self, rhs: Rhs) -> BooleanChunked {
        apply_method_numeric_series!(self, equal, rhs)
    }

    fn not_equal(&self, rhs: Rhs) -> BooleanChunked {
        apply_method_numeric_series!(self, not_equal, rhs)
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
        self.equal(rhs)
    }

    fn equal(&self, rhs: &str) -> BooleanChunked {
        use DataType::*;
        match self.dtype() {
            Utf8 => self.utf8().unwrap().equal(rhs),
            #[cfg(feature = "dtype-categorical")]
            Categorical(_) => {
                compare_cat_to_str_value(self, rhs, self.name(), |lhs, idx| lhs.equal(idx), false)
            }
            _ => BooleanChunked::full(self.name(), false, self.len()),
        }
    }

    fn not_equal(&self, rhs: &str) -> BooleanChunked {
        use DataType::*;
        match self.dtype() {
            Utf8 => self.utf8().unwrap().not_equal(rhs),
            #[cfg(feature = "dtype-categorical")]
            Categorical(_) => compare_cat_to_str_value(
                self,
                rhs,
                self.name(),
                |lhs, idx| lhs.not_equal(idx),
                true,
            ),
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
