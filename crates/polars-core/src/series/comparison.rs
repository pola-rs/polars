//! Comparison operations on Series.

#[cfg(feature = "dtype-struct")]
use std::ops::Deref;

use super::Series;
use crate::apply_method_physical_numeric;
use crate::prelude::*;
use crate::series::arithmetic::coerce_lhs_rhs;
use crate::series::nulls::replace_non_null;

macro_rules! impl_compare {
    ($self:expr, $rhs:expr, $method:ident) => {{
        use DataType::*;
        let (lhs, rhs) = ($self, $rhs);
        validate_types(lhs.dtype(), rhs.dtype())?;

        #[cfg(feature = "dtype-categorical")]
        match (lhs.dtype(), rhs.dtype()) {
            (Categorical(_, _), Categorical(_, _)) => {
                return Ok(lhs
                    .categorical()
                    .unwrap()
                    .$method(rhs.categorical().unwrap())?
                    .with_name(lhs.name()));
            },
            (Categorical(_, _), Utf8) => {
                return Ok(lhs
                    .categorical()
                    .unwrap()
                    .$method(rhs.utf8().unwrap())?
                    .with_name(lhs.name()));
            },
            (Utf8, Categorical(_, _)) => {
                return Ok(rhs
                    .categorical()
                    .unwrap()
                    .$method(lhs.utf8().unwrap())?
                    .with_name(lhs.name()));
            },
            _ => (),
        };

        let (lhs, rhs) = coerce_lhs_rhs(lhs, rhs).expect("cannot coerce datatypes");
        let lhs = lhs.to_physical_repr();
        let rhs = rhs.to_physical_repr();
        let mut out = match lhs.dtype() {
            Boolean => lhs.bool().unwrap().$method(rhs.bool().unwrap()),
            Utf8 => lhs.utf8().unwrap().$method(rhs.utf8().unwrap()),
            Binary => lhs.binary().unwrap().$method(rhs.binary().unwrap()),
            UInt8 => lhs.u8().unwrap().$method(rhs.u8().unwrap()),
            UInt16 => lhs.u16().unwrap().$method(rhs.u16().unwrap()),
            UInt32 => lhs.u32().unwrap().$method(rhs.u32().unwrap()),
            UInt64 => lhs.u64().unwrap().$method(rhs.u64().unwrap()),
            Int8 => lhs.i8().unwrap().$method(rhs.i8().unwrap()),
            Int16 => lhs.i16().unwrap().$method(rhs.i16().unwrap()),
            Int32 => lhs.i32().unwrap().$method(rhs.i32().unwrap()),
            Int64 => lhs.i64().unwrap().$method(rhs.i64().unwrap()),
            Float32 => lhs.f32().unwrap().$method(rhs.f32().unwrap()),
            Float64 => lhs.f64().unwrap().$method(rhs.f64().unwrap()),
            List(_) => lhs.list().unwrap().$method(rhs.list().unwrap()),
            #[cfg(feature = "dtype-array")]
            Array(_, _) => lhs.array().unwrap().$method(rhs.array().unwrap()),
            #[cfg(feature = "dtype-struct")]
            Struct(_) => lhs
                .struct_()
                .unwrap()
                .$method(rhs.struct_().unwrap().deref()),

            _ => unimplemented!(),
        };
        out.rename(lhs.name());
        Ok(out) as PolarsResult<BooleanChunked>
    }};
}

fn validate_types(left: &DataType, right: &DataType) -> PolarsResult<()> {
    use DataType::*;
    #[cfg(feature = "dtype-categorical")]
    {
        let mismatch = matches!(left, Utf8 | Categorical(_, _)) && right.is_numeric()
            || left.is_numeric() && matches!(right, Utf8 | Categorical(_, _));
        polars_ensure!(!mismatch, ComputeError: "cannot compare utf-8 with numeric data");
    }
    #[cfg(not(feature = "dtype-categorical"))]
    {
        let mismatch = matches!(left, Utf8) && right.is_numeric()
            || left.is_numeric() && matches!(right, Utf8);
        polars_ensure!(!mismatch, ComputeError: "cannot compare utf-8 with numeric data");
    }
    Ok(())
}

impl ChunkCompare<&Series> for Series {
    type Item = PolarsResult<BooleanChunked>;

    /// Create a boolean mask by checking for equality.
    fn equal(&self, rhs: &Series) -> PolarsResult<BooleanChunked> {
        match (self.dtype(), rhs.dtype()) {
            (DataType::Null, DataType::Null) => {
                Ok(BooleanChunked::full_null(self.name(), self.len()))
            },
            _ => impl_compare!(self, rhs, equal),
        }
    }

    /// Create a boolean mask by checking for equality.
    fn equal_missing(&self, rhs: &Series) -> PolarsResult<BooleanChunked> {
        match (self.dtype(), rhs.dtype()) {
            (DataType::Null, DataType::Null) => {
                Ok(BooleanChunked::full(self.name(), true, self.len()))
            },
            _ => impl_compare!(self, rhs, equal_missing),
        }
    }

    /// Create a boolean mask by checking for inequality.
    fn not_equal(&self, rhs: &Series) -> PolarsResult<BooleanChunked> {
        match (self.dtype(), rhs.dtype()) {
            (DataType::Null, DataType::Null) => {
                Ok(BooleanChunked::full_null(self.name(), self.len()))
            },
            _ => impl_compare!(self, rhs, not_equal),
        }
    }

    /// Create a boolean mask by checking for inequality.
    fn not_equal_missing(&self, rhs: &Series) -> PolarsResult<BooleanChunked> {
        match (self.dtype(), rhs.dtype()) {
            (DataType::Null, DataType::Null) => {
                Ok(BooleanChunked::full(self.name(), false, self.len()))
            },
            _ => impl_compare!(self, rhs, not_equal_missing),
        }
    }

    /// Create a boolean mask by checking if self > rhs.
    fn gt(&self, rhs: &Series) -> PolarsResult<BooleanChunked> {
        impl_compare!(self, rhs, gt)
    }

    /// Create a boolean mask by checking if self >= rhs.
    fn gt_eq(&self, rhs: &Series) -> PolarsResult<BooleanChunked> {
        impl_compare!(self, rhs, gt_eq)
    }

    /// Create a boolean mask by checking if self < rhs.
    fn lt(&self, rhs: &Series) -> PolarsResult<BooleanChunked> {
        impl_compare!(self, rhs, lt)
    }

    /// Create a boolean mask by checking if self <= rhs.
    fn lt_eq(&self, rhs: &Series) -> PolarsResult<BooleanChunked> {
        impl_compare!(self, rhs, lt_eq)
    }
}

impl<Rhs> ChunkCompare<Rhs> for Series
where
    Rhs: NumericNative,
{
    type Item = PolarsResult<BooleanChunked>;

    fn equal(&self, rhs: Rhs) -> PolarsResult<BooleanChunked> {
        validate_types(self.dtype(), &DataType::Int8)?;
        let s = self.to_physical_repr();
        Ok(apply_method_physical_numeric!(&s, equal, rhs))
    }

    fn equal_missing(&self, rhs: Rhs) -> Self::Item {
        validate_types(self.dtype(), &DataType::Int8)?;
        let s = self.to_physical_repr();
        Ok(apply_method_physical_numeric!(&s, equal_missing, rhs))
    }

    fn not_equal(&self, rhs: Rhs) -> PolarsResult<BooleanChunked> {
        validate_types(self.dtype(), &DataType::Int8)?;
        let s = self.to_physical_repr();
        Ok(apply_method_physical_numeric!(&s, not_equal, rhs))
    }

    fn not_equal_missing(&self, rhs: Rhs) -> Self::Item {
        validate_types(self.dtype(), &DataType::Int8)?;
        let s = self.to_physical_repr();
        Ok(apply_method_physical_numeric!(&s, not_equal_missing, rhs))
    }

    fn gt(&self, rhs: Rhs) -> PolarsResult<BooleanChunked> {
        validate_types(self.dtype(), &DataType::Int8)?;
        let s = self.to_physical_repr();
        Ok(apply_method_physical_numeric!(&s, gt, rhs))
    }

    fn gt_eq(&self, rhs: Rhs) -> PolarsResult<BooleanChunked> {
        validate_types(self.dtype(), &DataType::Int8)?;
        let s = self.to_physical_repr();
        Ok(apply_method_physical_numeric!(&s, gt_eq, rhs))
    }

    fn lt(&self, rhs: Rhs) -> PolarsResult<BooleanChunked> {
        validate_types(self.dtype(), &DataType::Int8)?;
        let s = self.to_physical_repr();
        Ok(apply_method_physical_numeric!(&s, lt, rhs))
    }

    fn lt_eq(&self, rhs: Rhs) -> PolarsResult<BooleanChunked> {
        validate_types(self.dtype(), &DataType::Int8)?;
        let s = self.to_physical_repr();
        Ok(apply_method_physical_numeric!(&s, lt_eq, rhs))
    }
}

impl ChunkCompare<&str> for Series {
    type Item = PolarsResult<BooleanChunked>;

    fn equal(&self, rhs: &str) -> PolarsResult<BooleanChunked> {
        validate_types(self.dtype(), &DataType::Utf8)?;
        match self.dtype() {
            DataType::Utf8 => Ok(self.utf8().unwrap().equal(rhs)),
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical(_, _) => self.categorical().unwrap().equal(rhs),
            _ => Ok(BooleanChunked::full(self.name(), false, self.len())),
        }
    }

    fn equal_missing(&self, rhs: &str) -> Self::Item {
        validate_types(self.dtype(), &DataType::Utf8)?;
        match self.dtype() {
            DataType::Utf8 => Ok(self.utf8().unwrap().equal_missing(rhs)),
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical(_, _) => self.categorical().unwrap().equal_missing(rhs),
            _ => Ok(replace_non_null(self.name(), self.0.chunks(), false)),
        }
    }

    fn not_equal(&self, rhs: &str) -> PolarsResult<BooleanChunked> {
        validate_types(self.dtype(), &DataType::Utf8)?;
        match self.dtype() {
            DataType::Utf8 => Ok(self.utf8().unwrap().not_equal(rhs)),
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical(_, _) => self.categorical().unwrap().not_equal(rhs),
            _ => Ok(BooleanChunked::full(self.name(), true, self.len())),
        }
    }

    fn not_equal_missing(&self, rhs: &str) -> Self::Item {
        validate_types(self.dtype(), &DataType::Utf8)?;
        match self.dtype() {
            DataType::Utf8 => Ok(self.utf8().unwrap().not_equal_missing(rhs)),
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical(_, _) => self.categorical().unwrap().not_equal_missing(rhs),
            _ => Ok(replace_non_null(self.name(), self.0.chunks(), true)),
        }
    }

    fn gt(&self, rhs: &str) -> PolarsResult<BooleanChunked> {
        validate_types(self.dtype(), &DataType::Utf8)?;
        match self.dtype() {
            DataType::Utf8 => Ok(self.utf8().unwrap().gt(rhs)),
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical(_, _) => self.categorical().unwrap().gt(rhs),
            _ => polars_bail!(
                ComputeError: "cannot compare str value to series of type {}", self.dtype(),
            ),
        }
    }

    fn gt_eq(&self, rhs: &str) -> Self::Item {
        validate_types(self.dtype(), &DataType::Utf8)?;
        match self.dtype() {
            DataType::Utf8 => Ok(self.utf8().unwrap().gt_eq(rhs)),
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical(_, _) => self.categorical().unwrap().gt_eq(rhs),
            _ => polars_bail!(
                ComputeError: "cannot compare str value to series of type {}", self.dtype(),
            ),
        }
    }

    fn lt(&self, rhs: &str) -> Self::Item {
        validate_types(self.dtype(), &DataType::Utf8)?;
        match self.dtype() {
            DataType::Utf8 => Ok(self.utf8().unwrap().lt(rhs)),
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical(_, _) => self.categorical().unwrap().lt(rhs),
            _ => polars_bail!(
                ComputeError: "cannot compare str value to series of type {}", self.dtype(),
            ),
        }
    }

    fn lt_eq(&self, rhs: &str) -> Self::Item {
        validate_types(self.dtype(), &DataType::Utf8)?;
        match self.dtype() {
            DataType::Utf8 => Ok(self.utf8().unwrap().lt_eq(rhs)),
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical(_, _) => self.categorical().unwrap().lt_eq(rhs),
            _ => polars_bail!(
                ComputeError: "cannot compare str value to series of type {}", self.dtype(),
            ),
        }
    }
}
