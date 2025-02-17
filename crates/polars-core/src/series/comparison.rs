//! Comparison operations on Series.

use polars_error::feature_gated;

use crate::prelude::*;
use crate::series::arithmetic::coerce_lhs_rhs;
use crate::series::nulls::replace_non_null;

macro_rules! impl_eq_compare {
    ($self:expr, $rhs:expr, $method:ident) => {{
        use DataType::*;
        let (lhs, rhs) = ($self, $rhs);
        validate_types(lhs.dtype(), rhs.dtype())?;

        polars_ensure!(
            lhs.len() == rhs.len() ||

            // Broadcast
            lhs.len() == 1 ||
            rhs.len() == 1,
            ShapeMismatch: "could not compare between two series of different length ({} != {})",
            lhs.len(),
            rhs.len()
        );

        #[cfg(feature = "dtype-categorical")]
        match (lhs.dtype(), rhs.dtype()) {
            (Categorical(_, _) | Enum(_, _), Categorical(_, _) | Enum(_, _)) => {
                return Ok(lhs
                    .categorical()
                    .unwrap()
                    .$method(rhs.categorical().unwrap())?
                    .with_name(lhs.name().clone()));
            },
            (Categorical(_, _) | Enum(_, _), String) => {
                return Ok(lhs
                    .categorical()
                    .unwrap()
                    .$method(rhs.str().unwrap())?
                    .with_name(lhs.name().clone()));
            },
            (String, Categorical(_, _) | Enum(_, _)) => {
                return Ok(rhs
                    .categorical()
                    .unwrap()
                    .$method(lhs.str().unwrap())?
                    .with_name(lhs.name().clone()));
            },
            _ => (),
        };

        let (lhs, rhs) = coerce_lhs_rhs(lhs, rhs)
            .map_err(|_| polars_err!(
                    SchemaMismatch: "could not evaluate comparison between series '{}' of dtype: {} and series '{}' of dtype: {}",
                    lhs.name(), lhs.dtype(), rhs.name(), rhs.dtype()
            ))?;
        let lhs = lhs.to_physical_repr();
        let rhs = rhs.to_physical_repr();
        let mut out = match lhs.dtype() {
            Null => lhs.null().unwrap().$method(rhs.null().unwrap()),
            Boolean => lhs.bool().unwrap().$method(rhs.bool().unwrap()),
            String => lhs.str().unwrap().$method(rhs.str().unwrap()),
            Binary => lhs.binary().unwrap().$method(rhs.binary().unwrap()),
            UInt8 => feature_gated!("dtype-u8", lhs.u8().unwrap().$method(rhs.u8().unwrap())),
            UInt16 => feature_gated!("dtype-u16", lhs.u16().unwrap().$method(rhs.u16().unwrap())),
            UInt32 => lhs.u32().unwrap().$method(rhs.u32().unwrap()),
            UInt64 => lhs.u64().unwrap().$method(rhs.u64().unwrap()),
            Int8 => feature_gated!("dtype-i8", lhs.i8().unwrap().$method(rhs.i8().unwrap())),
            Int16 => feature_gated!("dtype-i16", lhs.i16().unwrap().$method(rhs.i16().unwrap())),
            Int32 => lhs.i32().unwrap().$method(rhs.i32().unwrap()),
            Int64 => lhs.i64().unwrap().$method(rhs.i64().unwrap()),
            Int128 => feature_gated!("dtype-i128", lhs.i128().unwrap().$method(rhs.i128().unwrap())),
            Float32 => lhs.f32().unwrap().$method(rhs.f32().unwrap()),
            Float64 => lhs.f64().unwrap().$method(rhs.f64().unwrap()),
            List(_) => lhs.list().unwrap().$method(rhs.list().unwrap()),
            #[cfg(feature = "dtype-array")]
            Array(_, _) => lhs.array().unwrap().$method(rhs.array().unwrap()),
            #[cfg(feature = "dtype-struct")]
            Struct(_) => lhs.struct_().unwrap().$method(rhs.struct_().unwrap()),

            dt => polars_bail!(InvalidOperation: "could not apply comparison on series of dtype '{}; operand names: '{}', '{}'", dt, lhs.name(), rhs.name()),
        };
        out.rename(lhs.name().clone());
        PolarsResult::Ok(out)
    }};
}

macro_rules! bail_invalid_ineq {
    ($lhs:expr, $rhs:expr, $op:literal) => {
        polars_bail!(
            InvalidOperation: "cannot perform '{}' comparison between series '{}' of dtype: {} and series '{}' of dtype: {}",
            $op,
            $lhs.name(), $lhs.dtype(),
            $rhs.name(), $rhs.dtype(),
        )
    };
}

macro_rules! impl_ineq_compare {
    ($self:expr, $rhs:expr, $method:ident, $op:literal) => {{
        use DataType::*;
        let (lhs, rhs) = ($self, $rhs);
        validate_types(lhs.dtype(), rhs.dtype())?;

        polars_ensure!(
            lhs.len() == rhs.len() ||

            // Broadcast
            lhs.len() == 1 ||
            rhs.len() == 1,
            ShapeMismatch:
                "could not perform '{}' comparison between series '{}' of length: {} and series '{}' of length: {}, because they have different lengths",
            $op,
            lhs.name(), lhs.len(),
            rhs.name(), rhs.len()
        );

        #[cfg(feature = "dtype-categorical")]
        match (lhs.dtype(), rhs.dtype()) {
            (Categorical(_, _) | Enum(_, _), Categorical(_, _) | Enum(_, _)) => {
                return Ok(lhs
                    .categorical()
                    .unwrap()
                    .$method(rhs.categorical().unwrap())?
                    .with_name(lhs.name().clone()));
            },
            (Categorical(_, _) | Enum(_, _), String) => {
                return Ok(lhs
                    .categorical()
                    .unwrap()
                    .$method(rhs.str().unwrap())?
                    .with_name(lhs.name().clone()));
            },
            (String, Categorical(_, _) | Enum(_, _)) => {
                return Ok(rhs
                    .categorical()
                    .unwrap()
                    .$method(lhs.str().unwrap())?
                    .with_name(lhs.name().clone()));
            },
            _ => (),
        };

        let (lhs, rhs) = coerce_lhs_rhs(lhs, rhs).map_err(|_|
            polars_err!(
                SchemaMismatch: "could not evaluate '{}' comparison between series '{}' of dtype: {} and series '{}' of dtype: {}",
                $op,
                lhs.name(), lhs.dtype(),
                rhs.name(), rhs.dtype()
            )
        )?;
        let lhs = lhs.to_physical_repr();
        let rhs = rhs.to_physical_repr();
        let mut out = match lhs.dtype() {
            Null => lhs.null().unwrap().$method(rhs.null().unwrap()),
            Boolean => lhs.bool().unwrap().$method(rhs.bool().unwrap()),
            String => lhs.str().unwrap().$method(rhs.str().unwrap()),
            Binary => lhs.binary().unwrap().$method(rhs.binary().unwrap()),
            UInt8 => feature_gated!("dtype-u8", lhs.u8().unwrap().$method(rhs.u8().unwrap())),
            UInt16 => feature_gated!("dtype-u16", lhs.u16().unwrap().$method(rhs.u16().unwrap())),
            UInt32 => lhs.u32().unwrap().$method(rhs.u32().unwrap()),
            UInt64 => lhs.u64().unwrap().$method(rhs.u64().unwrap()),
            Int8 => feature_gated!("dtype-i8", lhs.i8().unwrap().$method(rhs.i8().unwrap())),
            Int16 => feature_gated!("dtype-i16", lhs.i16().unwrap().$method(rhs.i16().unwrap())),
            Int32 => lhs.i32().unwrap().$method(rhs.i32().unwrap()),
            Int64 => lhs.i64().unwrap().$method(rhs.i64().unwrap()),
            Int128 => feature_gated!("dtype-i128", lhs.i128().unwrap().$method(rhs.i128().unwrap())),
            Float32 => lhs.f32().unwrap().$method(rhs.f32().unwrap()),
            Float64 => lhs.f64().unwrap().$method(rhs.f64().unwrap()),
            List(_) => bail_invalid_ineq!(lhs, rhs, $op),
            #[cfg(feature = "dtype-array")]
            Array(_, _) => bail_invalid_ineq!(lhs, rhs, $op),
            #[cfg(feature = "dtype-struct")]
            Struct(_) => bail_invalid_ineq!(lhs, rhs, $op),

            dt => polars_bail!(InvalidOperation: "could not apply comparison on series of dtype '{}; operand names: '{}', '{}'", dt, lhs.name(), rhs.name()),
        };
        out.rename(lhs.name().clone());
        PolarsResult::Ok(out)
    }};
}

fn validate_types(left: &DataType, right: &DataType) -> PolarsResult<()> {
    use DataType::*;

    match (left, right) {
        (String, dt) | (dt, String) if dt.is_primitive_numeric() => {
            polars_bail!(ComputeError: "cannot compare string with numeric type ({})", dt)
        },
        #[cfg(feature = "dtype-categorical")]
        (Categorical(_, _) | Enum(_, _), dt) | (dt, Categorical(_, _) | Enum(_, _))
            if !(dt.is_categorical() | dt.is_string() | dt.is_enum()) =>
        {
            polars_bail!(ComputeError: "cannot compare categorical with {}", dt);
        },
        _ => (),
    };
    Ok(())
}

impl ChunkCompareEq<&Series> for Series {
    type Item = PolarsResult<BooleanChunked>;

    /// Create a boolean mask by checking for equality.
    fn equal(&self, rhs: &Series) -> Self::Item {
        impl_eq_compare!(self, rhs, equal)
    }

    /// Create a boolean mask by checking for equality.
    fn equal_missing(&self, rhs: &Series) -> Self::Item {
        impl_eq_compare!(self, rhs, equal_missing)
    }

    /// Create a boolean mask by checking for inequality.
    fn not_equal(&self, rhs: &Series) -> Self::Item {
        impl_eq_compare!(self, rhs, not_equal)
    }

    /// Create a boolean mask by checking for inequality.
    fn not_equal_missing(&self, rhs: &Series) -> Self::Item {
        impl_eq_compare!(self, rhs, not_equal_missing)
    }
}

impl ChunkCompareIneq<&Series> for Series {
    type Item = PolarsResult<BooleanChunked>;

    /// Create a boolean mask by checking if self > rhs.
    fn gt(&self, rhs: &Series) -> Self::Item {
        impl_ineq_compare!(self, rhs, gt, ">")
    }

    /// Create a boolean mask by checking if self >= rhs.
    fn gt_eq(&self, rhs: &Series) -> Self::Item {
        impl_ineq_compare!(self, rhs, gt_eq, ">=")
    }

    /// Create a boolean mask by checking if self < rhs.
    fn lt(&self, rhs: &Series) -> Self::Item {
        impl_ineq_compare!(self, rhs, lt, "<")
    }

    /// Create a boolean mask by checking if self <= rhs.
    fn lt_eq(&self, rhs: &Series) -> Self::Item {
        impl_ineq_compare!(self, rhs, lt_eq, "<=")
    }
}

impl<Rhs> ChunkCompareEq<Rhs> for Series
where
    Rhs: NumericNative,
{
    type Item = PolarsResult<BooleanChunked>;

    fn equal(&self, rhs: Rhs) -> Self::Item {
        validate_types(self.dtype(), &DataType::Int8)?;
        let s = self.to_physical_repr();
        Ok(apply_method_physical_numeric!(&s, equal, rhs))
    }

    fn equal_missing(&self, rhs: Rhs) -> Self::Item {
        validate_types(self.dtype(), &DataType::Int8)?;
        let s = self.to_physical_repr();
        Ok(apply_method_physical_numeric!(&s, equal_missing, rhs))
    }

    fn not_equal(&self, rhs: Rhs) -> Self::Item {
        validate_types(self.dtype(), &DataType::Int8)?;
        let s = self.to_physical_repr();
        Ok(apply_method_physical_numeric!(&s, not_equal, rhs))
    }

    fn not_equal_missing(&self, rhs: Rhs) -> Self::Item {
        validate_types(self.dtype(), &DataType::Int8)?;
        let s = self.to_physical_repr();
        Ok(apply_method_physical_numeric!(&s, not_equal_missing, rhs))
    }
}

impl<Rhs> ChunkCompareIneq<Rhs> for Series
where
    Rhs: NumericNative,
{
    type Item = PolarsResult<BooleanChunked>;

    fn gt(&self, rhs: Rhs) -> Self::Item {
        validate_types(self.dtype(), &DataType::Int8)?;
        let s = self.to_physical_repr();
        Ok(apply_method_physical_numeric!(&s, gt, rhs))
    }

    fn gt_eq(&self, rhs: Rhs) -> Self::Item {
        validate_types(self.dtype(), &DataType::Int8)?;
        let s = self.to_physical_repr();
        Ok(apply_method_physical_numeric!(&s, gt_eq, rhs))
    }

    fn lt(&self, rhs: Rhs) -> Self::Item {
        validate_types(self.dtype(), &DataType::Int8)?;
        let s = self.to_physical_repr();
        Ok(apply_method_physical_numeric!(&s, lt, rhs))
    }

    fn lt_eq(&self, rhs: Rhs) -> Self::Item {
        validate_types(self.dtype(), &DataType::Int8)?;
        let s = self.to_physical_repr();
        Ok(apply_method_physical_numeric!(&s, lt_eq, rhs))
    }
}

impl ChunkCompareEq<&str> for Series {
    type Item = PolarsResult<BooleanChunked>;

    fn equal(&self, rhs: &str) -> PolarsResult<BooleanChunked> {
        validate_types(self.dtype(), &DataType::String)?;
        match self.dtype() {
            DataType::String => Ok(self.str().unwrap().equal(rhs)),
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical(_, _) | DataType::Enum(_, _) => {
                self.categorical().unwrap().equal(rhs)
            },
            _ => Ok(BooleanChunked::full(self.name().clone(), false, self.len())),
        }
    }

    fn equal_missing(&self, rhs: &str) -> Self::Item {
        validate_types(self.dtype(), &DataType::String)?;
        match self.dtype() {
            DataType::String => Ok(self.str().unwrap().equal_missing(rhs)),
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical(_, _) | DataType::Enum(_, _) => {
                self.categorical().unwrap().equal_missing(rhs)
            },
            _ => Ok(replace_non_null(
                self.name().clone(),
                self.0.chunks(),
                false,
            )),
        }
    }

    fn not_equal(&self, rhs: &str) -> PolarsResult<BooleanChunked> {
        validate_types(self.dtype(), &DataType::String)?;
        match self.dtype() {
            DataType::String => Ok(self.str().unwrap().not_equal(rhs)),
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical(_, _) | DataType::Enum(_, _) => {
                self.categorical().unwrap().not_equal(rhs)
            },
            _ => Ok(BooleanChunked::full(self.name().clone(), true, self.len())),
        }
    }

    fn not_equal_missing(&self, rhs: &str) -> Self::Item {
        validate_types(self.dtype(), &DataType::String)?;
        match self.dtype() {
            DataType::String => Ok(self.str().unwrap().not_equal_missing(rhs)),
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical(_, _) | DataType::Enum(_, _) => {
                self.categorical().unwrap().not_equal_missing(rhs)
            },
            _ => Ok(replace_non_null(self.name().clone(), self.0.chunks(), true)),
        }
    }
}

impl ChunkCompareIneq<&str> for Series {
    type Item = PolarsResult<BooleanChunked>;

    fn gt(&self, rhs: &str) -> Self::Item {
        validate_types(self.dtype(), &DataType::String)?;
        match self.dtype() {
            DataType::String => Ok(self.str().unwrap().gt(rhs)),
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical(_, _) | DataType::Enum(_, _) => {
                self.categorical().unwrap().gt(rhs)
            },
            _ => polars_bail!(
                ComputeError: "cannot compare str value to series of type {}", self.dtype(),
            ),
        }
    }

    fn gt_eq(&self, rhs: &str) -> Self::Item {
        validate_types(self.dtype(), &DataType::String)?;
        match self.dtype() {
            DataType::String => Ok(self.str().unwrap().gt_eq(rhs)),
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical(_, _) | DataType::Enum(_, _) => {
                self.categorical().unwrap().gt_eq(rhs)
            },
            _ => polars_bail!(
                ComputeError: "cannot compare str value to series of type {}", self.dtype(),
            ),
        }
    }

    fn lt(&self, rhs: &str) -> Self::Item {
        validate_types(self.dtype(), &DataType::String)?;
        match self.dtype() {
            DataType::String => Ok(self.str().unwrap().lt(rhs)),
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical(_, _) | DataType::Enum(_, _) => {
                self.categorical().unwrap().lt(rhs)
            },
            _ => polars_bail!(
                ComputeError: "cannot compare str value to series of type {}", self.dtype(),
            ),
        }
    }

    fn lt_eq(&self, rhs: &str) -> Self::Item {
        validate_types(self.dtype(), &DataType::String)?;
        match self.dtype() {
            DataType::String => Ok(self.str().unwrap().lt_eq(rhs)),
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical(_, _) | DataType::Enum(_, _) => {
                self.categorical().unwrap().lt_eq(rhs)
            },
            _ => polars_bail!(
                ComputeError: "cannot compare str value to series of type {}", self.dtype(),
            ),
        }
    }
}
