//! Comparison operations on Series.

use crate::prelude::*;
use crate::series::arithmetic::coerce_lhs_rhs;
use crate::series::nulls::replace_non_null;

macro_rules! impl_compare {
    ($self:expr, $rhs:expr, $method:ident, $struct_function:expr) => {{
        use DataType::*;
        let (lhs, rhs) = ($self, $rhs);
        validate_types(lhs.dtype(), rhs.dtype())?;

        #[cfg(feature = "dtype-categorical")]
        match (lhs.dtype(), rhs.dtype()) {
            (Categorical(_, _) | Enum(_, _), Categorical(_, _) | Enum(_, _)) => {
                return Ok(lhs
                    .categorical()
                    .unwrap()
                    .$method(rhs.categorical().unwrap())?
                    .with_name(lhs.name()));
            },
            (Categorical(_, _) | Enum(_, _), String) => {
                return Ok(lhs
                    .categorical()
                    .unwrap()
                    .$method(rhs.str().unwrap())?
                    .with_name(lhs.name()));
            },
            (String, Categorical(_, _) | Enum(_, _)) => {
                return Ok(rhs
                    .categorical()
                    .unwrap()
                    .$method(lhs.str().unwrap())?
                    .with_name(lhs.name()));
            },
            _ => (),
        };

        let (lhs, rhs) = coerce_lhs_rhs(lhs, rhs).map_err(|_| polars_err!(SchemaMismatch: "could not evaluate comparison between series '{}' of dtype: {} and series '{}' of dtype: {}",
        lhs.name(), lhs.dtype(), rhs.name(), rhs.dtype()))?;
        let lhs = lhs.to_physical_repr();
        let rhs = rhs.to_physical_repr();
        let mut out = match lhs.dtype() {
            Null => lhs.null().unwrap().$method(rhs.null().unwrap()),
            Boolean => lhs.bool().unwrap().$method(rhs.bool().unwrap()),
            String => lhs.str().unwrap().$method(rhs.str().unwrap()),
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
            Struct(_) => {
                let lhs = lhs
                .struct_()
                .unwrap();
                let rhs = rhs.struct_().unwrap();

                $struct_function(lhs, rhs)?
            },
            #[cfg(feature = "dtype-decimal")]
            Decimal(_, s1) => {
                let DataType::Decimal(_, s2) = rhs.dtype() else {
                    unreachable!()
                };
                let scale = s1.max(s2).unwrap();
                let lhs = lhs.decimal().unwrap().to_scale(scale).unwrap();
                let rhs = rhs.decimal().unwrap().to_scale(scale).unwrap();
                lhs.0.$method(&rhs.0)
            },

            dt => polars_bail!(InvalidOperation: "could not apply comparison on series of dtype '{}; operand names: '{}', '{}'", dt, lhs.name(), rhs.name()),
        };
        out.rename(lhs.name());
        PolarsResult::Ok(out)
    }};
}

#[cfg(feature = "dtype-struct")]
fn raise_struct(_a: &StructChunked, _b: &StructChunked) -> PolarsResult<BooleanChunked> {
    polars_bail!(InvalidOperation: "order comparison not support for struct dtype")
}

#[cfg(not(feature = "dtype-struct"))]
fn raise_struct(_a: &(), _b: &()) -> PolarsResult<BooleanChunked> {
    unimplemented!()
}

fn validate_types(left: &DataType, right: &DataType) -> PolarsResult<()> {
    use DataType::*;

    match (left, right) {
        (String, dt) | (dt, String) if dt.is_numeric() => {
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

impl ChunkCompare<&Series> for Series {
    type Item = PolarsResult<BooleanChunked>;

    /// Create a boolean mask by checking for equality.
    fn equal(&self, rhs: &Series) -> PolarsResult<BooleanChunked> {
        impl_compare!(self, rhs, equal, |a: &StructChunked, b: &StructChunked| {
            PolarsResult::Ok(a.equal(b))
        })
    }

    /// Create a boolean mask by checking for equality.
    fn equal_missing(&self, rhs: &Series) -> PolarsResult<BooleanChunked> {
        impl_compare!(
            self,
            rhs,
            equal_missing,
            |a: &StructChunked, b: &StructChunked| PolarsResult::Ok(a.equal_missing(b))
        )
    }

    /// Create a boolean mask by checking for inequality.
    fn not_equal(&self, rhs: &Series) -> PolarsResult<BooleanChunked> {
        impl_compare!(
            self,
            rhs,
            not_equal,
            |a: &StructChunked, b: &StructChunked| PolarsResult::Ok(a.not_equal(b))
        )
    }

    /// Create a boolean mask by checking for inequality.
    fn not_equal_missing(&self, rhs: &Series) -> PolarsResult<BooleanChunked> {
        impl_compare!(
            self,
            rhs,
            not_equal_missing,
            |a: &StructChunked, b: &StructChunked| PolarsResult::Ok(a.not_equal_missing(b))
        )
    }

    /// Create a boolean mask by checking if self > rhs.
    fn gt(&self, rhs: &Series) -> PolarsResult<BooleanChunked> {
        impl_compare!(self, rhs, gt, raise_struct)
    }

    /// Create a boolean mask by checking if self >= rhs.
    fn gt_eq(&self, rhs: &Series) -> PolarsResult<BooleanChunked> {
        impl_compare!(self, rhs, gt_eq, raise_struct)
    }

    /// Create a boolean mask by checking if self < rhs.
    fn lt(&self, rhs: &Series) -> PolarsResult<BooleanChunked> {
        impl_compare!(self, rhs, lt, raise_struct)
    }

    /// Create a boolean mask by checking if self <= rhs.
    fn lt_eq(&self, rhs: &Series) -> PolarsResult<BooleanChunked> {
        impl_compare!(self, rhs, lt_eq, raise_struct)
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
        validate_types(self.dtype(), &DataType::String)?;
        match self.dtype() {
            DataType::String => Ok(self.str().unwrap().equal(rhs)),
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical(_, _) | DataType::Enum(_, _) => {
                self.categorical().unwrap().equal(rhs)
            },
            _ => Ok(BooleanChunked::full(self.name(), false, self.len())),
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
            _ => Ok(replace_non_null(self.name(), self.0.chunks(), false)),
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
            _ => Ok(BooleanChunked::full(self.name(), true, self.len())),
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
            _ => Ok(replace_non_null(self.name(), self.0.chunks(), true)),
        }
    }

    fn gt(&self, rhs: &str) -> PolarsResult<BooleanChunked> {
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
