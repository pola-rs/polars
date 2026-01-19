use super::*;

pub(super) mod dtypes {
    use DataType::*;
    use polars_core::utils::materialize_dyn_int;

    use super::*;

    pub fn cum_sum(dt: &DataType) -> DataType {
        if dt.is_logical() {
            dt.clone()
        } else {
            match dt {
                Boolean => UInt32,
                Int32 => Int32,
                Int128 => Int128,
                UInt32 => UInt32,
                UInt64 => UInt64,
                #[cfg(feature = "dtype-f16")]
                Float16 => Float16,
                Float32 => Float32,
                Float64 => Float64,
                Unknown(kind) => match kind {
                    UnknownKind::Int(v) => cum_sum(&materialize_dyn_int(*v).dtype()),
                    UnknownKind::Float => Float64,
                    _ => dt.clone(),
                },
                _ => Int64,
            }
        }
    }

    pub fn cum_prod(dt: &DataType) -> DataType {
        match dt {
            Boolean => Int64,
            UInt64 => UInt64,
            Int128 => Int128,
            #[cfg(feature = "dtype-f16")]
            Float16 => Float16,
            Float32 => Float32,
            Float64 => Float64,
            _ => Int64,
        }
    }

    pub fn cum_mean(dt: &DataType) -> DataType {
        match dt {
            #[cfg(feature = "dtype-duration")]
            Duration(_) => dt.clone(),
            #[cfg(feature = "dtype-datetime")]
            Datetime(tu, tz) => Datetime(*tu, tz.clone()),
            #[cfg(feature = "dtype-date")]
            Date => Datetime(polars_core::prelude::TimeUnit::Microseconds, None),
            #[cfg(feature = "dtype-decimal")]
            Decimal(precision, scale) => Decimal(*precision, *scale),
            #[cfg(feature = "dtype-f16")]
            Float16 => Float16,
            Float32 => Float32,
            _ => Float64,
        }
    }
}
