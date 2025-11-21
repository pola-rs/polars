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
            Float32 => Float32,
            Float64 => Float64,
            _ => Int64,
        }
    }
}
