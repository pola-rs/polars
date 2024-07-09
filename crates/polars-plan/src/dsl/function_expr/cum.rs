use super::*;

pub(super) fn cum_count(s: &Series, reverse: bool) -> PolarsResult<Series> {
    polars_ops::prelude::cum_count(s, reverse)
}

pub(super) fn cum_sum(s: &Series, reverse: bool) -> PolarsResult<Series> {
    polars_ops::prelude::cum_sum(s, reverse)
}

pub(super) fn cum_prod(s: &Series, reverse: bool) -> PolarsResult<Series> {
    polars_ops::prelude::cum_prod(s, reverse)
}

pub(super) fn cum_min(s: &Series, reverse: bool) -> PolarsResult<Series> {
    polars_ops::prelude::cum_min(s, reverse)
}

pub(super) fn cum_max(s: &Series, reverse: bool) -> PolarsResult<Series> {
    polars_ops::prelude::cum_max(s, reverse)
}

pub(super) mod dtypes {
    use polars_core::utils::materialize_dyn_int;
    use DataType::*;

    use super::*;

    pub fn cum_sum(dt: &DataType) -> DataType {
        if dt.is_logical() {
            dt.clone()
        } else {
            match dt {
                Boolean => UInt32,
                Int32 => Int32,
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
            Float32 => Float32,
            Float64 => Float64,
            _ => Int64,
        }
    }
}
