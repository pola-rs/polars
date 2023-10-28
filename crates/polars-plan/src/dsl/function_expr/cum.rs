use super::*;

pub(super) fn cumcount(s: &Series, reverse: bool) -> PolarsResult<Series> {
    polars_ops::prelude::cumcount(s, reverse)
}

pub(super) fn cumsum(s: &Series, reverse: bool) -> PolarsResult<Series> {
    polars_ops::prelude::cumsum(s, reverse)
}

pub(super) fn cumprod(s: &Series, reverse: bool) -> PolarsResult<Series> {
    polars_ops::prelude::cumprod(s, reverse)
}

pub(super) fn cummin(s: &Series, reverse: bool) -> PolarsResult<Series> {
    polars_ops::prelude::cummin(s, reverse)
}

pub(super) fn cummax(s: &Series, reverse: bool) -> PolarsResult<Series> {
    polars_ops::prelude::cummax(s, reverse)
}

pub(super) mod dtypes {
    use DataType::*;

    use super::*;

    pub fn cumsum(dt: &DataType) -> DataType {
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
                _ => Int64,
            }
        }
    }

    pub fn cumprod(dt: &DataType) -> DataType {
        match dt {
            Boolean => Int64,
            UInt64 => UInt64,
            Float32 => Float32,
            Float64 => Float64,
            _ => Int64,
        }
    }
}
