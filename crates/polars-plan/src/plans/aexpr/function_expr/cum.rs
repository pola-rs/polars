use super::*;

pub(super) fn cum_count(s: &Column, reverse: bool) -> PolarsResult<Column> {
    // @scalar-opt
    polars_ops::prelude::cum_count(s.as_materialized_series(), reverse).map(Column::from)
}

pub(super) fn cum_sum(s: &Column, reverse: bool) -> PolarsResult<Column> {
    // @scalar-opt
    polars_ops::prelude::cum_sum(s.as_materialized_series(), reverse).map(Column::from)
}

pub(super) fn cum_prod(s: &Column, reverse: bool) -> PolarsResult<Column> {
    // @scalar-opt
    polars_ops::prelude::cum_prod(s.as_materialized_series(), reverse).map(Column::from)
}

pub(super) fn cum_min(s: &Column, reverse: bool) -> PolarsResult<Column> {
    // @scalar-opt
    polars_ops::prelude::cum_min(s.as_materialized_series(), reverse).map(Column::from)
}

pub(super) fn cum_max(s: &Column, reverse: bool) -> PolarsResult<Column> {
    // @scalar-opt
    polars_ops::prelude::cum_max(s.as_materialized_series(), reverse).map(Column::from)
}

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
