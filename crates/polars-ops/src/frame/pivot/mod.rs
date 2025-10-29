mod positioning;
mod unpivot;

use std::borrow::Cow;

use polars_core::frame::group_by::expr::PhysicalAggExpr;
use polars_core::prelude::*;
use polars_core::utils::_split_offsets;
use polars_core::{POOL, downcast_as_macro_arg_physical};
use polars_utils::format_pl_smallstr;
use rayon::prelude::*;
pub use unpivot::UnpivotDF;

const HASHMAP_INIT_SIZE: usize = 512;

#[derive(Clone)]
pub struct PivotAgg(pub Arc<dyn PhysicalAggExpr + Send + Sync>);

fn restore_logical_type(s: &Series, logical_type: &DataType) -> Series {
    // restore logical type
    match (logical_type, s.dtype()) {
        (DataType::Float32, DataType::UInt32) => {
            let ca = s.u32().unwrap();
            ca._reinterpret_float().into_series()
        },
        (DataType::Float64, DataType::UInt64) => {
            let ca = s.u64().unwrap();
            ca._reinterpret_float().into_series()
        },
        (DataType::Int32, DataType::UInt32) => {
            let ca = s.u32().unwrap();
            ca.reinterpret_signed()
        },
        (DataType::Int64, DataType::UInt64) => {
            let ca = s.u64().unwrap();
            ca.reinterpret_signed()
        },
        #[cfg(feature = "dtype-duration")]
        (DataType::Duration(_), DataType::UInt64) => {
            let ca = s.u64().unwrap();
            ca.reinterpret_signed().cast(logical_type).unwrap()
        },
        #[cfg(feature = "dtype-datetime")]
        (DataType::Datetime(_, _), DataType::UInt64) => {
            let ca = s.u64().unwrap();
            ca.reinterpret_signed().cast(logical_type).unwrap()
        },
        #[cfg(feature = "dtype-date")]
        (DataType::Date, DataType::UInt32) => {
            let ca = s.u32().unwrap();
            ca.reinterpret_signed().cast(logical_type).unwrap()
        },
        #[cfg(feature = "dtype-time")]
        (DataType::Time, DataType::UInt64) => {
            let ca = s.u64().unwrap();
            ca.reinterpret_signed().cast(logical_type).unwrap()
        },
        (dt, DataType::Null) => {
            let ca = Series::full_null(s.name().clone(), s.len(), dt);
            ca.into_series()
        },
        _ => unsafe { s.from_physical_unchecked(logical_type).unwrap() },
    }
}
