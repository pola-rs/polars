#[cfg(feature = "dtype-array")]
pub mod array;
mod binary;
pub mod list;
#[cfg(feature = "dtype-map")]
pub mod map;
#[cfg(feature = "propagate_nans")]
pub mod nan_propagating_aggregate;
#[cfg(feature = "peaks")]
pub mod peaks;
mod scatter;
pub mod strings;
mod sum;
#[cfg(feature = "top_k")]
mod top_k;

#[cfg(feature = "mode")]
pub mod mode;

#[cfg(feature = "cov")]
pub mod cov;
pub(crate) mod gather;
#[cfg(feature = "gather")]
pub mod gather_skip_nulls;
#[cfg(feature = "hist")]
mod hist;
#[cfg(feature = "repeat_by")]
mod repeat_by;

pub use binary::*;
#[cfg(feature = "chunked_ids")]
pub use gather::*;
#[cfg(feature = "hist")]
pub use hist::*;
pub use list::*;
#[cfg(feature = "dtype-map")]
pub use map::*;
#[allow(unused_imports)]
use polars_core::prelude::*;
#[cfg(feature = "repeat_by")]
pub use repeat_by::*;
pub use scatter::ChunkedSet;
pub use strings::*;
#[cfg(feature = "top_k")]
pub use top_k::*;

#[allow(unused_imports)]
use crate::prelude::*;

/// A count as `usize`; a count beyond `usize::MAX` covers any string or binary value the same way.
pub(crate) fn saturating_usize(n: u64) -> usize {
    usize::try_from(n).unwrap_or(usize::MAX)
}

/// `n` as Int64 counts for `len` values. A UInt64 count beyond the Int64 range is clipped, as
/// it covers any value. A repeated count is returned once, so that it is applied as a scalar.
pub(crate) fn count_as_i64(n: &Column, len: usize) -> PolarsResult<Column> {
    let n = match n.as_scalar_column() {
        Some(sc) if n.len() == len && len > 1 => &sc.as_single_value_series().into_column(),
        _ => n,
    };
    match n.dtype() {
        DataType::UInt64 => n.try_apply_unary_elementwise(|n| {
            Ok(
                arity::unary_elementwise_values::<_, Int64Type, _>(n.u64()?, |v| {
                    v.min(i64::MAX as u64) as i64
                })
                .into_series(),
            )
        }),
        _ => n.strict_cast(&DataType::Int64),
    }
}
