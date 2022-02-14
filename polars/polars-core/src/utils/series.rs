use crate::prelude::*;

/// Transform to physical type and coerce floating point and similar sized integer to a bit representation
/// to reduce compiler bloat
pub(crate) fn to_physical_and_bit_repr(s: &[Series]) -> Vec<Series> {
    s.iter()
        .map(|s| {
            let physical = s.to_physical_repr();
            match physical.dtype() {
                DataType::Int64 => physical.bit_repr_large().into_series(),
                DataType::Int32 => physical.bit_repr_small().into_series(),
                DataType::Float32 => physical.bit_repr_small().into_series(),
                DataType::Float64 => physical.bit_repr_large().into_series(),
                _ => physical.into_owned(),
            }
        })
        .collect()
}
