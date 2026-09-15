pub mod cov;
pub mod mean;
pub mod options;
pub mod sum;
use arrow::types::NativeType;
pub use cov::{EwmCovState, EwmStdState, EwmVarState, ewm_std, ewm_var};
pub use mean::{EwmMeanState, ewm_mean};
pub use options::EWMOptions;
use polars_array::{PlArray, PlPrimitiveArray};
pub use sum::{EwmSumState, ewm_sum};

/// A stateful exponentially weighted kernel, folded over the chunks of a column in order.
pub trait EwmStateUpdate {
    fn ewm_state_update(&mut self, values: &dyn PlArray) -> Box<dyn PlArray>;
}

/// The elements of `values`, as the primitive chunk the state was built to read.
fn chunk_of<T: NativeType>(values: &dyn PlArray) -> &PlPrimitiveArray<T> {
    values
        .as_any()
        .downcast_ref()
        .expect("EWM state reads a different primitive type than the chunk it was given")
}

#[cfg(test)]
macro_rules! assert_allclose {
    ($xs:expr, $ys:expr, $tol:expr) => {{
        // Bound once: the operands are call expressions that consume what they are given.
        let (xs, ys) = (&$xs, &$ys);
        assert_eq!(xs.len(), ys.len(), "compared chunks of different lengths");
        assert!(xs.iter().zip(ys.iter()).all(|(x, z)| {
            match (x, z) {
                (Some(a), Some(b)) => (a - b).abs() < $tol,
                (None, None) => true,
                _ => false,
            }
        }));
    }};
}

#[cfg(test)]
use assert_allclose;
