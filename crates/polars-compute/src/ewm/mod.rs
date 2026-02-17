pub mod cov;
pub mod mean;
pub mod options;
use arrow::array::Array;
pub use cov::{EwmCovState, EwmStdState, EwmVarState, ewm_std, ewm_var};
pub use mean::{EwmMeanState, ewm_mean};
pub use options::EWMOptions;

pub trait EwmStateUpdate {
    fn ewm_state_update(&mut self, values: &dyn Array) -> Box<dyn Array>;
}

#[cfg(test)]
macro_rules! assert_allclose {
    ($xs:expr, $ys:expr, $tol:expr) => {
        assert!($xs.iter().zip($ys.iter()).all(|(x, z)| {
            match (x, z) {
                (Some(a), Some(b)) => (a - b).abs() < $tol,
                (None, None) => true,
                _ => false,
            }
        }));
    };
}

#[cfg(test)]
use assert_allclose;
