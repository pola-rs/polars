pub mod mean;

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
