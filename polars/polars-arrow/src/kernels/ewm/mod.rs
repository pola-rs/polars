mod average;
mod variance;

pub use average::*;
pub use variance::*;

#[derive(Debug, Copy, Clone)]
#[must_use]
pub struct EWMOptions {
    pub alpha: f64,
    pub adjust: bool,
    pub bias: bool,
    pub min_periods: usize,
    pub ignore_nulls: bool,
}

impl Default for EWMOptions {
    fn default() -> Self {
        Self {
            alpha: 0.5,
            adjust: true,
            bias: false,
            min_periods: 1,
            ignore_nulls: true,
        }
    }
}

impl EWMOptions {
    pub fn and_min_periods(mut self, min_periods: usize) -> Self {
        self.min_periods = min_periods;
        self
    }
    pub fn and_adjust(mut self, adjust: bool) -> Self {
        self.adjust = adjust;
        self
    }
    pub fn and_span(mut self, span: usize) -> Self {
        assert!(span >= 1);
        self.alpha = 2.0 / (span as f64 + 1.0);
        self
    }
    pub fn and_half_life(mut self, half_life: f64) -> Self {
        assert!(half_life > 0.0);
        self.alpha = 1.0 - ((-2.0f64).ln() / half_life).exp();
        self
    }
    pub fn and_com(mut self, com: f64) -> Self {
        assert!(com > 0.0);
        self.alpha = 1.0 / (1.0 + com);
        self
    }
    pub fn and_ignore_nulls(mut self, ignore_nulls: bool) -> Self {
        self.ignore_nulls = ignore_nulls;
        self
    }
}

#[cfg(test)]
macro_rules! assert_allclose {
    ($xs:expr, $ys:expr, $tol:expr) => {
        assert!($xs
            .iter()
            .zip($ys.iter())
            .map(|(x, z)| {
                match (x, z) {
                    (Some(a), Some(b)) => (a - b).abs() < $tol,
                    (None, None) => true,
                    _ => false,
                }
            })
            .fold(true, |acc, b| acc && b));
    };
}
#[cfg(test)]
pub(crate) use assert_allclose;
