mod average;

pub use average::*;

#[derive(Debug, Copy, Clone)]
pub struct EWMOptions {
    pub alpha: f64,
    pub adjust: bool,
}

impl Default for EWMOptions {
    fn default() -> Self {
        Self {
            alpha: 0.5,
            adjust: true,
        }
    }
}

impl EWMOptions {
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
}
