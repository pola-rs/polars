use super::RollingFnParams;
use crate::moment::{KurtosisState, SkewState, VarState};

pub trait StateUpdate {
    fn new(params: Option<RollingFnParams>) -> Self;
    fn insert_one(&mut self, x: f64);

    fn remove_one(&mut self, x: f64);

    fn finalize(&self) -> Option<f64>;
}

pub struct VarianceMoment {
    state: VarState,
    ddof: u8,
}

impl StateUpdate for VarianceMoment {
    fn new(params: Option<RollingFnParams>) -> Self {
        let ddof = if let Some(RollingFnParams::Var(params)) = params {
            params.ddof
        } else {
            1
        };

        Self {
            state: VarState::default(),
            ddof,
        }
    }

    fn insert_one(&mut self, x: f64) {
        self.state.insert_one(x);
    }

    fn remove_one(&mut self, x: f64) {
        self.state.remove_one(x);
    }
    fn finalize(&self) -> Option<f64> {
        self.state.finalize(self.ddof)
    }
}

pub struct KurtosisMoment {
    state: KurtosisState,
    fisher: bool,
    bias: bool,
}

impl StateUpdate for KurtosisMoment {
    fn new(params: Option<RollingFnParams>) -> Self {
        let (fisher, bias) = if let Some(RollingFnParams::Kurtosis { fisher, bias }) = params {
            (fisher, bias)
        } else {
            (false, false)
        };

        Self {
            state: KurtosisState::default(),
            fisher,
            bias,
        }
    }

    fn insert_one(&mut self, x: f64) {
        self.state.insert_one(x);
    }

    fn remove_one(&mut self, x: f64) {
        self.state.remove_one(x);
    }
    fn finalize(&self) -> Option<f64> {
        self.state.finalize(self.fisher, self.bias)
    }
}

pub struct SkewMoment {
    state: SkewState,
    bias: bool,
}

impl StateUpdate for SkewMoment {
    fn new(params: Option<RollingFnParams>) -> Self {
        let bias = if let Some(RollingFnParams::Skew { bias }) = params {
            bias
        } else {
            false
        };

        Self {
            state: SkewState::default(),
            bias,
        }
    }

    fn insert_one(&mut self, x: f64) {
        self.state.insert_one(x);
    }

    fn remove_one(&mut self, x: f64) {
        self.state.remove_one(x);
    }
    fn finalize(&self) -> Option<f64> {
        self.state.finalize(self.bias)
    }
}
