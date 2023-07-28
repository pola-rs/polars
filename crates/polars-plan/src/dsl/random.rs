use std::sync::atomic::AtomicU64;

use super::*;

fn get_atomic_seed(seed: Option<u64>) -> Option<SpecialEq<Arc<AtomicU64>>> {
    seed.map(|v| SpecialEq::new(Arc::new(AtomicU64::new(v))))
}

impl Expr {
    pub fn shuffle(self, seed: Option<u64>, fixed_seed: bool) -> Self {
        self.apply_private(FunctionExpr::Random {
            method: RandomMethod::Shuffle,
            atomic_seed: get_atomic_seed(seed),
            seed,
            fixed_seed,
        })
    }

    pub fn sample_n(
        self,
        n: usize,
        with_replacement: bool,
        shuffle: bool,
        seed: Option<u64>,
        fixed_seed: bool,
    ) -> Self {
        self.apply_private(FunctionExpr::Random {
            method: RandomMethod::SampleN {
                n,
                with_replacement,
                shuffle,
            },
            atomic_seed: get_atomic_seed(seed),
            seed,
            fixed_seed,
        })
    }

    pub fn sample_frac(
        self,
        frac: f64,
        with_replacement: bool,
        shuffle: bool,
        seed: Option<u64>,
        fixed_seed: bool,
    ) -> Self {
        self.apply_private(FunctionExpr::Random {
            method: RandomMethod::SampleFrac {
                frac,
                with_replacement,
                shuffle,
            },
            atomic_seed: get_atomic_seed(seed),
            seed,
            fixed_seed,
        })
    }
}
