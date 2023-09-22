use super::*;

impl Expr {
    pub fn shuffle(self, seed: Option<u64>) -> Self {
        self.apply_private(FunctionExpr::Random {
            method: RandomMethod::Shuffle,
            seed,
        })
    }

    pub fn sample_n(
        self,
        n: Expr,
        with_replacement: bool,
        shuffle: bool,
        seed: Option<u64>,
    ) -> Self {
        self.apply_many_private(
            FunctionExpr::Random {
                method: RandomMethod::SampleN {
                    with_replacement,
                    shuffle,
                },
                seed,
            },
            &[n],
            false,
            false,
        )
    }

    pub fn sample_frac(
        self,
        frac: f64,
        with_replacement: bool,
        shuffle: bool,
        seed: Option<u64>,
    ) -> Self {
        self.apply_private(FunctionExpr::Random {
            method: RandomMethod::SampleFrac {
                frac,
                with_replacement,
                shuffle,
            },
            seed,
        })
    }
}
