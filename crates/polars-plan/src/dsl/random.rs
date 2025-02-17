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
                method: RandomMethod::Sample {
                    is_fraction: false,
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
        frac: Expr,
        with_replacement: bool,
        shuffle: bool,
        seed: Option<u64>,
    ) -> Self {
        self.apply_many_private(
            FunctionExpr::Random {
                method: RandomMethod::Sample {
                    is_fraction: true,
                    with_replacement,
                    shuffle,
                },
                seed,
            },
            &[frac],
            false,
            false,
        )
    }
}
