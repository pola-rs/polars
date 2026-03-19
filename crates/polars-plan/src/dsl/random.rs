use super::*;

impl Expr {
    pub fn shuffle(self, seed: Option<u64>) -> Self {
        self.map_unary(FunctionExpr::Random {
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
        self.map_binary(
            FunctionExpr::Random {
                method: RandomMethod::Sample {
                    is_fraction: false,
                    with_replacement,
                    shuffle,
                },
                seed,
            },
            n,
        )
    }

    pub fn sample_frac(
        self,
        frac: Expr,
        with_replacement: bool,
        shuffle: bool,
        seed: Option<u64>,
    ) -> Self {
        self.map_binary(
            FunctionExpr::Random {
                method: RandomMethod::Sample {
                    is_fraction: true,
                    with_replacement,
                    shuffle,
                },
                seed,
            },
            frac,
        )
    }
}
