use polars_core::random::get_global_random_u64;

use super::*;

impl Expr {
    pub fn shuffle(self, seed: Option<u64>) -> Self {
        self.apply_private(FunctionExpr::Random {
            method: RandomMethod::Shuffle,
            seed: seed.unwrap_or_else(get_global_random_u64),
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
                seed: seed.unwrap_or_else(get_global_random_u64),
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
                seed: seed.unwrap_or_else(get_global_random_u64),
            },
            &[frac],
            false,
            false,
        )
    }
}
