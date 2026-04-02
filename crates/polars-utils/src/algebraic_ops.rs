#[inline(always)]
pub fn alg_add_f64(a: f64, b: f64) -> f64 {
    #[cfg(feature = "nightly")]
    {
        std::intrinsics::fadd_algebraic(a, b)
    }
    #[cfg(not(feature = "nightly"))]
    {
        a + b
    }
}

#[inline(always)]
pub fn alg_mul_f64(a: f64, b: f64) -> f64 {
    #[cfg(feature = "nightly")]
    {
        std::intrinsics::fmul_algebraic(a, b)
    }
    #[cfg(not(feature = "nightly"))]
    {
        a * b
    }
}

pub fn alg_sum_f64(it: impl IntoIterator<Item = f64>) -> f64 {
    it.into_iter().fold(0.0, alg_add_f64)
}
