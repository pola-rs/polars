#[inline(always)]
pub fn alg_add_f64(a: f64, b: f64) -> f64 {
    #[cfg(feature = "nightly")]
    {
        a.algebraic_add(b)
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
        a.algebraic_mul(b)
    }
    #[cfg(not(feature = "nightly"))]
    {
        a * b
    }
}

pub fn alg_sum_f64(it: impl IntoIterator<Item = f64>) -> f64 {
    // Negative zero is identity element of floating point addition, not
    // positive zero (since -0.0 + 0.0 = 0.0).
    it.into_iter().fold(-0.0, alg_add_f64)
}
