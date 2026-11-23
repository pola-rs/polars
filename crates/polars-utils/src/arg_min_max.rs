use crate::float16::pf16;

pub trait ArgMinMax {
    fn argmin(&self) -> usize;

    fn argmax(&self) -> usize;

    /// Skips the values whose bit in the `validity` bitmap (starting at bit `offset`) is
    /// not set. Returns `None` if no value is valid.
    fn argmin_masked(&self, validity: &[u8], offset: usize) -> Option<usize>;

    fn argmax_masked(&self, validity: &[u8], offset: usize) -> Option<usize>;
}

macro_rules! impl_argminmax {
    ($T:ty) => {
        impl ArgMinMax for $T {
            fn argmin(&self) -> usize {
                argminmax::ArgMinMax::argmin(self)
            }

            fn argmax(&self) -> usize {
                argminmax::ArgMinMax::argmax(self)
            }

            fn argmin_masked(&self, validity: &[u8], offset: usize) -> Option<usize> {
                argminmax::ArgMinMaxMasked::argmin_masked(self, validity, offset)
            }

            fn argmax_masked(&self, validity: &[u8], offset: usize) -> Option<usize> {
                argminmax::ArgMinMaxMasked::argmax_masked(self, validity, offset)
            }
        }
    };
}

impl_argminmax!(&[u8]);
impl_argminmax!(&[u16]);
impl_argminmax!(&[u32]);
impl_argminmax!(&[u64]);
impl_argminmax!(&[u128]);
impl_argminmax!(&[i8]);
impl_argminmax!(&[i16]);
impl_argminmax!(&[i32]);
impl_argminmax!(&[i64]);
impl_argminmax!(&[i128]);
impl_argminmax!(&[f32]);
impl_argminmax!(&[f64]);

impl ArgMinMax for &[pf16] {
    fn argmin(&self) -> usize {
        let transmuted: &&[half::f16] = unsafe { std::mem::transmute(self) };
        argminmax::ArgMinMax::argmin(transmuted)
    }

    fn argmax(&self) -> usize {
        let transmuted: &&[half::f16] = unsafe { std::mem::transmute(self) };
        argminmax::ArgMinMax::argmax(transmuted)
    }

    fn argmin_masked(&self, validity: &[u8], offset: usize) -> Option<usize> {
        let transmuted: &&[half::f16] = unsafe { std::mem::transmute(self) };
        argminmax::ArgMinMaxMasked::argmin_masked(transmuted, validity, offset)
    }

    fn argmax_masked(&self, validity: &[u8], offset: usize) -> Option<usize> {
        let transmuted: &&[half::f16] = unsafe { std::mem::transmute(self) };
        argminmax::ArgMinMaxMasked::argmax_masked(transmuted, validity, offset)
    }
}
