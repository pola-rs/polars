use crate::float16::pf16;

pub trait ArgMinMax {
    fn argmin(&self) -> usize;

    fn argmax(&self) -> usize;
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
        }
    };
}

impl_argminmax!(&[u8]);
impl_argminmax!(&[u16]);
impl_argminmax!(&[u32]);
impl_argminmax!(&[u64]);
impl_argminmax!(&[i8]);
impl_argminmax!(&[i16]);
impl_argminmax!(&[i32]);
impl_argminmax!(&[i64]);
impl_argminmax!(&[f32]);
impl_argminmax!(&[f64]);

impl ArgMinMax for &[i128] {
    fn argmin(&self) -> usize {
        let mut min_val = i128::MAX;
        let mut min_idx = 0;
        for (idx, val) in self.iter().enumerate() {
            if *val < min_val {
                min_val = *val;
                min_idx = idx;
            }
        }
        min_idx
    }

    fn argmax(&self) -> usize {
        let mut max_val = i128::MIN;
        let mut max_idx = 0;
        for (idx, val) in self.iter().enumerate() {
            if *val > max_val {
                max_val = *val;
                max_idx = idx;
            }
        }
        max_idx
    }
}

impl ArgMinMax for &[u128] {
    fn argmin(&self) -> usize {
        let mut min_val = u128::MAX;
        let mut min_idx = 0;
        for (idx, val) in self.iter().enumerate() {
            if *val < min_val {
                min_val = *val;
                min_idx = idx;
            }
        }
        min_idx
    }

    fn argmax(&self) -> usize {
        let mut max_val = u128::MIN;
        let mut max_idx = 0;
        for (idx, val) in self.iter().enumerate() {
            if *val > max_val {
                max_val = *val;
                max_idx = idx;
            }
        }
        max_idx
    }
}

impl ArgMinMax for &[pf16] {
    fn argmin(&self) -> usize {
        let transmuted: &&[half::f16] = unsafe { std::mem::transmute(self) };
        argminmax::ArgMinMax::argmin(transmuted)
    }

    fn argmax(&self) -> usize {
        let transmuted: &&[half::f16] = unsafe { std::mem::transmute(self) };
        argminmax::ArgMinMax::argmax(transmuted)
    }
}
