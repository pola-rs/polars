use arrow::bitmap::Bitmap;

pub trait TotalOrdKernel : Sized {
    fn tot_lt_kernel(&self, other: &Self) -> Bitmap;
    fn tot_le_kernel(&self, other: &Self) -> Bitmap;

    fn tot_gt_kernel(&self, other: &Self) -> Bitmap {
        other.tot_lt_kernel(self)
    }

    fn tot_ge_kernel(&self, other: &Self) -> Bitmap {
        other.tot_le_kernel(self)
    }
    
    fn tot_eq_kernel(&self, other: &Self) -> Bitmap;
    fn tot_ne_kernel(&self, other: &Self) -> Bitmap;
}

trait NotSimd { }

macro_rules! impl_not_simd {
    ($($T:ty,)*) => {
        $(impl NotSimd for $T { })*
    };
}

#[cfg(not(feature="simd"))]
impl<T> NotSimd for T {}

#[cfg(feature="simd")]
impl_not_simd!(u128, i128,);

mod scalar;

#[cfg(feature="simd")]
mod simd;
