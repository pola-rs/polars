use std::cmp::Ordering;

use crate::data_types::IsFloat;
use crate::kernels::rolling::compare_fn_nan_max;

/// A utility type that make floats Ord by
/// nan == nan == true
/// nan > float::max == true
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct OrdFloat<T>(T);

impl<T: IsFloat + PartialEq + PartialOrd> PartialOrd for OrdFloat<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(compare_fn_nan_max(&self.0, &other.0))
    }
}

impl<T: IsFloat + PartialEq + PartialOrd> Ord for OrdFloat<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        compare_fn_nan_max(&self.0, &other.0)
    }
}

impl<T: IsFloat + PartialEq> PartialEq for OrdFloat<T> {
    fn eq(&self, other: &Self) -> bool {
        match (self.0.is_nan(), other.0.is_nan()) {
            (true, true) => true,
            _ => self.0 == other.0,
        }
    }
}

impl<T: PartialEq + IsFloat> Eq for OrdFloat<T> {}

impl<T: num::ToPrimitive> num::ToPrimitive for OrdFloat<T> {
    fn to_isize(&self) -> Option<isize> {
        self.0.to_isize()
    }

    fn to_i8(&self) -> Option<i8> {
        self.0.to_i8()
    }

    fn to_i16(&self) -> Option<i16> {
        self.0.to_i16()
    }

    fn to_i32(&self) -> Option<i32> {
        self.0.to_i32()
    }

    fn to_i64(&self) -> Option<i64> {
        self.0.to_i64()
    }

    fn to_usize(&self) -> Option<usize> {
        self.0.to_usize()
    }

    fn to_u8(&self) -> Option<u8> {
        self.0.to_u8()
    }

    fn to_u16(&self) -> Option<u16> {
        self.0.to_u16()
    }

    fn to_u32(&self) -> Option<u32> {
        self.0.to_u32()
    }

    fn to_u64(&self) -> Option<u64> {
        self.0.to_u64()
    }

    fn to_f32(&self) -> Option<f32> {
        self.0.to_f32()
    }

    fn to_f64(&self) -> Option<f64> {
        self.0.to_f64()
    }
}

pub fn f32_to_ordablef32(vals: &mut [f32]) -> &mut [OrdFloat<f32>] {
    unsafe { std::mem::transmute(vals) }
}

pub fn f64_to_ordablef64(vals: &mut [f64]) -> &mut [OrdFloat<f64>] {
    unsafe { std::mem::transmute(vals) }
}
