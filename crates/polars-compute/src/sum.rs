use std::borrow::Cow;
use std::ops::Add;
#[cfg(feature = "simd")]
use std::simd::Select;
#[cfg(feature = "simd")]
use std::simd::prelude::*;

use arrow::bitmap::Bitmap;
use arrow::bitmap::bitmask::BitMask;
use arrow::types::NativeType;
use num_traits::Zero;
use polars_array::PlPrimitiveArray;
use polars_utils::float16::pf16;

macro_rules! wrapping_impl {
    ($trait_name:ident, $method:ident, $t:ty) => {
        impl $trait_name for $t {
            #[inline(always)]
            fn wrapping_add(&self, v: &Self) -> Self {
                <$t>::$method(*self, *v)
            }
        }
    };
}

/// Performs addition that wraps around on overflow.
///
/// Differs from num::WrappingAdd in that this is also implemented for floats, which have no
/// overflow to wrap around: they add algebraically, letting a run of additions be reassociated.
pub trait WrappingAdd: Sized {
    /// Wrapping (modular) addition. Computes `self + other`, wrapping around at
    /// the boundary of the type.
    fn wrapping_add(&self, v: &Self) -> Self;
}

wrapping_impl!(WrappingAdd, wrapping_add, u8);
wrapping_impl!(WrappingAdd, wrapping_add, u16);
wrapping_impl!(WrappingAdd, wrapping_add, u32);
wrapping_impl!(WrappingAdd, wrapping_add, u64);
wrapping_impl!(WrappingAdd, wrapping_add, usize);
wrapping_impl!(WrappingAdd, wrapping_add, u128);

wrapping_impl!(WrappingAdd, wrapping_add, i8);
wrapping_impl!(WrappingAdd, wrapping_add, i16);
wrapping_impl!(WrappingAdd, wrapping_add, i32);
wrapping_impl!(WrappingAdd, wrapping_add, i64);
wrapping_impl!(WrappingAdd, wrapping_add, isize);
wrapping_impl!(WrappingAdd, wrapping_add, i128);

// `pf16` has no algebraic addition of its own; it is summed through the `f32` kernel anyway.
wrapping_impl!(WrappingAdd, add, pf16);
wrapping_impl!(WrappingAdd, algebraic_add, f32);
wrapping_impl!(WrappingAdd, algebraic_add, f64);

#[cfg(feature = "simd")]
const STRIPE: usize = 16;

fn wrapping_sum_with_mask_scalar<T: Zero + WrappingAdd + Copy>(vals: &[T], mask: &BitMask) -> T {
    assert!(vals.len() == mask.len());
    vals.iter()
        .enumerate()
        .map(|(i, x)| {
            // No filter but rather select of 0 for cmov opt.
            if mask.get(i) { *x } else { T::zero() }
        })
        .fold(T::zero(), |a, b| a.wrapping_add(&b))
}

fn wrapping_sum_with_mask_scalar_upcast<T, S>(vals: &[T], mask: &BitMask) -> S
where
    T: NativeType + Zero + Into<S>,
    S: Zero + WrappingAdd + Copy,
{
    assert!(vals.len() == mask.len());
    vals.iter()
        .enumerate()
        .map(|(i, x)| {
            // No filter but rather select of 0 for cmov opt.
            if mask.get(i) { *x } else { T::zero() }
        })
        .fold(S::zero(), |a, b| a.wrapping_add(&b.into()))
}

#[cfg(not(feature = "simd"))]
impl<T> WrappingSum for T
where
    T: NativeType + WrappingAdd + Zero,
{
    fn wrapping_sum(vals: &[Self]) -> Self {
        vals.iter()
            .copied()
            .fold(T::zero(), |a, b| a.wrapping_add(&b))
    }

    fn wrapping_sum_with_validity(vals: &[Self], mask: &BitMask) -> Self {
        wrapping_sum_with_mask_scalar(vals, mask)
    }
}

#[cfg(feature = "simd")]
impl<T> WrappingSum for T
where
    T: NativeType + WrappingAdd + Zero + crate::SimdPrimitive,
{
    fn wrapping_sum(vals: &[Self]) -> Self {
        vals.iter()
            .copied()
            .fold(T::zero(), |a, b| a.wrapping_add(&b))
    }

    fn wrapping_sum_with_validity(vals: &[Self], mask: &BitMask) -> Self {
        assert!(vals.len() == mask.len());
        let remainder = vals.len() % STRIPE;
        let (rest, main) = vals.split_at(remainder);
        let (rest_mask, main_mask) = mask.split_at(remainder);
        let zero: Simd<T, STRIPE> = Simd::default();

        let vsum = main
            .as_chunks::<STRIPE>()
            .0
            .iter()
            .enumerate()
            .map(|(i, a)| {
                let m: Mask<T::Mask, STRIPE> = main_mask.get_simd(i * STRIPE);
                m.select(Simd::from_slice(a), zero)
            })
            .fold(zero, |a, b| {
                let a = a.to_array();
                let b = b.to_array();
                Simd::from_array(std::array::from_fn(|i| a[i].wrapping_add(&b[i])))
            });

        let mainsum = vsum
            .to_array()
            .into_iter()
            .fold(T::zero(), |a, b| a.wrapping_add(&b));

        // TODO: faster remainder.
        let restsum = wrapping_sum_with_mask_scalar(rest, &rest_mask);
        mainsum.wrapping_add(&restsum)
    }
}

#[cfg(feature = "simd")]
impl WrappingSum for u128 {
    fn wrapping_sum(vals: &[Self]) -> Self {
        vals.iter().copied().fold(0, |a, b| a.wrapping_add(b))
    }

    fn wrapping_sum_with_validity(vals: &[Self], mask: &BitMask) -> Self {
        wrapping_sum_with_mask_scalar(vals, mask)
    }
}

#[cfg(feature = "simd")]
impl WrappingSum for i128 {
    fn wrapping_sum(vals: &[Self]) -> Self {
        vals.iter().copied().fold(0, |a, b| a.wrapping_add(b))
    }

    fn wrapping_sum_with_validity(vals: &[Self], mask: &BitMask) -> Self {
        wrapping_sum_with_mask_scalar(vals, mask)
    }
}

#[cfg(feature = "simd")]
impl WrappingSum for pf16 {
    fn wrapping_sum(_vals: &[Self]) -> Self {
        unimplemented!("should have been dispatched to other sum kernel")
    }

    fn wrapping_sum_with_validity(_vals: &[Self], _mask: &BitMask) -> Self {
        unimplemented!("should have been dispatched to other sum kernel")
    }
}

/// Adding up a slice of values, wrapping around on overflow.
pub trait WrappingSum: WrappingAdd + Zero + Sized {
    fn wrapping_sum(vals: &[Self]) -> Self;
    fn wrapping_sum_with_validity(vals: &[Self], mask: &BitMask) -> Self;
}

/// The validity mask of `arr` laid out one bit per element, or `None` where every element is valid.
fn flat_mask_of<T: NativeType>(arr: &PlPrimitiveArray<T>, count: usize) -> Option<Cow<'_, Bitmap>> {
    (count < arr.len()).then(|| {
        arr.validity()
            .expect("a mask that leaves an element null is present")
            .to_flat()
    })
}

/// Adds up every non-null element of `arr`, wrapping around on overflow.
pub fn wrapping_sum_arr<T>(arr: &PlPrimitiveArray<T>) -> T
where
    T: NativeType + WrappingSum,
{
    let count = arr.len() - arr.null_count();
    if count == 0 {
        return T::zero();
    }

    // A chunk that repeats one value adds that value up once per non-null element, which for an
    // integer is a single multiplication rather than a pass over the chunk. A float still pays a
    // pass, but a float chunk is summed by [`crate::float_sum`] instead.
    if let Some(value) = arr.scalar_value_ignore_validity() {
        return repeat_wrapping_add(value, count);
    }

    let values = arr.flat_values().unwrap();

    match flat_mask_of(arr, count) {
        Some(mask) => WrappingSum::wrapping_sum_with_validity(values, &BitMask::from_bitmap(&mask)),
        None => WrappingSum::wrapping_sum(values),
    }
}

/// As [`wrapping_sum_arr`], accumulating into the wider type `S`.
pub fn wrapping_sum_arr_upcast<T, S>(arr: &PlPrimitiveArray<T>) -> S
where
    T: NativeType + Zero + Into<S>,
    S: Zero + WrappingAdd + Copy,
{
    let count = arr.len() - arr.null_count();
    if count == 0 {
        return S::zero();
    }

    if let Some(value) = arr.scalar_value_ignore_validity() {
        return repeat_wrapping_add(value.into(), count);
    }

    let values = arr.flat_values().unwrap();

    match flat_mask_of(arr, count) {
        Some(mask) => wrapping_sum_with_mask_scalar_upcast(values, &BitMask::from_bitmap(&mask)),
        None => values
            .iter()
            .fold(S::zero(), |a, b| a.wrapping_add(&(*b).into())),
    }
}

/// `value` added to itself `count` times, wrapping around on overflow.
///
/// Written as the run of additions it is, which each type's addition then collapses on its own:
/// the optimiser recognises the integer add-recurrence and rewrites the whole run as one
/// multiplication, and the floats add algebraically, which lets it vectorise the run instead.
fn repeat_wrapping_add<T: Zero + WrappingAdd + Copy>(value: T, count: usize) -> T {
    (0..count).fold(T::zero(), |total, _| total.wrapping_add(&value))
}

#[cfg(test)]
mod tests {
    use polars_buffer::Buffer;

    use super::*;

    /// A chunk of `len` elements that all repeat `value`, in the scalar representation.
    fn scalar<T: NativeType>(value: T, len: usize) -> PlPrimitiveArray<T> {
        PlPrimitiveArray::new_broadcast(Buffer::from(vec![value]), len.max(1), None).sliced(0, len)
    }

    /// The scalar path collapses a run of additions into a multiplication, which has to wrap
    /// around exactly where adding the value up one at a time would have.
    #[test]
    fn scalar_chunk_sums_as_the_run_of_additions_it_stands_for() {
        for count in [0, 1, 2, 3, 255, 256, 257, 1000] {
            let value = i64::MAX / 3;
            let added = (0..count).fold(0i64, |a, _| a.wrapping_add(value));
            assert_eq!(wrapping_sum_arr(&scalar(value, count)), added, "{count}");

            let added = (0..count).fold(0u8, |a, _| a.wrapping_add(200));
            assert_eq!(wrapping_sum_arr(&scalar(200u8, count)), added, "{count}");
        }
    }

    /// The upcast accumulates in the wider type, so it wraps around at that type's boundary.
    #[test]
    fn scalar_chunk_sums_upcast_at_the_wider_boundary() {
        for count in [0, 1, 2, 3, 257, 1000] {
            let added = (0..count).fold(0i128, |a, _| a.wrapping_add(i32::MIN as i128));
            let summed: i128 = wrapping_sum_arr_upcast(&scalar(i32::MIN, count));
            assert_eq!(summed, added, "{count}");
        }
    }
}
