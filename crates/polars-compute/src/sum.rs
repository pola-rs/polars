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
/// Differs from num::WrappingAdd in that this is also implemented for floats.
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

wrapping_impl!(WrappingAdd, add, pf16);
wrapping_impl!(WrappingAdd, add, f32);
wrapping_impl!(WrappingAdd, add, f64);

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
///
/// Every implementation adds values up one at a time and starts from zero, so the two operations
/// that takes are required here rather than at each of the kernels below: a caller that has a
/// [`WrappingSum`] type in hand can add up a chunk of it however the chunk is laid out.
pub trait WrappingSum: WrappingAdd + Zero + Sized {
    fn wrapping_sum(vals: &[Self]) -> Self;
    fn wrapping_sum_with_validity(vals: &[Self], mask: &BitMask) -> Self;
}

/// The validity mask of `arr` laid out one bit per element, for the kernels below to read as
/// words, or `None` where every element is valid.
///
/// `count` is the number of non-null elements, which the callers have already worked out. A mask
/// that leaves none of them null has nothing left to say and is dropped, and one that repeats a
/// single bit never reaches here: the callers return before asking when no element is non-null,
/// which is the only other thing a repeated bit could mean.
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

    // A chunk that repeats one value adds that value up once per non-null element, which is
    // `O(log n)` doublings rather than a pass over the chunk.
    if let Some(value) = arr.scalar_values() {
        return repeat_wrapping_add(value, count);
    }

    let values = arr
        .flat_values()
        .expect("a values buffer that is not scalar is flat");
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

    if let Some(value) = arr.scalar_values() {
        return repeat_wrapping_add(value.into(), count);
    }

    let values = arr
        .flat_values()
        .expect("a values buffer that is not scalar is flat");
    match flat_mask_of(arr, count) {
        Some(mask) => wrapping_sum_with_mask_scalar_upcast(values, &BitMask::from_bitmap(&mask)),
        None => values
            .iter()
            .fold(S::zero(), |a, b| a.wrapping_add(&(*b).into())),
    }
}

/// `value` added to itself `count` times, wrapping around on overflow, by repeated doubling.
fn repeat_wrapping_add<T: Zero + WrappingAdd + Copy>(value: T, count: usize) -> T {
    let mut total = T::zero();
    let mut addend = value;
    let mut remaining = count;

    while remaining > 0 {
        if remaining % 2 == 1 {
            total = total.wrapping_add(&addend);
        }
        remaining /= 2;
        if remaining > 0 {
            addend = addend.wrapping_add(&addend);
        }
    }

    total
}

#[cfg(test)]
mod tests {
    use arrow::bitmap::Bitmap;

    use super::*;

    /// `length` copies of `value`, marked by `validity`, in both representations.
    fn repeated<T: NativeType>(
        value: T,
        validity: Option<&Bitmap>,
        length: usize,
    ) -> [PlPrimitiveArray<T>; 2] {
        let scalar =
            PlPrimitiveArray::new_scalar(value, length).with_validity_broadcast(validity.cloned());
        let flat = PlPrimitiveArray::from_vec(vec![value; length])
            .with_validity_broadcast(validity.cloned());
        assert_eq!(scalar, flat);
        [scalar, flat]
    }

    /// A chunk that repeats one value adds that value up once per non-null element, and reaches
    /// the same total as walking the chunk would.
    #[test]
    fn a_repeated_value_is_added_up_once_per_element() {
        for length in [0, 1, 2, 3, 65, 300] {
            for valid in 0..=length {
                let mask: Bitmap = (0..length).map(|i| i < valid).collect();
                for validity in [None, Some(&mask)] {
                    let [scalar, flat] = repeated(7i32, validity, length);
                    let count = validity.map_or(length, |_| valid) as i32;

                    assert_eq!(
                        wrapping_sum_arr(&scalar),
                        7 * count,
                        "the sum of {scalar:?}",
                    );
                    assert_eq!(wrapping_sum_arr(&scalar), wrapping_sum_arr(&flat));

                    let upcast: i64 = wrapping_sum_arr_upcast(&scalar);
                    assert_eq!(upcast, 7 * i64::from(count));
                    assert_eq!(upcast, wrapping_sum_arr_upcast::<i32, i64>(&flat));
                }
            }
        }
    }

    /// The doubling wraps around exactly where adding the value up one at a time would.
    #[test]
    fn a_repeated_value_wraps_around() {
        for (value, length) in [(1u8, 256), (1u8, 300), (200u8, 3), (u8::MAX, 2)] {
            let [scalar, flat] = repeated(value, None, length);
            let expected = (0..length).fold(0u8, |acc, _| acc.wrapping_add(value));

            assert_eq!(wrapping_sum_arr(&scalar), expected, "{length} x {value}");
            assert_eq!(wrapping_sum_arr(&flat), expected);
        }

        // Signed values wrap the same way, and a negative one adds up downwards.
        let [scalar, flat] = repeated(-3i8, None, 50);
        let expected = (0..50).fold(0i8, |acc, _| acc.wrapping_add(-3));
        assert_eq!(wrapping_sum_arr(&scalar), expected);
        assert_eq!(wrapping_sum_arr(&flat), expected);
    }

    /// Upcasting adds up in the wider type, so a total that would have wrapped does not.
    #[test]
    fn upcasting_adds_up_without_wrapping() {
        let [scalar, flat] = repeated(200u8, None, 100);

        let scalar_sum: i64 = wrapping_sum_arr_upcast(&scalar);
        assert_eq!(scalar_sum, 20_000);
        assert_eq!(scalar_sum, wrapping_sum_arr_upcast::<u8, i64>(&flat));

        // The narrow kernel wraps at the width of the values instead.
        assert_eq!(wrapping_sum_arr(&scalar), 20_000u32 as u8);
    }

    /// A chunk with no non-null element adds up to nothing, whatever value sits under the mask.
    #[test]
    fn nothing_adds_up_to_zero() {
        let all_null = PlPrimitiveArray::<i32>::new_full_null(300);
        assert_eq!(wrapping_sum_arr(&all_null), 0);
        assert_eq!(wrapping_sum_arr_upcast::<i32, i64>(&all_null), 0);

        // Values laid out one per element under a mask that leaves every one of them null.
        let masked = PlPrimitiveArray::from_vec(vec![5i32; 4])
            .with_validity_broadcast(Some(Bitmap::new_zeroed(1)));
        assert_eq!(wrapping_sum_arr(&masked), 0);

        let empty = PlPrimitiveArray::<i32>::new_empty();
        assert_eq!(wrapping_sum_arr(&empty), 0);
        assert_eq!(wrapping_sum_arr_upcast::<i32, i64>(&empty), 0);
    }

    /// A chunk laid out one value per element adds up its non-null elements as it always has.
    #[test]
    fn null_elements_are_passed_over() {
        let arr = PlPrimitiveArray::from_iter([Some(1i32), None, Some(2), Some(3)]);
        assert_eq!(wrapping_sum_arr(&arr), 6);
        assert_eq!(wrapping_sum_arr_upcast::<i32, i64>(&arr), 6);

        // A mask that repeats a set bit leaves every element counting.
        let all_valid = PlPrimitiveArray::from_vec(vec![1i32, 2, 3])
            .with_validity_broadcast(Some(Bitmap::new_with_value(true, 1)));
        assert_eq!(wrapping_sum_arr(&all_valid), 6);
    }

    /// A value repeated `count` times doubles up to the same total as adding it up one at a time,
    /// for every count a chunk could have.
    #[test]
    fn doubling_matches_adding_one_at_a_time() {
        for count in 0..=200usize {
            let expected = (0..count).fold(0u16, |acc, _| acc.wrapping_add(37));
            assert_eq!(repeat_wrapping_add(37u16, count), expected, "{count} x 37");
        }
    }
}
