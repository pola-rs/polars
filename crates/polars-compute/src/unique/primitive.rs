use std::ops::{Add, RangeInclusive, Sub};

use arrow::array::PrimitiveArray;
use arrow::bitmap::MutableBitmap;
use arrow::datatypes::ArrowDataType;
use arrow::types::NativeType;
use num_traits::FromPrimitive;
use polars_utils::float::IsFloat;
use polars_utils::total_ord::TotalOrd;

use super::RangedUniqueKernel;

/// A specialized unique kernel for [`PrimitiveArray`] for when all values are in a small known
/// range.
pub struct PrimitiveRangedUniqueState<T: NativeType> {
    seen: u128,
    range: RangeInclusive<T>,
    has_null: bool,
    data_type: ArrowDataType,
}

impl<T: NativeType> PrimitiveRangedUniqueState<T>
where
    T: Add<T, Output = T> + Sub<T, Output = T> + FromPrimitive + IsFloat,
{
    pub fn new(
        min_value: T,
        max_value: T,
        has_null: bool,
        data_type: ArrowDataType,
    ) -> Option<Self> {
        // We cannot really do this for floating point number as these are not as discrete as
        // integers.
        if T::is_float() {
            return None;
        }

        if TotalOrd::tot_gt(
            &(max_value - min_value),
            &T::from_u8(128 - u8::from(has_null)).unwrap(),
        ) {
            return None;
        }

        Some(Self {
            seen: 0,
            range: min_value..=max_value,
            has_null,
            data_type,
        })
    }

    fn len(&self) -> u8 {
        (*self.range.end() - *self.range.start()).to_le_bytes()[0]
    }

    fn has_seen_null(&self) -> bool {
        self.has_null && self.seen & 1 != 0
    }

    #[inline(always)]
    fn to_value(&self, scalar: Option<T>) -> u8 {
        match scalar {
            None => {
                debug_assert!(self.has_null);
                0
            },
            Some(v) => {
                debug_assert!(<T as TotalOrd>::tot_le(&v, self.range.end()));
                debug_assert!(<T as TotalOrd>::tot_ge(&v, self.range.start()));

                (v - *self.range.start()).to_le_bytes()[0] + u8::from(self.has_null)
            },
        }
    }
}

impl<T: NativeType> RangedUniqueKernel for PrimitiveRangedUniqueState<T>
where
    T: Add<T, Output = T> + Sub<T, Output = T> + FromPrimitive + IsFloat,
{
    type Array = PrimitiveArray<T>;

    fn has_seen_all(&self) -> bool {
        let len = self.len();
        let bit_length = len + u8::from(self.has_null);

        debug_assert!(bit_length > 0);
        debug_assert!(bit_length <= 128);

        self.seen == (1u128 << len).wrapping_sub(1)
    }

    fn append(&mut self, array: &Self::Array) {
        const STEP_SIZE: usize = 128;

        if !self.has_null {
            let mut i = 0;
            let values = array.values().as_slice();

            while !self.has_seen_all() && i < values.len() {
                for v in values[i..].iter().take(STEP_SIZE) {
                    self.seen |= 1 << self.to_value(Some(*v));
                }

                i += STEP_SIZE;
            }
        } else {
            let mut i = 0;
            let mut values = array.iter();

            while !self.has_seen_all() && i < values.len() {
                for _ in 0..STEP_SIZE {
                    let Some(v) = values.next() else {
                        break;
                    };
                    self.seen |= 1 << self.to_value(v.copied());
                }

                i += STEP_SIZE;
            }
        }
    }

    fn finalize_unique(self) -> Self::Array {
        let mut seen = self.seen;

        let num_values = seen.count_ones() as usize;
        let mut values = Vec::with_capacity(num_values);

        let (values, validity) = if self.has_seen_null() {
            let mut validity = MutableBitmap::with_capacity(num_values);

            values.push(T::zeroed());
            validity.push(false);
            seen >>= 1;

            let mut offset = 0u8;
            while seen != 0 {
                let shift = self.seen.trailing_zeros();
                offset += shift as u8;
                values.push(*self.range.start() + T::from_u8(offset).unwrap());
                validity.push(true);

                seen >>= shift + 1;
                offset += 1;
            }

            (values, Some(validity.freeze()))
        } else {
            seen >>= u8::from(self.has_null);

            let mut offset = 0u8;
            while seen != 0 {
                let shift = seen.trailing_zeros();
                offset += shift as u8;
                values.push(*self.range.start() + T::from_u8(offset).unwrap());

                seen >>= shift + 1;
                offset += 1;
            }

            (values, None)
        };

        PrimitiveArray::new(self.data_type, values.into(), validity)
    }

    fn finalize_n_unique(self) -> usize {
        self.seen.count_ones() as usize
    }

    fn finalize_n_unique_non_null(self) -> usize {
        (self.seen & !1).count_ones() as usize
    }
}
