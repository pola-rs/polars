use std::ops::{Add, RangeInclusive, Sub};

use arrow::array::PrimitiveArray;
use arrow::bitmap::bitmask::BitMask;
use arrow::bitmap::{BitmapBuilder, MutableBitmap};
use arrow::datatypes::ArrowDataType;
use arrow::types::NativeType;
use num_traits::{FromPrimitive, ToPrimitive};
use polars_utils::total_ord::TotalOrd;

use super::RangedUniqueKernel;

/// A specialized unique kernel for [`PrimitiveArray`] for when all values are in a small known
/// range.
pub struct PrimitiveRangedUniqueState<T: NativeType> {
    seen: Seen,
    range: RangeInclusive<T>,
}

enum Seen {
    Small(u128),
    Large(MutableBitmap),
}

impl Seen {
    pub fn from_size(size: usize) -> Self {
        if size <= 128 {
            Self::Small(0)
        } else {
            Self::Large(MutableBitmap::from_len_zeroed(size))
        }
    }

    fn num_seen(&self) -> usize {
        match self {
            Seen::Small(v) => v.count_ones() as usize,
            Seen::Large(v) => v.set_bits(),
        }
    }

    fn has_seen_null(&self, size: usize) -> bool {
        match self {
            Self::Small(v) => v >> (size - 1) != 0,
            Self::Large(v) => v.get(size - 1),
        }
    }
}

impl<T: NativeType> PrimitiveRangedUniqueState<T>
where
    T: Add<T, Output = T> + Sub<T, Output = T> + ToPrimitive + FromPrimitive,
{
    pub fn new(min_value: T, max_value: T) -> Self {
        let size = (max_value - min_value).to_usize().unwrap();
        // Range is inclusive
        let size = size + 1;
        // One value is left for null
        let size = size + 1;

        Self {
            seen: Seen::from_size(size),
            range: min_value..=max_value,
        }
    }

    fn size(&self) -> usize {
        (*self.range.end() - *self.range.start())
            .to_usize()
            .unwrap()
            + 1
    }
}

impl<T: NativeType> RangedUniqueKernel for PrimitiveRangedUniqueState<T>
where
    T: Add<T, Output = T> + Sub<T, Output = T> + ToPrimitive + FromPrimitive,
{
    type Array = PrimitiveArray<T>;

    fn has_seen_all(&self) -> bool {
        let size = self.size();
        match &self.seen {
            Seen::Small(v) if size == 128 => !v == 0,
            Seen::Small(v) => *v == ((1 << size) - 1),
            Seen::Large(v) => BitMask::new(v.as_slice(), 0, size).unset_bits() == 0,
        }
    }

    fn append(&mut self, array: &Self::Array) {
        let size = self.size();
        match array.validity().as_ref().filter(|v| v.unset_bits() > 0) {
            None => {
                const STEP_SIZE: usize = 512;

                let mut i = 0;
                let values = array.values().as_slice();

                match self.seen {
                    Seen::Small(ref mut seen) => {
                        // Check every so often whether we have already seen all the values.
                        while *seen != ((1 << (size - 1)) - 1) && i < values.len() {
                            for v in values[i..].iter().take(STEP_SIZE) {
                                if cfg!(debug_assertions) {
                                    assert!(TotalOrd::tot_ge(v, self.range.start()));
                                    assert!(TotalOrd::tot_le(v, self.range.end()));
                                }

                                let v = *v - *self.range.start();
                                let v = unsafe { v.to_usize().unwrap_unchecked() };
                                *seen |= 1 << v;
                            }

                            i += STEP_SIZE;
                        }
                    },
                    Seen::Large(ref mut seen) => {
                        // Check every so often whether we have already seen all the values.
                        while BitMask::new(seen.as_slice(), 0, size - 1).unset_bits() > 0
                            && i < values.len()
                        {
                            for v in values[i..].iter().take(STEP_SIZE) {
                                if cfg!(debug_assertions) {
                                    assert!(TotalOrd::tot_ge(v, self.range.start()));
                                    assert!(TotalOrd::tot_le(v, self.range.end()));
                                }

                                let v = *v - *self.range.start();
                                let v = unsafe { v.to_usize().unwrap_unchecked() };
                                seen.set(v, true);
                            }

                            i += STEP_SIZE;
                        }
                    },
                }
            },
            Some(_) => {
                let iter = array.non_null_values_iter();

                match self.seen {
                    Seen::Small(ref mut seen) => {
                        *seen |= 1 << (size - 1);

                        for v in iter {
                            if cfg!(debug_assertions) {
                                assert!(TotalOrd::tot_ge(&v, self.range.start()));
                                assert!(TotalOrd::tot_le(&v, self.range.end()));
                            }

                            let v = v - *self.range.start();
                            let v = unsafe { v.to_usize().unwrap_unchecked() };
                            *seen |= 1 << v;
                        }
                    },
                    Seen::Large(ref mut seen) => {
                        seen.set(size - 1, true);

                        for v in iter {
                            if cfg!(debug_assertions) {
                                assert!(TotalOrd::tot_ge(&v, self.range.start()));
                                assert!(TotalOrd::tot_le(&v, self.range.end()));
                            }

                            let v = v - *self.range.start();
                            let v = unsafe { v.to_usize().unwrap_unchecked() };
                            seen.set(v, true);
                        }
                    },
                }
            },
        }
    }

    fn append_state(&mut self, other: &Self) {
        debug_assert_eq!(self.size(), other.size());
        match (&mut self.seen, &other.seen) {
            (Seen::Small(lhs), Seen::Small(rhs)) => *lhs |= rhs,
            (Seen::Large(lhs), Seen::Large(ref rhs)) => {
                let mut lhs = lhs;
                <&mut MutableBitmap as std::ops::BitOrAssign<&MutableBitmap>>::bitor_assign(
                    &mut lhs, rhs,
                )
            },
            _ => unreachable!(),
        }
    }

    fn finalize_unique(self) -> Self::Array {
        let size = self.size();
        let seen = self.seen;

        let has_null = seen.has_seen_null(size);
        let num_values = seen.num_seen();
        let mut values = Vec::with_capacity(num_values);

        let mut offset = 0;
        match seen {
            Seen::Small(mut v) => {
                while v != 0 {
                    let shift = v.trailing_zeros();
                    offset += shift as u8;
                    values.push(*self.range.start() + T::from_u8(offset).unwrap());

                    v >>= shift + 1;
                    offset += 1;
                }
            },
            Seen::Large(v) => {
                for offset in v.freeze().true_idx_iter() {
                    values.push(*self.range.start() + T::from_usize(offset).unwrap());
                }
            },
        }

        let validity = if has_null {
            let mut validity = BitmapBuilder::new();
            validity.extend_constant(values.len() - 1, true);
            validity.push(false);
            // The null has already been pushed.
            *values.last_mut().unwrap() = T::zeroed();
            Some(validity.freeze())
        } else {
            None
        };

        PrimitiveArray::new(ArrowDataType::from(T::PRIMITIVE), values.into(), validity)
    }

    fn finalize_n_unique(&self) -> usize {
        self.seen.num_seen()
    }

    fn finalize_n_unique_non_null(&self) -> usize {
        self.seen.num_seen() - usize::from(self.seen.has_seen_null(self.size()))
    }
}
