//! The {n,arg}-unique kernels over a chunk, amortized over the several groups they answer for.

use arrow::array::View;
use arrow::bitmap::Bitmap;
use arrow::bitmap::bitmask::BitMask;
use arrow::types::NativeType;
use polars_array::{
    PlArray, PlArrayType, PlBinaryArray, PlBinaryViewArray, PlBitmapRef, PlBooleanArray,
    PlPrimitiveArray, PrimitiveType, with_match_pl_primitive_array_type,
};
use polars_utils::aliases::{InitHashMaps, PlHashSet};
use polars_utils::float16::pf16;
use polars_utils::total_ord::{TotalEq, TotalHash, TotalOrdWrap};
use polars_utils::{IdxSize, UnitVec};

use crate::nesting::downcast;

/// What a state that is not [`RepeatedUnique`] was built over, and is therefore handed.
const FLAT: &str =
    "a state is built over the chunk it walks, which lays its values out one per element";

/// The values of a chunk whose state walks them one per element.
fn flat_values<T: NativeType>(values: &PlPrimitiveArray<T>) -> &[T] {
    values.flat_values().expect(FLAT).as_slice()
}

/// As [`flat_values`], over the bits of a boolean chunk.
fn flat_bits(values: &PlBooleanArray) -> &Bitmap {
    values.flat_values().expect(FLAT)
}

/// As [`flat_values`], over the views of a binary view chunk.
fn flat_views(values: &PlBinaryViewArray) -> &[View] {
    values.flat_views().expect(FLAT).as_slice()
}

/// The validity mask of a chunk that has a null under it, which holds one bit per element.
fn flat_validity(values: &dyn PlArray) -> &Bitmap {
    values
        .validity()
        .expect("a chunk with a null under it carries a mask")
        .flat_bitmap()
        .expect(FLAT)
}

// Rebuild the amortized hashset when capacity exceeds `needed` by this
// factor and is above `REBUILD_MIN_CAPACITY`. `.clear()` is O(capacity);
// rebuilding bounds worst-case clear cost under heavy group-size skew.
// See polars#27655.
const REBUILD_CAPACITY_RATIO: usize = 4;
const REBUILD_MIN_CAPACITY: usize = 1024;

#[inline]
fn reset_amortized<T>(set: &mut PlHashSet<T>, needed: usize) {
    if set.capacity() > REBUILD_CAPACITY_RATIO * needed && set.capacity() > REBUILD_MIN_CAPACITY {
        *set = PlHashSet::with_capacity(needed);
    } else {
        set.clear();
    }
}

pub trait AmortizedUnique: Send + Sync + 'static {
    fn new_empty(&self) -> Box<dyn AmortizedUnique>;

    /// Retain indices of items that are unique.
    ///
    /// This is always stable.
    ///
    /// # Safety
    ///
    /// All indices i should be 0 <= i < values.len()
    unsafe fn retain_unique(&mut self, values: &dyn PlArray, idxs: &mut UnitVec<IdxSize>);

    /// Get the indices of unique items in an array slice.
    ///
    /// This is always stable.
    fn arg_unique(
        &mut self,
        values: &dyn PlArray,
        idxs: &mut UnitVec<IdxSize>,
        start: IdxSize,
        length: IdxSize,
    );

    /// Get the number of unique items in an array at `idxs`.
    ///
    /// # Safety
    ///
    /// All indices i should be 0 <= i < values.len()
    unsafe fn n_unique_idx(&mut self, values: &dyn PlArray, idxs: &[IdxSize]) -> IdxSize;

    /// Get the number of unique items in an array slice.
    fn n_unique_slice(&mut self, values: &dyn PlArray, start: IdxSize, length: IdxSize) -> IdxSize;
}

/// The state that answers the unique kernels over `values`, and over any chunk like it.
pub fn amortized_unique_like(values: &dyn PlArray) -> Box<dyn AmortizedUnique> {
    // A chunk that holds one value over and over has that value and a null in it and nothing
    // else, whichever value it is: which of the two an element is is all its validity mask says,
    // and neither a hashset nor the values buffer has anything to add. A chunk of nothing but
    // nulls — an empty one included — is one of these in its own right, the one element it holds
    // being the null.
    if repeats_one_value(values) || values.null_count() == values.len() {
        return Box::new(RepeatedUnique);
    }

    match values.array_type() {
        PlArrayType::Boolean => Box::new(BooleanUnique) as _,
        PlArrayType::Primitive(pt) => match pt {
            PrimitiveType::Int8 => Box::new(PrimitiveArgUnique::<i8>::default()) as _,
            PrimitiveType::Int16 => Box::new(PrimitiveArgUnique::<i16>::default()) as _,
            PrimitiveType::Int32 => Box::new(PrimitiveArgUnique::<i32>::default()) as _,
            PrimitiveType::Int64 => Box::new(PrimitiveArgUnique::<i64>::default()) as _,
            PrimitiveType::Int128 => Box::new(PrimitiveArgUnique::<i128>::default()) as _,
            PrimitiveType::UInt8 => Box::new(PrimitiveArgUnique::<u8>::default()) as _,
            PrimitiveType::UInt16 => Box::new(PrimitiveArgUnique::<u16>::default()) as _,
            PrimitiveType::UInt32 => Box::new(PrimitiveArgUnique::<u32>::default()) as _,
            PrimitiveType::UInt64 => Box::new(PrimitiveArgUnique::<u64>::default()) as _,
            PrimitiveType::UInt128 => Box::new(PrimitiveArgUnique::<u128>::default()) as _,
            PrimitiveType::Float16 => Box::new(PrimitiveArgUnique::<pf16>::default()) as _,
            PrimitiveType::Float32 => Box::new(PrimitiveArgUnique::<f32>::default()) as _,
            PrimitiveType::Float64 => Box::new(PrimitiveArgUnique::<f64>::default()) as _,
            PrimitiveType::Int256 => unreachable!(),
            PrimitiveType::DaysMs => unreachable!(),
            PrimitiveType::MonthDayNano => unreachable!(),
            PrimitiveType::MonthDayMillis => unreachable!(),
        },
        PlArrayType::BinaryView => Box::new(BinaryViewUnique::default()) as _,
        PlArrayType::Binary => Box::new(BinaryUnique::default()) as _,

        // A null chunk holds nothing but nulls, and was answered above.
        PlArrayType::Null => unreachable!(),

        PlArrayType::FixedSizeBinary => unreachable!(),

        // Should be handled through BinaryView.
        PlArrayType::Utf8View => unreachable!(),

        // Should be handled through row encoding.
        PlArrayType::FixedSizeList => unreachable!(),
        PlArrayType::List => unreachable!(),
        PlArrayType::Struct => unreachable!(),

        PlArrayType::Object { .. } => unreachable!(),
    }
}

/// Whether the values of `values` are one slot standing for every element.
fn repeats_one_value(values: &dyn PlArray) -> bool {
    match values.array_type() {
        PlArrayType::Boolean => downcast::<PlBooleanArray>(values).values_are_scalar(),
        PlArrayType::Primitive(_) => with_match_pl_primitive_array_type!(values, |$T| {
            downcast::<PlPrimitiveArray<$T>>(values).values_are_scalar()
        })
        .expect("a primitive array has a primitive element type"),
        PlArrayType::BinaryView => downcast::<PlBinaryViewArray>(values).views_are_scalar(),
        // Offsets that repeat cut the same bytes out of the values buffer for every element.
        PlArrayType::Binary => downcast::<PlBinaryArray>(values).offsets_are_scalar(),
        _ => false,
    }
}

/// The unique kernels over a chunk that holds one value over and over.
struct RepeatedUnique;

impl RepeatedUnique {
    /// Whether the element at `i` is there at all, read through a mask resolved once.
    fn valid<'a>(validity: &'a Option<PlBitmapRef<'a>>) -> impl Fn(IdxSize) -> bool + 'a {
        move |i| validity.as_ref().is_none_or(|mask| mask.get(i as usize))
    }

    /// The first of the `length` elements from `start` that is not of the kind of the one there.
    fn first_differing(values: &dyn PlArray, start: IdxSize, length: IdxSize) -> Option<IdxSize> {
        // A chunk with no mask over it is the repeated value throughout.
        let validity = values.validity()?;
        // And a mask that repeats one bit says the same of every element it covers.
        let mask = validity.flat_bitmap()?;

        let mask = BitMask::from_bitmap(mask).sliced(start as usize, length as usize);
        let leading_zeros = mask.leading_zeros();
        let leading = if leading_zeros == 0 {
            mask.leading_ones()
        } else {
            leading_zeros
        };

        (leading < mask.len()).then(|| start + leading as IdxSize)
    }
}

impl AmortizedUnique for RepeatedUnique {
    fn new_empty(&self) -> Box<dyn AmortizedUnique> {
        Box::new(RepeatedUnique)
    }

    unsafe fn retain_unique(&mut self, values: &dyn PlArray, idxs: &mut UnitVec<IdxSize>) {
        if idxs.len() <= 1 {
            return;
        }

        let validity = values.validity();
        let valid = Self::valid(&validity);

        // SAFETY: function invariant.
        let first = valid(idxs[0]);
        *idxs = match idxs[1..].iter().position(|&i| valid(i) != first) {
            None => UnitVec::from_slice(&[idxs[0]]),
            Some(i) => UnitVec::from_slice(&[idxs[0], idxs[1 + i]]),
        };
    }

    fn arg_unique(
        &mut self,
        values: &dyn PlArray,
        idxs: &mut UnitVec<IdxSize>,
        start: IdxSize,
        length: IdxSize,
    ) {
        assert!(start.saturating_add(length) as usize <= values.len());
        if length == 0 {
            return;
        }

        idxs.push(start);
        idxs.extend(Self::first_differing(values, start, length));
    }

    unsafe fn n_unique_idx(&mut self, values: &dyn PlArray, idxs: &[IdxSize]) -> IdxSize {
        if idxs.len() <= 1 {
            return idxs.len() as IdxSize;
        }

        let validity = values.validity();
        let valid = Self::valid(&validity);

        // SAFETY: function invariant.
        let first = valid(idxs[0]);
        1 + IdxSize::from(idxs[1..].iter().any(|&i| valid(i) != first))
    }

    fn n_unique_slice(&mut self, values: &dyn PlArray, start: IdxSize, length: IdxSize) -> IdxSize {
        assert!(start.saturating_add(length) as usize <= values.len());
        if length <= 1 {
            return length;
        }

        1 + IdxSize::from(Self::first_differing(values, start, length).is_some())
    }
}

struct BooleanUnique;
#[derive(Default)]
struct PrimitiveArgUnique<T>(
    PlHashSet<TotalOrdWrap<T>>,
    PlHashSet<Option<TotalOrdWrap<T>>>,
);
#[derive(Default)]
struct BinaryViewUnique(PlHashSet<&'static [u8]>, PlHashSet<Option<&'static [u8]>>);
#[derive(Default)]
struct BinaryUnique(PlHashSet<&'static [u8]>, PlHashSet<Option<&'static [u8]>>);

impl AmortizedUnique for BooleanUnique {
    fn new_empty(&self) -> Box<dyn AmortizedUnique> {
        Box::new(BooleanUnique)
    }

    unsafe fn retain_unique(&mut self, values: &dyn PlArray, idxs: &mut UnitVec<IdxSize>) {
        if idxs.len() <= 1 {
            return;
        }

        let values = downcast::<PlBooleanArray>(values);

        if values.has_nulls() {
            let mut seen = 0u8;
            idxs.retain(|i| {
                if seen == 0b111 {
                    return false;
                }

                // SAFETY: function invariant.
                let v = match unsafe { values.get_unchecked(i as usize) } {
                    None => 1 << 0,
                    Some(false) => 1 << 1,
                    Some(true) => 1 << 2,
                };

                let keep = seen & v == 0;
                seen |= v;
                keep
            });
        } else {
            let values = flat_bits(values);
            if values.set_bits() == 0 || values.unset_bits() == 0 {
                *idxs = UnitVec::from_slice(&[idxs[0]]);
                return;
            }

            // SAFETY: function invariant.
            let fst = unsafe { values.get_bit_unchecked(idxs[0] as usize) };
            *idxs = match idxs[1..]
                .iter()
                // SAFETY: function invariant.
                .position(|&i| fst != unsafe { values.get_bit_unchecked(i as usize) })
            {
                None => UnitVec::from_slice(&[idxs[0]]),
                Some(i) => UnitVec::from_slice(&[idxs[0], idxs[1 + i]]),
            };
        }
    }

    fn arg_unique(
        &mut self,
        values: &dyn PlArray,
        idxs: &mut UnitVec<IdxSize>,
        start: IdxSize,
        length: IdxSize,
    ) {
        if length <= 1 {
            if length == 1 {
                idxs.push(start);
            }
            return;
        }

        assert!(start.saturating_add(length) as usize <= values.len());
        let values = downcast::<PlBooleanArray>(values);

        if values.has_nulls() {
            let mut seen = 0u8;
            idxs.extend((start..start + length).filter(|i| {
                if seen == 0b111 {
                    return false;
                }

                // SAFETY: asserted before.
                let v = match unsafe { values.get_unchecked(*i as usize) } {
                    None => 1 << 0,
                    Some(false) => 1 << 1,
                    Some(true) => 1 << 2,
                };

                let keep = seen & v == 0;
                seen |= v;
                keep
            }));
        } else {
            let values = flat_bits(values);
            if values.set_bits() == 0 || values.unset_bits() == 0 {
                *idxs = UnitVec::from_slice(&[start]);
                return;
            }

            let values = BitMask::from_bitmap(values);
            let values = values.sliced(start as usize, length as usize);

            let leading_zeros = values.leading_zeros();
            if leading_zeros == values.len() {
                *idxs = UnitVec::from_slice(&[start]);
            } else if leading_zeros == 0 {
                let leading_ones = values.leading_ones();
                if leading_ones == values.len() {
                    *idxs = UnitVec::from_slice(&[start]);
                } else {
                    *idxs = UnitVec::from_slice(&[start, start + leading_ones as IdxSize]);
                }
            } else {
                *idxs = UnitVec::from_slice(&[start, start + leading_zeros as IdxSize]);
            }
        }
    }

    unsafe fn n_unique_idx(&mut self, values: &dyn PlArray, idxs: &[IdxSize]) -> IdxSize {
        if idxs.len() <= 1 {
            return idxs.len() as IdxSize;
        }

        let values = downcast::<PlBooleanArray>(values);

        if values.has_nulls() {
            let mut seen = 0u8;
            for &i in idxs {
                if seen == 0b111 {
                    break;
                }
                // SAFETY: function invariant.
                seen |= match unsafe { values.get_unchecked(i as usize) } {
                    None => 1 << 0,
                    Some(false) => 1 << 1,
                    Some(true) => 1 << 2,
                };
            }
            IdxSize::from(seen.count_ones())
        } else {
            let values = flat_bits(values);
            if values.set_bits() == 0 || values.unset_bits() == 0 {
                return 1;
            }

            // SAFETY: function invariant.
            let fst = unsafe { values.get_bit_unchecked(idxs[0] as usize) };
            for &i in &idxs[1..] {
                // SAFETY: function invariant.
                if fst != unsafe { values.get_bit_unchecked(i as usize) } {
                    return 2;
                }
            }
            1
        }
    }

    fn n_unique_slice(&mut self, values: &dyn PlArray, start: IdxSize, length: IdxSize) -> IdxSize {
        if length <= 1 {
            return length;
        }

        let values = downcast::<PlBooleanArray>(values);
        assert!(start.saturating_add(length) as usize <= values.len());

        if values.has_nulls() {
            let validity = BitMask::from_bitmap(flat_validity(values));
            let values = BitMask::from_bitmap(flat_bits(values));

            let validity = validity.sliced(start as usize, length as usize);
            let values = values.sliced(start as usize, length as usize);

            let num_valid = validity.set_bits();
            if num_valid == 0 {
                return 1;
            }

            if num_valid as IdxSize == length {
                let num_trues = values.set_bits() as IdxSize;
                1 + IdxSize::from(num_trues != length && num_trues != 0)
            } else {
                let num_trues = values.num_intersections_with(validity);
                2 + IdxSize::from(num_trues != num_valid && num_trues != 0)
            }
        } else {
            let values = flat_bits(values);
            if values.set_bits() == 0 || values.unset_bits() == 0 {
                return 1;
            }

            let values = BitMask::from_bitmap(values);
            let values = values.sliced(start as usize, length as usize);
            let num_trues = values.set_bits();
            1 + IdxSize::from(num_trues != 0 && num_trues != values.len())
        }
    }
}

impl<T: NativeType + TotalHash + TotalEq> AmortizedUnique for PrimitiveArgUnique<T> {
    fn new_empty(&self) -> Box<dyn AmortizedUnique> {
        Box::new(PrimitiveArgUnique::<T>::default())
    }

    unsafe fn retain_unique(&mut self, values: &dyn PlArray, idxs: &mut UnitVec<IdxSize>) {
        if idxs.len() <= 1 {
            return;
        }

        let values = downcast::<PlPrimitiveArray<T>>(values);

        if values.has_nulls() {
            reset_amortized(&mut self.1, idxs.len());
            idxs.retain(|i| {
                // SAFETY: function invariant.
                let value = unsafe { values.get_unchecked(i as usize) };
                let value = value.map(TotalOrdWrap);
                self.1.insert(value)
            });
        } else {
            reset_amortized(&mut self.0, idxs.len());
            let values = flat_values(values);
            idxs.retain(|i| {
                // SAFETY: function invariant.
                let value = *unsafe { values.get_unchecked(i as usize) };
                let value = TotalOrdWrap(value);
                self.0.insert(value)
            });
        }
    }

    fn arg_unique(
        &mut self,
        values: &dyn PlArray,
        idxs: &mut UnitVec<IdxSize>,
        start: IdxSize,
        length: IdxSize,
    ) {
        if length <= 1 {
            if length == 1 {
                idxs.push(start);
            }
            return;
        }

        let values = downcast::<PlPrimitiveArray<T>>(values);
        assert!(start.saturating_add(length) as usize <= values.len());

        if values.has_nulls() {
            reset_amortized(&mut self.1, length as usize);
            idxs.extend((start..start + length).filter(|i| {
                // SAFETY: asserted before.
                let value = unsafe { values.get_unchecked(*i as usize) };
                let value = value.map(TotalOrdWrap);
                self.1.insert(value)
            }));
        } else {
            reset_amortized(&mut self.0, length as usize);
            let values = flat_values(values);
            idxs.extend(
                values[start as usize..][..length as usize]
                    .iter()
                    .enumerate()
                    .filter_map(|(i, value)| {
                        let value = TotalOrdWrap(*value);
                        self.0.insert(value).then_some(i as IdxSize + start)
                    }),
            );
        }
    }

    unsafe fn n_unique_idx(&mut self, values: &dyn PlArray, idxs: &[IdxSize]) -> IdxSize {
        if idxs.len() <= 1 {
            return idxs.len() as IdxSize;
        }

        let values = downcast::<PlPrimitiveArray<T>>(values);

        if values.has_nulls() {
            reset_amortized(&mut self.1, idxs.len());
            self.1.extend(idxs.iter().map(|&i| {
                // SAFETY: function invariant.
                let value = unsafe { values.get_unchecked(i as usize) };
                value.map(TotalOrdWrap)
            }));
            self.1.len() as IdxSize
        } else {
            let values = flat_values(values);
            reset_amortized(&mut self.0, idxs.len());
            self.0.extend(idxs.iter().map(|&i| {
                // SAFETY: function invariant.
                let value = *unsafe { values.get_unchecked(i as usize) };
                TotalOrdWrap(value)
            }));
            self.0.len() as IdxSize
        }
    }

    fn n_unique_slice(&mut self, values: &dyn PlArray, start: IdxSize, length: IdxSize) -> IdxSize {
        if length <= 1 {
            return length;
        }

        let values = downcast::<PlPrimitiveArray<T>>(values);
        assert!(start.saturating_add(length) as usize <= values.len());

        if values.has_nulls() {
            reset_amortized(&mut self.1, length as usize);
            self.1.extend((start..start + length).map(|i| {
                // SAFETY: asserted before.
                let value = unsafe { values.get_unchecked(i as usize) };
                value.map(TotalOrdWrap)
            }));
            self.1.len() as IdxSize
        } else {
            let values = flat_values(values);
            reset_amortized(&mut self.0, length as usize);
            self.0.extend(
                values[start as usize..][..length as usize]
                    .iter()
                    .map(|&v| TotalOrdWrap(v)),
            );
            self.0.len() as IdxSize
        }
    }
}

impl AmortizedUnique for BinaryViewUnique {
    fn new_empty(&self) -> Box<dyn AmortizedUnique> {
        Box::new(BinaryViewUnique::default())
    }

    fn arg_unique(
        &mut self,
        values: &dyn PlArray,
        idxs: &mut UnitVec<IdxSize>,
        start: IdxSize,
        length: IdxSize,
    ) {
        if length <= 1 {
            if length == 1 {
                idxs.push(start);
            }
            return;
        }

        let values = downcast::<PlBinaryViewArray>(values);
        assert!(start.saturating_add(length) as usize <= values.len());

        if values.has_nulls() {
            self.1.reserve(length as usize);
            idxs.extend((start..start + length).filter(|i| {
                // SAFETY: asserted before.
                let value = unsafe { values.get_unchecked(*i as usize) };
                // SAFETY: Gets cleared at end of the scope.
                let value =
                    value.map(|v| unsafe { std::mem::transmute::<&[u8], &'static [u8]>(v) });
                self.1.insert(value)
            }));
            reset_amortized(&mut self.1, length as usize);
        } else {
            self.0.reserve(length as usize);
            if values.total_buffer_len() == 0 {
                let views = flat_views(values);
                idxs.extend(
                    views[start as usize..][..length as usize]
                        .iter()
                        .enumerate()
                        .filter_map(|(i, value)| {
                            debug_assert!(value.is_inline());

                            // SAFETY: buffer length == 0.
                            let value = unsafe { value.get_inlined_slice_unchecked() };
                            // SAFETY: Gets cleared at end of the scope.
                            let value =
                                unsafe { std::mem::transmute::<&[u8], &'static [u8]>(value) };
                            self.0.insert(value).then_some(i as IdxSize + start)
                        }),
                );
            } else {
                idxs.extend((start..start + length).filter(|i| {
                    // SAFETY: asserted before.
                    let value = unsafe { values.value_unchecked(*i as usize) };
                    // SAFETY: Gets cleared at end of the scope.
                    let value = unsafe { std::mem::transmute::<&[u8], &'static [u8]>(value) };
                    self.0.insert(value)
                }));
            }
            reset_amortized(&mut self.0, length as usize);
        }
    }

    unsafe fn retain_unique(&mut self, values: &dyn PlArray, idxs: &mut UnitVec<IdxSize>) {
        if idxs.len() <= 1 {
            return;
        }

        let values = downcast::<PlBinaryViewArray>(values);
        if values.has_nulls() {
            self.1.reserve(idxs.len());
            idxs.retain(|i| {
                // SAFETY: asserted before.
                let value = unsafe { values.get_unchecked(i as usize) };
                // SAFETY: Gets cleared at end of the scope.
                let value =
                    value.map(|v| unsafe { std::mem::transmute::<&[u8], &'static [u8]>(v) });
                self.1.insert(value)
            });
            reset_amortized(&mut self.1, idxs.len());
        } else {
            self.0.reserve(idxs.len());
            if values.total_buffer_len() == 0 {
                let views = flat_views(values);
                idxs.retain(|i| {
                    let value = unsafe { views.get_unchecked(i as usize) };
                    debug_assert!(value.is_inline());

                    // SAFETY: buffer length == 0.
                    let value = unsafe { value.get_inlined_slice_unchecked() };
                    // SAFETY: Gets cleared at end of the scope.
                    let value = unsafe { std::mem::transmute::<&[u8], &'static [u8]>(value) };
                    self.0.insert(value)
                });
            } else {
                idxs.retain(|i| {
                    // SAFETY: asserted before.
                    let value = unsafe { values.value_unchecked(i as usize) };
                    // SAFETY: Gets cleared at end of the scope.
                    let value = unsafe { std::mem::transmute::<&[u8], &'static [u8]>(value) };
                    self.0.insert(value)
                });
            }
            reset_amortized(&mut self.0, idxs.len());
        }
    }

    unsafe fn n_unique_idx(&mut self, values: &dyn PlArray, idxs: &[IdxSize]) -> IdxSize {
        if idxs.len() <= 1 {
            return idxs.len() as IdxSize;
        }

        let values = downcast::<PlBinaryViewArray>(values);

        if values.has_nulls() {
            self.1.reserve(idxs.len());
            self.1.extend(idxs.iter().map(|&i| {
                // SAFETY: function invariant.
                let value = unsafe { values.get_unchecked(i as usize) };
                // SAFETY: Gets cleared at end of the scope.
                value.map(|v| unsafe { std::mem::transmute::<&[u8], &'static [u8]>(v) })
            }));
            let out = self.1.len() as IdxSize;
            reset_amortized(&mut self.1, idxs.len());
            out
        } else {
            self.0.reserve(idxs.len());
            if values.total_buffer_len() == 0 {
                let views = flat_views(values);
                self.0.extend(idxs.iter().map(|&i| {
                    let value = unsafe { views.get_unchecked(i as usize) };
                    debug_assert!(value.is_inline());

                    // SAFETY: buffer length == 0.
                    let value = unsafe { value.get_inlined_slice_unchecked() };
                    // SAFETY: Gets cleared at end of the scope.
                    unsafe { std::mem::transmute::<&[u8], &'static [u8]>(value) }
                }));
            } else {
                self.0.extend(idxs.iter().map(|&i| {
                    // SAFETY: function invariant.
                    let value = unsafe { values.value_unchecked(i as usize) };
                    // SAFETY: Gets cleared at end of the scope.
                    unsafe { std::mem::transmute::<&[u8], &'static [u8]>(value) }
                }));
            }
            let out = self.0.len() as IdxSize;
            reset_amortized(&mut self.0, idxs.len());
            out
        }
    }

    fn n_unique_slice(&mut self, values: &dyn PlArray, start: IdxSize, length: IdxSize) -> IdxSize {
        if length <= 1 {
            return length;
        }

        let values = downcast::<PlBinaryViewArray>(values);
        assert!(start.saturating_add(length) as usize <= values.len());

        if values.has_nulls() {
            self.1.reserve(length as usize);
            self.1.extend((start..start + length).map(|i| {
                // SAFETY: asserted before.
                let value = unsafe { values.get_unchecked(i as usize) };
                // SAFETY: Gets cleared at end of the scope.
                value.map(|v| unsafe { std::mem::transmute::<&[u8], &'static [u8]>(v) })
            }));
            let out = self.1.len() as IdxSize;
            reset_amortized(&mut self.1, length as usize);
            out
        } else {
            self.0.reserve(length as usize);
            if values.total_buffer_len() == 0 {
                let views = flat_views(values);
                self.0.extend(
                    views[start as usize..][..length as usize]
                        .iter()
                        .map(|value| {
                            debug_assert!(value.is_inline());

                            // SAFETY: buffer length == 0.
                            let value = unsafe { value.get_inlined_slice_unchecked() };
                            // SAFETY: Gets cleared at end of the scope.
                            unsafe { std::mem::transmute::<&[u8], &'static [u8]>(value) }
                        }),
                );
            } else {
                self.0.extend((start..start + length).map(|i| {
                    // SAFETY: asserted before.
                    let value = unsafe { values.value_unchecked(i as usize) };
                    // SAFETY: Gets cleared at end of the scope.
                    unsafe { std::mem::transmute::<&[u8], &'static [u8]>(value) }
                }));
            }
            let out = self.0.len() as IdxSize;
            reset_amortized(&mut self.0, length as usize);
            out
        }
    }
}

impl AmortizedUnique for BinaryUnique {
    fn new_empty(&self) -> Box<dyn AmortizedUnique> {
        Box::new(BinaryUnique::default())
    }

    fn arg_unique(
        &mut self,
        values: &dyn PlArray,
        idxs: &mut UnitVec<IdxSize>,
        start: IdxSize,
        length: IdxSize,
    ) {
        if length <= 1 {
            if length == 1 {
                idxs.push(start);
            }
            return;
        }

        let values = downcast::<PlBinaryArray>(values);
        assert!(start.saturating_add(length) as usize <= values.len());

        if values.has_nulls() {
            self.1.reserve(length as usize);
            idxs.extend((start..start + length).filter(|i| {
                // SAFETY: asserted before.
                let value = unsafe { values.get_unchecked(*i as usize) };
                // SAFETY: Gets cleared at end of the scope.
                let value =
                    value.map(|v| unsafe { std::mem::transmute::<&[u8], &'static [u8]>(v) });
                self.1.insert(value)
            }));
            reset_amortized(&mut self.1, length as usize);
        } else {
            self.0.reserve(length as usize);
            idxs.extend((start..start + length).filter(|i| {
                // SAFETY: asserted before.
                let value = unsafe { values.value_unchecked(*i as usize) };
                let value = unsafe { std::mem::transmute::<&[u8], &'static [u8]>(value) };
                self.0.insert(value)
            }));
            reset_amortized(&mut self.0, length as usize);
        }
    }

    unsafe fn retain_unique(&mut self, values: &dyn PlArray, idxs: &mut UnitVec<IdxSize>) {
        if idxs.len() <= 1 {
            return;
        }

        let values = downcast::<PlBinaryArray>(values);

        if values.has_nulls() {
            self.1.reserve(idxs.len());
            idxs.retain(|i| {
                // SAFETY: function invariant.
                let value = unsafe { values.get_unchecked(i as usize) };
                // SAFETY: Gets cleared at end of the scope.
                let value =
                    value.map(|v| unsafe { std::mem::transmute::<&[u8], &'static [u8]>(v) });
                self.1.insert(value)
            });
            reset_amortized(&mut self.1, idxs.len());
        } else {
            self.0.reserve(idxs.len());
            idxs.retain(|i| {
                // SAFETY: function invariant.
                let value = unsafe { values.value_unchecked(i as usize) };
                let value = unsafe { std::mem::transmute::<&[u8], &'static [u8]>(value) };
                self.0.insert(value)
            });
            reset_amortized(&mut self.0, idxs.len());
        }
    }

    unsafe fn n_unique_idx(&mut self, values: &dyn PlArray, idxs: &[IdxSize]) -> IdxSize {
        if idxs.len() <= 1 {
            return idxs.len() as IdxSize;
        }

        let values = downcast::<PlBinaryArray>(values);

        if values.has_nulls() {
            self.1.reserve(idxs.len());
            self.1.extend(idxs.iter().map(|&i| {
                // SAFETY: function invariant.
                let value = unsafe { values.get_unchecked(i as usize) };
                // SAFETY: Gets cleared at end of the scope.
                value.map(|v| unsafe { std::mem::transmute::<&[u8], &'static [u8]>(v) })
            }));
            let out = self.1.len() as IdxSize;
            reset_amortized(&mut self.1, idxs.len());
            out
        } else {
            self.0.reserve(idxs.len());
            self.0.extend(idxs.iter().map(|&i| {
                // SAFETY: function invariant.
                let value = unsafe { values.value_unchecked(i as usize) };
                // SAFETY: Gets cleared at end of the scope.
                unsafe { std::mem::transmute::<&[u8], &'static [u8]>(value) }
            }));
            let out = self.0.len() as IdxSize;
            reset_amortized(&mut self.0, idxs.len());
            out
        }
    }

    fn n_unique_slice(&mut self, values: &dyn PlArray, start: IdxSize, length: IdxSize) -> IdxSize {
        if length <= 1 {
            return length;
        }

        let values = downcast::<PlBinaryArray>(values);
        assert!(start.saturating_add(length) as usize <= values.len());

        if values.has_nulls() {
            self.1.reserve(length as usize);
            self.1.extend((start..start + length).map(|i| {
                // SAFETY: asserted before.
                let value = unsafe { values.get_unchecked(i as usize) };
                // SAFETY: Gets cleared at end of the scope.
                value.map(|v| unsafe { std::mem::transmute::<&[u8], &'static [u8]>(v) })
            }));
            let out = self.1.len() as IdxSize;
            reset_amortized(&mut self.1, length as usize);
            out
        } else {
            self.0.reserve(length as usize);
            self.0.extend((start..start + length).map(|i| {
                // SAFETY: asserted before.
                let value = unsafe { values.value_unchecked(i as usize) };
                // SAFETY: Gets cleared at end of the scope.
                unsafe { std::mem::transmute::<&[u8], &'static [u8]>(value) }
            }));
            let out = self.0.len() as IdxSize;
            reset_amortized(&mut self.0, length as usize);
            out
        }
    }
}
