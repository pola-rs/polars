/// Implementations for {n,arg}-unique on [`Array`] that can be amortized over several invocations.
use arrow::array::{Array, BinaryViewArray, BooleanArray, PrimitiveArray, StaticArray};
use arrow::bitmap::bitmask::BitMask;
use arrow::datatypes::ArrowDataType;
use arrow::legacy::prelude::LargeBinaryArray;
use arrow::types::{NativeType, PrimitiveType};
use polars_utils::aliases::PlHashSet;
use polars_utils::float16::pf16;
use polars_utils::total_ord::{TotalEq, TotalHash, TotalOrdWrap};
use polars_utils::{IdxSize, UnitVec};

pub trait AmortizedUnique: Send + Sync + 'static {
    fn new_empty(&self) -> Box<dyn AmortizedUnique>;

    /// Retain indices of items that are unique.
    ///
    /// This is always stable.
    ///
    /// # Safety
    ///
    /// All indices i should be 0 <= i < values.len()
    unsafe fn retain_unique(&mut self, values: &dyn Array, idxs: &mut UnitVec<IdxSize>);

    /// Get the indices of unique items in an array slice.
    ///
    /// This is always stable.
    fn arg_unique(
        &mut self,
        values: &dyn Array,
        idxs: &mut UnitVec<IdxSize>,
        start: IdxSize,
        length: IdxSize,
    );

    /// Get the number of unique items in an array at `idxs`.
    ///
    /// # Safety
    ///
    /// All indices i should be 0 <= i < values.len()
    unsafe fn n_unique_idx(&mut self, values: &dyn Array, idxs: &[IdxSize]) -> IdxSize;

    /// Get the number of unique items in an array slice.
    fn n_unique_slice(&mut self, values: &dyn Array, start: IdxSize, length: IdxSize) -> IdxSize;
}

pub fn amortized_unique_from_dtype(dtype: &ArrowDataType) -> Box<dyn AmortizedUnique> {
    use arrow::datatypes::PhysicalType as P;
    match dtype.to_physical_type() {
        P::Null => Box::new(NullUnique) as _,
        P::Boolean => Box::new(BooleanUnique) as _,
        P::Primitive(pt) => match pt {
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
        P::BinaryView => Box::new(BinaryViewUnique::default()) as _,
        P::LargeBinary => Box::new(BinaryUnique::default()) as _,

        P::Dictionary(_) => unreachable!(),
        P::Binary => unreachable!(),
        P::FixedSizeBinary => unreachable!(),
        P::Utf8 => unreachable!(),
        P::LargeUtf8 => unreachable!(),
        P::List => unreachable!(),
        P::Union => unreachable!(),
        P::Map => unreachable!(),

        // Should be handled through BinaryView.
        P::Utf8View => unreachable!(),

        // Should be handled through row encoding.
        P::FixedSizeList => unreachable!(),
        P::LargeList => unreachable!(),
        P::Struct => unreachable!(),
    }
}

struct NullUnique;
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

impl AmortizedUnique for NullUnique {
    fn new_empty(&self) -> Box<dyn AmortizedUnique> {
        Box::new(NullUnique)
    }

    unsafe fn retain_unique(&mut self, _values: &dyn Array, idxs: &mut UnitVec<IdxSize>) {
        if !idxs.is_empty() {
            *idxs = UnitVec::from_slice(&[idxs[0]]);
        }
    }

    fn arg_unique(
        &mut self,
        values: &dyn Array,
        idxs: &mut UnitVec<IdxSize>,
        start: IdxSize,
        length: IdxSize,
    ) {
        assert!(start.saturating_add(length) as usize <= values.len());
        if length > 0 {
            idxs.push(start);
        }
    }

    unsafe fn n_unique_idx(&mut self, _values: &dyn Array, idxs: &[IdxSize]) -> IdxSize {
        IdxSize::from(!idxs.is_empty())
    }

    fn n_unique_slice(&mut self, values: &dyn Array, start: IdxSize, length: IdxSize) -> IdxSize {
        assert!(start.saturating_add(length) as usize <= values.len());
        IdxSize::from(length > 0)
    }
}

impl AmortizedUnique for BooleanUnique {
    fn new_empty(&self) -> Box<dyn AmortizedUnique> {
        Box::new(BooleanUnique)
    }

    unsafe fn retain_unique(&mut self, values: &dyn Array, idxs: &mut UnitVec<IdxSize>) {
        if idxs.len() <= 1 {
            return;
        }

        let values = values.as_any().downcast_ref::<BooleanArray>().unwrap();

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
            let values = values.values();
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
        values: &dyn Array,
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
        let values = values.as_any().downcast_ref::<BooleanArray>().unwrap();

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
            let values = values.values();
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

    unsafe fn n_unique_idx(&mut self, values: &dyn Array, idxs: &[IdxSize]) -> IdxSize {
        if idxs.len() <= 1 {
            return idxs.len() as IdxSize;
        }

        let values = values.as_any().downcast_ref::<BooleanArray>().unwrap();

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
            let values = values.values();
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

    fn n_unique_slice(&mut self, values: &dyn Array, start: IdxSize, length: IdxSize) -> IdxSize {
        if length <= 1 {
            return length;
        }

        let values = values.as_any().downcast_ref::<BooleanArray>().unwrap();
        assert!(start.saturating_add(length) as usize <= values.len());

        if values.has_nulls() {
            let validity = BitMask::from_bitmap(values.validity().unwrap());
            let values = BitMask::from_bitmap(values.values());

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
            let values = values.values();
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

    unsafe fn retain_unique(&mut self, values: &dyn Array, idxs: &mut UnitVec<IdxSize>) {
        if idxs.len() <= 1 {
            return;
        }

        let values = values.as_any().downcast_ref::<PrimitiveArray<T>>().unwrap();

        if values.has_nulls() {
            self.1.clear();
            idxs.retain(|i| {
                // SAFETY: function invariant.
                let value = unsafe { values.get_unchecked(i as usize) };
                let value = value.map(TotalOrdWrap);
                self.1.insert(value)
            });
        } else {
            self.0.clear();
            let values = values.values().as_slice();
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
        values: &dyn Array,
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

        let values = values.as_any().downcast_ref::<PrimitiveArray<T>>().unwrap();
        assert!(start.saturating_add(length) as usize <= values.len());

        if values.has_nulls() {
            self.1.clear();
            idxs.extend((start..start + length).filter(|i| {
                // SAFETY: asserted before.
                let value = unsafe { values.get_unchecked(*i as usize) };
                let value = value.map(TotalOrdWrap);
                self.1.insert(value)
            }));
        } else {
            self.0.clear();
            let values = values.values().as_slice();
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

    unsafe fn n_unique_idx(&mut self, values: &dyn Array, idxs: &[IdxSize]) -> IdxSize {
        if idxs.len() <= 1 {
            return idxs.len() as IdxSize;
        }

        let values = values.as_any().downcast_ref::<PrimitiveArray<T>>().unwrap();

        if values.has_nulls() {
            self.1.clear();
            self.1.extend(idxs.iter().map(|&i| {
                // SAFETY: function invariant.
                let value = unsafe { values.get_unchecked(i as usize) };
                value.map(TotalOrdWrap)
            }));
            self.1.len() as IdxSize
        } else {
            let values = values.values();
            self.0.clear();
            self.0.extend(idxs.iter().map(|&i| {
                // SAFETY: function invariant.
                let value = *unsafe { values.get_unchecked(i as usize) };
                TotalOrdWrap(value)
            }));
            self.0.len() as IdxSize
        }
    }

    fn n_unique_slice(&mut self, values: &dyn Array, start: IdxSize, length: IdxSize) -> IdxSize {
        if length <= 1 {
            return length;
        }

        let values = values.as_any().downcast_ref::<PrimitiveArray<T>>().unwrap();
        assert!(start.saturating_add(length) as usize <= values.len());

        if values.has_nulls() {
            self.1.clear();
            self.1.extend((start..start + length).map(|i| {
                // SAFETY: asserted before.
                let value = unsafe { values.get_unchecked(i as usize) };
                value.map(TotalOrdWrap)
            }));
            self.1.len() as IdxSize
        } else {
            let values = values.values();
            self.0.clear();
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
        values: &dyn Array,
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

        let values = values.as_any().downcast_ref::<BinaryViewArray>().unwrap();
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
            self.1.clear();
        } else {
            self.0.reserve(length as usize);
            if values.total_buffer_len() == 0 {
                let views = values.views().as_slice();
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
            self.0.clear();
        }
    }

    unsafe fn retain_unique(&mut self, values: &dyn Array, idxs: &mut UnitVec<IdxSize>) {
        if idxs.len() <= 1 {
            return;
        }

        let values = values.as_any().downcast_ref::<BinaryViewArray>().unwrap();
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
            self.1.clear();
        } else {
            self.0.reserve(idxs.len());
            if values.total_buffer_len() == 0 {
                let views = values.views().as_slice();
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
            self.0.clear();
        }
    }

    unsafe fn n_unique_idx(&mut self, values: &dyn Array, idxs: &[IdxSize]) -> IdxSize {
        if idxs.len() <= 1 {
            return idxs.len() as IdxSize;
        }

        let values = values.as_any().downcast_ref::<BinaryViewArray>().unwrap();

        if values.has_nulls() {
            self.1.reserve(idxs.len());
            self.1.extend(idxs.iter().map(|&i| {
                // SAFETY: function invariant.
                let value = unsafe { values.get_unchecked(i as usize) };
                // SAFETY: Gets cleared at end of the scope.
                value.map(|v| unsafe { std::mem::transmute::<&[u8], &'static [u8]>(v) })
            }));
            let out = self.1.len() as IdxSize;
            self.1.clear();
            out
        } else {
            self.0.reserve(idxs.len());
            if values.total_buffer_len() == 0 {
                let views = values.views().as_slice();
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
            self.0.clear();
            out
        }
    }

    fn n_unique_slice(&mut self, values: &dyn Array, start: IdxSize, length: IdxSize) -> IdxSize {
        if length <= 1 {
            return length;
        }

        let values = values.as_any().downcast_ref::<BinaryViewArray>().unwrap();
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
            self.1.clear();
            out
        } else {
            self.0.reserve(length as usize);
            if values.total_buffer_len() == 0 {
                let views = values.views().as_slice();
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
            self.0.clear();
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
        values: &dyn Array,
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

        let values = values.as_any().downcast_ref::<LargeBinaryArray>().unwrap();
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
            self.1.clear();
        } else {
            self.0.reserve(length as usize);
            idxs.extend((start..start + length).filter(|i| {
                // SAFETY: asserted before.
                let value = unsafe { values.value_unchecked(*i as usize) };
                let value = unsafe { std::mem::transmute::<&[u8], &'static [u8]>(value) };
                self.0.insert(value)
            }));
            self.0.clear();
        }
    }

    unsafe fn retain_unique(&mut self, values: &dyn Array, idxs: &mut UnitVec<IdxSize>) {
        if idxs.len() <= 1 {
            return;
        }

        let values = values.as_any().downcast_ref::<LargeBinaryArray>().unwrap();

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
            self.1.clear();
        } else {
            self.0.reserve(idxs.len());
            idxs.retain(|i| {
                // SAFETY: function invariant.
                let value = unsafe { values.value_unchecked(i as usize) };
                let value = unsafe { std::mem::transmute::<&[u8], &'static [u8]>(value) };
                self.0.insert(value)
            });
            self.0.clear();
        }
    }

    unsafe fn n_unique_idx(&mut self, values: &dyn Array, idxs: &[IdxSize]) -> IdxSize {
        if idxs.len() <= 1 {
            return idxs.len() as IdxSize;
        }

        let values = values.as_any().downcast_ref::<LargeBinaryArray>().unwrap();

        if values.has_nulls() {
            self.1.reserve(idxs.len());
            self.1.extend(idxs.iter().map(|&i| {
                // SAFETY: function invariant.
                let value = unsafe { values.get_unchecked(i as usize) };
                // SAFETY: Gets cleared at end of the scope.
                value.map(|v| unsafe { std::mem::transmute::<&[u8], &'static [u8]>(v) })
            }));
            let out = self.1.len() as IdxSize;
            self.1.clear();
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
            self.0.clear();
            out
        }
    }

    fn n_unique_slice(&mut self, values: &dyn Array, start: IdxSize, length: IdxSize) -> IdxSize {
        if length <= 1 {
            return length;
        }

        let values = values.as_any().downcast_ref::<LargeBinaryArray>().unwrap();
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
            self.1.clear();
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
            self.0.clear();
            out
        }
    }
}
