use arrow::array::{Array, PrimitiveArray};
use polars_core::prelude::*;
use polars_core::series::IsSorted;
use polars_core::utils::arrow::bitmap::MutableBitmap;
use polars_core::utils::arrow::types::NativeType;
use polars_utils::index::check_bounds;

pub trait ChunkedSet<T: Copy> {
    /// Invariant for implementations: if the scatter() fails, typically because
    /// of bad indexes, then self should remain unmodified.
    fn scatter<V>(self, idx: &[IdxSize], values: V) -> PolarsResult<Series>
    where
        V: IntoIterator<Item = Option<T>>;
}
fn check_sorted(idx: &[IdxSize]) -> PolarsResult<()> {
    if idx.is_empty() {
        return Ok(());
    }
    let mut sorted = true;
    let mut previous = idx[0];
    for &i in &idx[1..] {
        if i < previous {
            // we will not break here as that prevents SIMD
            sorted = false;
        }
        previous = i;
    }
    polars_ensure!(sorted, ComputeError: "set indices must be sorted");
    Ok(())
}

trait PolarsOpsNumericType: PolarsNumericType {}

impl PolarsOpsNumericType for UInt8Type {}
impl PolarsOpsNumericType for UInt16Type {}
impl PolarsOpsNumericType for UInt32Type {}
impl PolarsOpsNumericType for UInt64Type {}
impl PolarsOpsNumericType for Int8Type {}
impl PolarsOpsNumericType for Int16Type {}
impl PolarsOpsNumericType for Int32Type {}
impl PolarsOpsNumericType for Int64Type {}
impl PolarsOpsNumericType for Float32Type {}
impl PolarsOpsNumericType for Float64Type {}

unsafe fn scatter_impl<V, T: NativeType>(
    new_values_slice: &mut [T],
    set_values: V,
    arr: &mut PrimitiveArray<T>,
    idx: &[IdxSize],
    len: usize,
) where
    V: IntoIterator<Item = Option<T>>,
{
    let mut values_iter = set_values.into_iter();

    if arr.null_count() > 0 {
        arr.apply_validity(|v| {
            let mut mut_validity = v.make_mut();

            for (idx, val) in idx.iter().zip(&mut values_iter) {
                match val {
                    Some(value) => {
                        mut_validity.set_unchecked(*idx as usize, true);
                        *new_values_slice.get_unchecked_mut(*idx as usize) = value
                    },
                    None => mut_validity.set_unchecked(*idx as usize, false),
                }
            }
            mut_validity.into()
        })
    } else {
        let mut null_idx = vec![];
        for (idx, val) in idx.iter().zip(values_iter) {
            match val {
                Some(value) => *new_values_slice.get_unchecked_mut(*idx as usize) = value,
                None => {
                    null_idx.push(*idx);
                },
            }
        }
        // only make a validity bitmap when null values are set
        if !null_idx.is_empty() {
            let mut validity = MutableBitmap::with_capacity(len);
            validity.extend_constant(len, true);
            for idx in null_idx {
                validity.set_unchecked(idx as usize, false)
            }
            arr.set_validity(Some(validity.into()))
        }
    }
}

impl<T: PolarsOpsNumericType> ChunkedSet<T::Native> for &mut ChunkedArray<T>
where
    ChunkedArray<T>: IntoSeries,
{
    fn scatter<V>(self, idx: &[IdxSize], values: V) -> PolarsResult<Series>
    where
        V: IntoIterator<Item = Option<T::Native>>,
    {
        check_bounds(idx, self.len() as IdxSize)?;
        let mut ca = std::mem::take(self).rechunk();

        // SAFETY:
        // we will not modify the length
        // and we unset the sorted flag.
        ca.set_sorted_flag(IsSorted::Not);
        let arr = unsafe { ca.downcast_iter_mut() }.next().unwrap();
        let len = arr.len();

        match arr.get_mut_values() {
            Some(current_values) => {
                let ptr = current_values.as_mut_ptr();

                // reborrow because the bck does not allow it
                let current_values = unsafe { &mut *std::slice::from_raw_parts_mut(ptr, len) };
                // SAFETY:
                // we checked bounds
                unsafe { scatter_impl(current_values, values, arr, idx, len) };
            },
            None => {
                let mut new_values = arr.values().as_slice().to_vec();
                // SAFETY:
                // we checked bounds
                unsafe { scatter_impl(&mut new_values, values, arr, idx, len) };
                arr.set_values(new_values.into());
            },
        };

        // The null count may have changed - make sure to update the ChunkedArray
        let new_null_count = arr.null_count();
        unsafe { ca.set_null_count(new_null_count.try_into().unwrap()) };

        Ok(ca.into_series())
    }
}

impl<'a> ChunkedSet<&'a str> for &'a StringChunked {
    fn scatter<V>(self, idx: &[IdxSize], values: V) -> PolarsResult<Series>
    where
        V: IntoIterator<Item = Option<&'a str>>,
    {
        check_bounds(idx, self.len() as IdxSize)?;
        check_sorted(idx)?;
        let mut ca_iter = self.into_iter().enumerate();
        let mut builder = StringChunkedBuilder::new(self.name(), self.len());

        for (current_idx, current_value) in idx.iter().zip(values) {
            for (cnt_idx, opt_val_self) in &mut ca_iter {
                if cnt_idx == *current_idx as usize {
                    builder.append_option(current_value);
                    break;
                } else {
                    builder.append_option(opt_val_self);
                }
            }
        }
        // the last idx is probably not the last value so we finish the iterator
        for (_, opt_val_self) in ca_iter {
            builder.append_option(opt_val_self);
        }

        let ca = builder.finish();
        Ok(ca.into_series())
    }
}
impl ChunkedSet<bool> for &BooleanChunked {
    fn scatter<V>(self, idx: &[IdxSize], values: V) -> PolarsResult<Series>
    where
        V: IntoIterator<Item = Option<bool>>,
    {
        check_bounds(idx, self.len() as IdxSize)?;
        check_sorted(idx)?;
        let mut ca_iter = self.into_iter().enumerate();
        let mut builder = BooleanChunkedBuilder::new(self.name(), self.len());

        for (current_idx, current_value) in idx.iter().zip(values) {
            for (cnt_idx, opt_val_self) in &mut ca_iter {
                if cnt_idx == *current_idx as usize {
                    builder.append_option(current_value);
                    break;
                } else {
                    builder.append_option(opt_val_self);
                }
            }
        }
        // the last idx is probably not the last value so we finish the iterator
        for (_, opt_val_self) in ca_iter {
            builder.append_option(opt_val_self);
        }

        let ca = builder.finish();
        Ok(ca.into_series())
    }
}
