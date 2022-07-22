use polars_arrow::export::arrow::array::PrimitiveArray;
use polars_core::export::arrow::array::Array;
use polars_core::prelude::*;
use polars_core::utils::arrow::bitmap::MutableBitmap;
use polars_core::utils::arrow::types::NativeType;

pub trait ChunkedSet<T: Copy> {
    fn set_at_idx2<V>(self, idx: &[IdxSize], values: V) -> Series
    where
        V: IntoIterator<Item = Option<T>>;
}

fn set_at_idx_impl<V, T: NativeType>(
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
                        mut_validity.set(*idx as usize, true);
                        new_values_slice[*idx as usize] = value
                    }
                    None => mut_validity.set(*idx as usize, false),
                }
            }
            mut_validity.into()
        })
    } else {
        let mut validity = MutableBitmap::default();
        for (idx, val) in idx.iter().zip(values_iter) {
            match val {
                Some(value) => {
                    if validity.is_empty() {
                        validity.extend_constant(len, true);
                    }
                    validity.set(*idx as usize, true);
                    new_values_slice[*idx as usize] = value
                }
                None => {
                    if validity.is_empty() {
                        validity.extend_constant(len, true);
                    }
                    validity.set(*idx as usize, false)
                }
            }
        }
        if !validity.is_empty() {
            arr.set_validity(Some(validity.into()))
        }
    }
}

impl<T: PolarsNumericType> ChunkedSet<T::Native> for ChunkedArray<T>
where
    ChunkedArray<T>: IntoSeries,
{
    fn set_at_idx2<V>(self, idx: &[IdxSize], values: V) -> Series
    where
        V: IntoIterator<Item = Option<T::Native>>,
    {
        let mut ca = self.rechunk();
        drop(self);

        // safety:
        // we will not modify the length
        let arr = unsafe { ca.downcast_iter_mut() }.next().unwrap();
        let len = arr.len();

        match arr.get_mut_values() {
            Some(current_values) => {
                let ptr = current_values.as_mut_ptr();

                // reborrow because the bck does not allow it
                let current_values = unsafe { &mut *std::slice::from_raw_parts_mut(ptr, len) };
                set_at_idx_impl(current_values, values, arr, idx, len)
            }
            None => {
                let mut new_values = arr.values().as_slice().to_vec();
                set_at_idx_impl(&mut new_values, values, arr, idx, len);
                arr.set_values(new_values.into());
            }
        };
        ca.into_series()
    }
}
