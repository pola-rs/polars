use arrow::compute::if_then_else::if_then_else;
use polars_arrow::array::default_arrays::FromData;

use crate::prelude::*;
use crate::utils::{align_chunks_ternary, CustomIterTools};

fn ternary_apply<T>(predicate: bool, truthy: T, falsy: T) -> T {
    if predicate {
        truthy
    } else {
        falsy
    }
}

fn prepare_mask(mask: &BooleanArray) -> BooleanArray {
    // make sure that zip works same as master branch
    // that is that null are ignored from mask and that we take from the right array

    match mask.validity() {
        // nulls are set to true meaning we take from the right in the zip/ if_then_else kernel
        Some(validity) if validity.unset_bits() != 0 => {
            let mask = mask.values() & validity;
            BooleanArray::from_data_default(mask, None)
        }
        _ => mask.clone(),
    }
}

macro_rules! impl_ternary_broadcast {
    ($self:ident, $self_len:expr, $other_len:expr, $other:expr, $mask:expr, $ty:ty) => {{
        match ($self_len, $other_len) {
            (1, 1) => {
                let left = $self.get(0);
                let right = $other.get(0);
                let mut val: ChunkedArray<$ty> = $mask
                    .into_no_null_iter()
                    .map(|mask_val| ternary_apply(mask_val, left, right))
                    .collect_trusted();
                val.rename($self.name());
                Ok(val)
            }
            (_, 1) => {
                let right = $other.get(0);
                let mut val: ChunkedArray<$ty> = $mask
                    .into_no_null_iter()
                    .zip($self)
                    .map(|(mask_val, left)| ternary_apply(mask_val, left, right))
                    .collect_trusted();
                val.rename($self.name());
                Ok(val)
            }
            (1, _) => {
                let left = $self.get(0);
                let mut val: ChunkedArray<$ty> = $mask
                    .into_no_null_iter()
                    .zip($other)
                    .map(|(mask_val, right)| ternary_apply(mask_val, left, right))
                    .collect_trusted();
                val.rename($self.name());
                Ok(val)
            }
            (_, _) => Err(PolarsError::ShapeMisMatch(
                "Shape of parameter `mask` and `other` could not be used in zip_with operation"
                    .into(),
            )),
        }
    }};
}

impl<T> ChunkZip<T> for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn zip_with(
        &self,
        mask: &BooleanChunked,
        other: &ChunkedArray<T>,
    ) -> PolarsResult<ChunkedArray<T>> {
        // broadcasting path
        if self.len() != mask.len() || other.len() != mask.len() {
            impl_ternary_broadcast!(self, self.len(), other.len(), other, mask, T)
        } else {
            let (left, right, mask) = align_chunks_ternary(self, other, mask);
            let chunks = left
                .downcast_iter()
                .zip(right.downcast_iter())
                .zip(mask.downcast_iter())
                .map(|((left_c, right_c), mask_c)| {
                    let mask_c = prepare_mask(mask_c);
                    let arr = if_then_else(&mask_c, left_c, right_c)?;
                    Ok(arr)
                })
                .collect::<PolarsResult<Vec<_>>>()?;
            unsafe { Ok(ChunkedArray::from_chunks(self.name(), chunks)) }
        }
    }
}

impl ChunkZip<BooleanType> for BooleanChunked {
    fn zip_with(
        &self,
        mask: &BooleanChunked,
        other: &BooleanChunked,
    ) -> PolarsResult<BooleanChunked> {
        // broadcasting path
        if self.len() != mask.len() || other.len() != mask.len() {
            impl_ternary_broadcast!(self, self.len(), other.len(), other, mask, BooleanType)
        } else {
            let (left, right, mask) = align_chunks_ternary(self, other, mask);
            let chunks = left
                .downcast_iter()
                .zip(right.downcast_iter())
                .zip(mask.downcast_iter())
                .map(|((left_c, right_c), mask_c)| {
                    let mask_c = prepare_mask(mask_c);
                    let arr = if_then_else(&mask_c, left_c, right_c)?;
                    Ok(arr)
                })
                .collect::<PolarsResult<Vec<_>>>()?;
            unsafe { Ok(ChunkedArray::from_chunks(self.name(), chunks)) }
        }
    }
}

impl ChunkZip<Utf8Type> for Utf8Chunked {
    fn zip_with(&self, mask: &BooleanChunked, other: &Utf8Chunked) -> PolarsResult<Utf8Chunked> {
        if self.len() != mask.len() || other.len() != mask.len() {
            impl_ternary_broadcast!(self, self.len(), other.len(), other, mask, Utf8Type)
        } else {
            let (left, right, mask) = align_chunks_ternary(self, other, mask);
            let chunks = left
                .downcast_iter()
                .zip(right.downcast_iter())
                .zip(mask.downcast_iter())
                .map(|((left_c, right_c), mask_c)| {
                    let mask_c = prepare_mask(mask_c);
                    let arr = if_then_else(&mask_c, left_c, right_c)?;
                    Ok(arr)
                })
                .collect::<PolarsResult<Vec<_>>>()?;
            unsafe { Ok(ChunkedArray::from_chunks(self.name(), chunks)) }
        }
    }
}

#[cfg(feature = "dtype-binary")]
impl ChunkZip<BinaryType> for BinaryChunked {
    fn zip_with(
        &self,
        mask: &BooleanChunked,
        other: &BinaryChunked,
    ) -> PolarsResult<BinaryChunked> {
        if self.len() != mask.len() || other.len() != mask.len() {
            impl_ternary_broadcast!(self, self.len(), other.len(), other, mask, BinaryType)
        } else {
            let (left, right, mask) = align_chunks_ternary(self, other, mask);
            let chunks = left
                .downcast_iter()
                .zip(right.downcast_iter())
                .zip(mask.downcast_iter())
                .map(|((left_c, right_c), mask_c)| {
                    let mask_c = prepare_mask(mask_c);
                    let arr = if_then_else(&mask_c, left_c, right_c)?;
                    Ok(arr)
                })
                .collect::<PolarsResult<Vec<_>>>()?;
            unsafe { Ok(ChunkedArray::from_chunks(self.name(), chunks)) }
        }
    }
}

impl ChunkZip<ListType> for ListChunked {
    fn zip_with(
        &self,
        mask: &BooleanChunked,
        other: &ChunkedArray<ListType>,
    ) -> PolarsResult<ChunkedArray<ListType>> {
        let (left, right, mask) = align_chunks_ternary(self, other, mask);
        let chunks = left
            .downcast_iter()
            .zip(right.downcast_iter())
            .zip(mask.downcast_iter())
            .map(|((left_c, right_c), mask_c)| {
                let mask_c = prepare_mask(mask_c);
                let arr = if_then_else(&mask_c, left_c, right_c)?;
                Ok(arr)
            })
            .collect::<PolarsResult<Vec<_>>>()?;
        unsafe { Ok(ChunkedArray::from_chunks(self.name(), chunks)) }
    }
}

#[cfg(feature = "object")]
impl<T: PolarsObject> ChunkZip<ObjectType<T>> for ObjectChunked<T> {
    fn zip_with(
        &self,
        mask: &BooleanChunked,
        other: &ChunkedArray<ObjectType<T>>,
    ) -> PolarsResult<ChunkedArray<ObjectType<T>>> {
        let (left, right, mask) = align_chunks_ternary(self, other, mask);
        let mut ca: Self = left
            .as_ref()
            .into_iter()
            .zip(right.into_iter())
            .zip(mask.into_iter())
            .map(|((left_c, right_c), mask_c)| match mask_c {
                Some(true) => left_c.cloned(),
                Some(false) => right_c.cloned(),
                None => None,
            })
            .collect();
        ca.rename(self.name());
        Ok(ca)
    }
}
