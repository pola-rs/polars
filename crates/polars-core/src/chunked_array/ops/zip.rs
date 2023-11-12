use arrow::compute::if_then_else::if_then_else;
use arrow::legacy::array::default_arrays::FromData;

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
    // make sure that zip works same as main branch
    // that is that null are ignored from mask and that we take from the right array

    match mask.validity() {
        // nulls are set to true meaning we take from the right in the zip/ if_then_else kernel
        Some(validity) if validity.unset_bits() != 0 => {
            let mask = mask.values() & validity;
            BooleanArray::from_data_default(mask, None)
        },
        _ => mask.clone(),
    }
}

macro_rules! impl_ternary_broadcast {
    ($self:ident, $self_len:expr, $other_len:expr, $mask_len: expr, $other:expr, $mask:expr, $ty:ty) => {{
        match ($self_len, $other_len, $mask_len) {
        (1, 1, _) => {
            let left = $self.get(0);
            let right = $other.get(0);
            let mut val: ChunkedArray<$ty> = $mask
                .into_no_null_iter()
                .map(|mask_val| ternary_apply(mask_val, left, right))
                .collect_trusted();
            val.rename($self.name());
            Ok(val)
        }
        (_, 1, 1) => {
            let right = $other.get(0);
            let mask = $mask.get(0).unwrap_or(false);
            let mut val: ChunkedArray<$ty> = $self
                .into_iter()
                .map(|left| ternary_apply(mask, left, right))
                .collect_trusted();
            val.rename($self.name());
            Ok(val)
        }
        (1, _, 1) => {
            let left = $self.get(0);
            let mask = $mask.get(0).unwrap_or(false);
            let mut val: ChunkedArray<$ty> = $other
                .into_iter()
                .map(|right| ternary_apply(mask, left, right))
                .collect_trusted();
            val.rename($self.name());
            Ok(val)
        },
        (1, r_len, mask_len) if r_len == mask_len =>{
            let left = $self.get(0);
            let mut val: ChunkedArray<$ty> = $mask
                .into_no_null_iter()
                .zip($other)
                .map(|(mask, right)| ternary_apply(mask, left, right))
                .collect_trusted();
            val.rename($self.name());
            Ok(val)
        },
        (l_len, 1, mask_len) if l_len == mask_len => {
            let right = $other.get(0);
            let mut val: ChunkedArray<$ty> = $mask
                .into_no_null_iter()
                .zip($self)
                .map(|(mask, left)| ternary_apply(mask, left, right))
                .collect_trusted();
            val.rename($self.name());
            Ok(val)
        },
        (l_len, r_len, 1) if l_len == r_len => {
            let mut val = if let Some(true) = $mask.get(0) {
                $self.clone()
            } else {
                $other.clone()
            };

            val.rename($self.name());
            Ok(val)
        },
        (_, _, 0) => {
            Ok($self.clear())
        }
        (_, _, _) => Err(polars_err!(
                ShapeMismatch: "shapes of `self`, `mask` and `other` are not suitable for `zip_with` operation"
            )),
    }
    }};
}

fn zip_with<T: PolarsDataType>(
    left: &ChunkedArray<T>,
    right: &ChunkedArray<T>,
    mask: &BooleanChunked,
) -> PolarsResult<ChunkedArray<T>> {
    if left.len() != right.len() || right.len() != mask.len() {
        return Err(polars_err!(
            ShapeMismatch: "shapes of `left`, `right` and `mask` are not suitable for `zip_with` operation"
        ));
    };

    let (left, right, mask) = align_chunks_ternary(left, right, mask);
    let chunks = left
        .chunks()
        .iter()
        .zip(right.chunks())
        .zip(mask.downcast_iter())
        .map(|((left_c, right_c), mask_c)| {
            let mask_c = prepare_mask(mask_c);
            let arr = if_then_else(&mask_c, left_c.as_ref(), right_c.as_ref())?;
            Ok(arr)
        })
        .collect::<PolarsResult<Vec<_>>>()?;
    unsafe { Ok(left.copy_with_chunks(chunks, false, false)) }
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
            impl_ternary_broadcast!(self, self.len(), other.len(), mask.len(), other, mask, T)
        } else {
            zip_with(self, other, mask)
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
            impl_ternary_broadcast!(
                self,
                self.len(),
                other.len(),
                mask.len(),
                other,
                mask,
                BooleanType
            )
        } else {
            zip_with(self, other, mask)
        }
    }
}

impl ChunkZip<Utf8Type> for Utf8Chunked {
    fn zip_with(&self, mask: &BooleanChunked, other: &Utf8Chunked) -> PolarsResult<Utf8Chunked> {
        unsafe {
            self.as_binary()
                .zip_with(mask, &other.as_binary())
                .map(|ca| ca.to_utf8())
        }
    }
}

impl ChunkZip<BinaryType> for BinaryChunked {
    fn zip_with(
        &self,
        mask: &BooleanChunked,
        other: &BinaryChunked,
    ) -> PolarsResult<BinaryChunked> {
        if self.len() != mask.len() || other.len() != mask.len() {
            impl_ternary_broadcast!(
                self,
                self.len(),
                other.len(),
                mask.len(),
                other,
                mask,
                BinaryType
            )
        } else {
            zip_with(self, other, mask)
        }
    }
}

impl ChunkZip<ListType> for ListChunked {
    fn zip_with(&self, mask: &BooleanChunked, other: &ListChunked) -> PolarsResult<ListChunked> {
        zip_with(self, other, mask)
    }
}

#[cfg(feature = "dtype-array")]
impl ChunkZip<FixedSizeListType> for ArrayChunked {
    fn zip_with(&self, mask: &BooleanChunked, other: &ArrayChunked) -> PolarsResult<ArrayChunked> {
        zip_with(self, other, mask)
    }
}

#[cfg(feature = "object")]
impl<T: PolarsObject> ChunkZip<ObjectType<T>> for ObjectChunked<T> {
    fn zip_with(
        &self,
        mask: &BooleanChunked,
        other: &ChunkedArray<ObjectType<T>>,
    ) -> PolarsResult<ChunkedArray<ObjectType<T>>> {
        if self.len() != mask.len() || mask.len() != other.len() {
            return Err(polars_err!(
                ShapeMismatch: "shapes of `self`, `mask` and `other` are not suitable for `zip_with` operation"
            ));
        };

        let (left, right, mask) = align_chunks_ternary(self, other, mask);
        let mut ca: Self = left
            .as_ref()
            .into_iter()
            .zip(right.as_ref())
            .zip(mask.as_ref())
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
