use crate::chunked_array::kernels;
use crate::prelude::*;
use crate::utils::Xob;
use arrow::array::ArrayRef;
use std::sync::Arc;

fn ternary_apply<T>(predicate: bool, truthy: T, falsy: T) -> T {
    if predicate {
        truthy
    } else {
        falsy
    }
}

fn ternary_apply_opt<T>(
    opt_predicate: Option<bool>,
    truthy: Option<T>,
    falsy: Option<T>,
) -> Option<T> {
    match opt_predicate {
        None => None,
        Some(predicate) => ternary_apply(predicate, truthy, falsy),
    }
}

macro_rules! impl_ternary {
    ($mask:expr, $self:expr, $other:expr, $ty:ty) => {{
        if $mask.null_count() > 0 {
            let mut val: ChunkedArray<$ty> = $mask
                .into_iter()
                .zip($self)
                .zip($other)
                .map(|((a, b), c)| ternary_apply_opt(a, b, c))
                .collect();
            val.rename($self.name());
            Ok(val)
        } else {
            let mut val: ChunkedArray<$ty> = $mask
                .into_no_null_iter()
                .zip($self)
                .zip($other)
                .map(|((a, b), c)| ternary_apply(a, b, c))
                .collect();
            val.rename($self.name());
            Ok(val)
        }
    }};
}
macro_rules! impl_ternary_broadcast {
    ($self:ident, $self_len:ident, $other_len:expr, $other:expr, $mask:expr, $ty:ty) => {{
        match ($self_len, $other_len) {
            (1, 1) => {
                let left = $self.get(0);
                let right = $other.get(0);
                let mut val: ChunkedArray<$ty> = $mask
                    .into_no_null_iter()
                    .map(|mask_val| ternary_apply(mask_val, left, right))
                    .collect();
                val.rename($self.name());
                Ok(val)
            }
            (_, 1) => {
                let right = $other.get(0);
                let mut val: ChunkedArray<$ty> = $mask
                    .into_no_null_iter()
                    .zip($self)
                    .map(|(mask_val, left)| ternary_apply(mask_val, left, right))
                    .collect();
                val.rename($self.name());
                Ok(val)
            }
            (1, _) => {
                let left = $self.get(0);
                let mut val: ChunkedArray<$ty> = $mask
                    .into_no_null_iter()
                    .zip($other)
                    .map(|(mask_val, right)| ternary_apply(mask_val, left, right))
                    .collect();
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
    fn zip_with(&self, mask: &BooleanChunked, other: &ChunkedArray<T>) -> Result<ChunkedArray<T>> {
        let self_len = self.len();
        let other_len = other.len();
        let mask_len = mask.len();

        // broadcasting path
        if self_len != mask_len || other_len != mask_len {
            impl_ternary_broadcast!(self, self_len, other_len, other, mask, T)

        // cache optimal path
        } else if self.chunk_id == other.chunk_id && other.chunk_id == mask.chunk_id {
            let chunks = self
                .downcast_chunks()
                .iter()
                .zip(&other.downcast_chunks())
                .zip(&mask.downcast_chunks())
                .map(|((left_c, right_c), mask_c)| {
                    kernels::zip(mask_c, left_c, right_c).map(|arr| Arc::new(arr) as ArrayRef)
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(ChunkedArray::new_from_chunks(self.name(), chunks))
        // no null path
        } else if self.null_count() == 0 && other.null_count() == 0 {
            let val: Xob<ChunkedArray<_>> = mask
                .into_no_null_iter()
                .zip(self.into_no_null_iter())
                .zip(other.into_no_null_iter())
                .map(|((a, b), c)| ternary_apply(a, b, c))
                .collect();
            let mut ca = val.into_inner();
            ca.rename(self.name());
            Ok(ca)
        // slowest path
        } else {
            impl_ternary!(mask, self, other, T)
        }
    }

    fn zip_with_series(&self, mask: &BooleanChunked, other: &Series) -> Result<ChunkedArray<T>> {
        let other = self.unpack_series_matching_type(other)?;
        self.zip_with(mask, other)
    }
}

impl ChunkZip<BooleanType> for BooleanChunked {
    fn zip_with(&self, mask: &BooleanChunked, other: &BooleanChunked) -> Result<BooleanChunked> {
        impl_ternary!(mask, self, other, BooleanType)
    }

    fn zip_with_series(
        &self,
        mask: &BooleanChunked,
        other: &Series,
    ) -> Result<ChunkedArray<BooleanType>> {
        let other = self.unpack_series_matching_type(other)?;
        self.zip_with(mask, other)
    }
}

impl ChunkZip<Utf8Type> for Utf8Chunked {
    fn zip_with(&self, mask: &BooleanChunked, other: &Utf8Chunked) -> Result<Utf8Chunked> {
        let self_len = self.len();
        let other_len = other.len();
        let mask_len = mask.len();

        if self_len != mask_len || other_len != mask_len {
            impl_ternary_broadcast!(self, self_len, other_len, other, mask, Utf8Type)
        } else {
            impl_ternary!(mask, self, other, Utf8Type)
        }
    }

    fn zip_with_series(
        &self,
        mask: &BooleanChunked,
        other: &Series,
    ) -> Result<ChunkedArray<Utf8Type>> {
        let other = self.unpack_series_matching_type(other)?;
        self.zip_with(mask, other)
    }
}
impl ChunkZip<ListType> for ListChunked {
    fn zip_with(
        &self,
        _mask: &BooleanChunked,
        _other: &ChunkedArray<ListType>,
    ) -> Result<ChunkedArray<ListType>> {
        Err(PolarsError::InvalidOperation(
            "zip_with method not supported for ChunkedArray of type List".into(),
        ))
    }

    fn zip_with_series(
        &self,
        _mask: &BooleanChunked,
        _other: &Series,
    ) -> Result<ChunkedArray<ListType>> {
        Err(PolarsError::InvalidOperation(
            "zip_with_series method not supported for ChunkedArray of type List".into(),
        ))
    }
}

#[cfg(feature = "object")]
impl<T> ChunkZip<ObjectType<T>> for ObjectChunked<T> {
    fn zip_with(
        &self,
        _mask: &BooleanChunked,
        _other: &ChunkedArray<ObjectType<T>>,
    ) -> Result<ChunkedArray<ObjectType<T>>> {
        Err(PolarsError::InvalidOperation(
            "zip_with method not supported for ChunkedArray of type Object".into(),
        ))
    }

    fn zip_with_series(
        &self,
        _mask: &BooleanChunked,
        _other: &Series,
    ) -> Result<ChunkedArray<ObjectType<T>>> {
        Err(PolarsError::InvalidOperation(
            "zip_with_series method not supported for ChunkedArray of type Object".into(),
        ))
    }
}
