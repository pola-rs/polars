use crate::prelude::*;
use crate::utils::align_chunks_binary;
use arrow::array::ArrayRef;
use polars_arrow::array::ValueSize;
use polars_arrow::kernels::set::{set_at_idx_no_null, set_with_mask};
use std::sync::Arc;

macro_rules! impl_set_at_idx_with {
    ($self:ident, $builder:ident, $idx:ident, $f:ident) => {{
        let mut idx_iter = $idx.into_iter();
        let mut ca_iter = $self.into_iter().enumerate();

        while let Some(current_idx) = idx_iter.next() {
            if current_idx > $self.len() {
                return Err(PolarsError::OutOfBounds(
                    format!(
                        "index: {} outside of ChunkedArray with length: {}",
                        current_idx,
                        $self.len()
                    )
                    .into(),
                ));
            }
            while let Some((cnt_idx, opt_val)) = ca_iter.next() {
                if cnt_idx == current_idx {
                    $builder.append_option($f(opt_val));
                    break;
                } else {
                    $builder.append_option(opt_val);
                }
            }
        }
        // the last idx is probably not the last value so we finish the iterator
        while let Some((_, opt_val)) = ca_iter.next() {
            $builder.append_option(opt_val);
        }

        let ca = $builder.finish();
        Ok(ca)
    }};
}

macro_rules! check_bounds {
    ($self:ident, $mask:ident) => {{
        if $self.len() != $mask.len() {
            return Err(PolarsError::ShapeMisMatch(
                "Shape of parameter `mask` could not be used in `set` operation.".into(),
            ));
        }
    }};
}

impl<'a, T> ChunkSet<'a, T::Native, T::Native> for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn set_at_idx<I: IntoIterator<Item = usize>>(
        &'a self,
        idx: I,
        value: Option<T::Native>,
    ) -> Result<Self> {
        if self.null_count() == 0 {
            if let Some(value) = value {
                // fast path uses kernel
                if self.chunks.len() == 1 {
                    let arr = set_at_idx_no_null(
                        self.downcast_iter().next().unwrap(),
                        idx.into_iter(),
                        value,
                        T::get_dtype().to_arrow(),
                    )?;
                    return Ok(Self::new_from_chunks(self.name(), vec![Arc::new(arr)]));
                }
                // Other fast path. Slightly slower as it does not do a memcpy
                else {
                    let mut av = self.into_no_null_iter().collect::<AlignedVec<_>>();
                    let data = av.as_mut_slice();

                    idx.into_iter().try_for_each::<_, Result<_>>(|idx| {
                        let val = data.get_mut(idx).ok_or_else(|| {
                            PolarsError::OutOfBounds(
                                format!("{} out of bounds on array of length: {}", idx, self.len())
                                    .into(),
                            )
                        })?;
                        *val = value;
                        Ok(())
                    })?;
                    return Ok(Self::new_from_aligned_vec(self.name(), av));
                }
            }
        }
        self.set_at_idx_with(idx, |_| value)
    }

    fn set_at_idx_with<I: IntoIterator<Item = usize>, F>(&'a self, idx: I, f: F) -> Result<Self>
    where
        F: Fn(Option<T::Native>) -> Option<T::Native>,
    {
        let mut builder = PrimitiveChunkedBuilder::<T>::new(self.name(), self.len());
        impl_set_at_idx_with!(self, builder, idx, f)
    }

    fn set(&'a self, mask: &BooleanChunked, value: Option<T::Native>) -> Result<Self> {
        check_bounds!(self, mask);

        // Fast path uses the kernel in polars-arrow
        if let Some(value) = value {
            // kernel expects no null values in mask
            if mask.null_count() == 0 {
                let (left, mask) = align_chunks_binary(self, mask);

                // apply binary kernel.
                let chunks = left
                    .downcast_iter()
                    .into_iter()
                    .zip(mask.downcast_iter())
                    .map(|(arr, mask)| {
                        let a = set_with_mask(arr, mask, value, T::get_dtype().to_arrow());
                        Arc::new(a) as ArrayRef
                    })
                    .collect();
                return Ok(ChunkedArray::new_from_chunks(self.name(), chunks));
            }
        }
        // slow path, could be optimized.
        let ca = mask
            .into_iter()
            .zip(self.into_iter())
            .map(|(mask_val, opt_val)| match mask_val {
                Some(true) => value,
                _ => opt_val,
            })
            .collect();
        Ok(ca)
    }

    fn set_with<F>(&'a self, mask: &BooleanChunked, f: F) -> Result<Self>
    where
        F: Fn(Option<T::Native>) -> Option<T::Native>,
    {
        check_bounds!(self, mask);
        let ca = mask
            .into_iter()
            .zip(self.into_iter())
            .map(|(mask_val, opt_val)| match mask_val {
                Some(true) => f(opt_val),
                _ => opt_val,
            })
            .collect();
        Ok(ca)
    }
}

impl<'a> ChunkSet<'a, bool, bool> for BooleanChunked {
    fn set_at_idx<I: IntoIterator<Item = usize>>(
        &'a self,
        idx: I,
        value: Option<bool>,
    ) -> Result<Self> {
        self.set_at_idx_with(idx, |_| value)
    }

    fn set_at_idx_with<I: IntoIterator<Item = usize>, F>(&'a self, idx: I, f: F) -> Result<Self>
    where
        F: Fn(Option<bool>) -> Option<bool>,
    {
        // TODO: implement fast path
        let mut builder = BooleanChunkedBuilder::new(self.name(), self.len());
        impl_set_at_idx_with!(self, builder, idx, f)
    }

    fn set(&'a self, mask: &BooleanChunked, value: Option<bool>) -> Result<Self> {
        check_bounds!(self, mask);
        let ca = mask
            .into_iter()
            .zip(self.into_iter())
            .map(|(mask_val, opt_val)| match mask_val {
                Some(true) => value,
                _ => opt_val,
            })
            .collect();
        Ok(ca)
    }

    fn set_with<F>(&'a self, mask: &BooleanChunked, f: F) -> Result<Self>
    where
        F: Fn(Option<bool>) -> Option<bool>,
    {
        check_bounds!(self, mask);
        let ca = mask
            .into_iter()
            .zip(self.into_iter())
            .map(|(mask_val, opt_val)| match mask_val {
                Some(true) => f(opt_val),
                _ => opt_val,
            })
            .collect();
        Ok(ca)
    }
}

impl<'a> ChunkSet<'a, &'a str, String> for Utf8Chunked {
    fn set_at_idx<I: IntoIterator<Item = usize>>(
        &'a self,
        idx: I,
        opt_value: Option<&'a str>,
    ) -> Result<Self>
    where
        Self: Sized,
    {
        let idx_iter = idx.into_iter();
        let mut ca_iter = self.into_iter().enumerate();
        let mut builder = Utf8ChunkedBuilder::new(self.name(), self.len(), self.get_values_size());

        for current_idx in idx_iter {
            if current_idx > self.len() {
                return Err(PolarsError::OutOfBounds(
                    format!(
                        "index: {} outside of ChunkedArray with length: {}",
                        current_idx,
                        self.len()
                    )
                    .into(),
                ));
            }
            for (cnt_idx, opt_val_self) in &mut ca_iter {
                if cnt_idx == current_idx {
                    builder.append_option(opt_value);
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
        Ok(ca)
    }

    fn set_at_idx_with<I: IntoIterator<Item = usize>, F>(&'a self, idx: I, f: F) -> Result<Self>
    where
        Self: Sized,
        F: Fn(Option<&'a str>) -> Option<String>,
    {
        let mut builder = Utf8ChunkedBuilder::new(self.name(), self.len(), self.get_values_size());
        impl_set_at_idx_with!(self, builder, idx, f)
    }

    fn set(&'a self, mask: &BooleanChunked, value: Option<&'a str>) -> Result<Self>
    where
        Self: Sized,
    {
        check_bounds!(self, mask);
        let ca = mask
            .into_iter()
            .zip(self.into_iter())
            .map(|(mask_val, opt_val)| match mask_val {
                Some(true) => value,
                _ => opt_val,
            })
            .collect();
        Ok(ca)
    }

    fn set_with<F>(&'a self, mask: &BooleanChunked, f: F) -> Result<Self>
    where
        Self: Sized,
        F: Fn(Option<&'a str>) -> Option<String>,
    {
        check_bounds!(self, mask);
        let mut builder = Utf8ChunkedBuilder::new(self.name(), self.len(), self.get_values_size());
        self.into_iter()
            .zip(mask)
            .for_each(|(opt_val, opt_mask)| match opt_mask {
                Some(true) => builder.append_option(f(opt_val)),
                _ => builder.append_option(opt_val),
            });
        Ok(builder.finish())
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    fn test_set() {
        let ca = Int32Chunked::new_from_slice("a", &[1, 2, 3]);
        let mask = BooleanChunked::new_from_slice("mask", &[false, true, false]);
        let ca = ca.set(&mask, Some(5)).unwrap();
        assert_eq!(Vec::from(&ca), &[Some(1), Some(5), Some(3)]);

        let ca = Int32Chunked::new_from_slice("a", &[1, 2, 3]);
        let mask = BooleanChunked::new_from_opt_slice("mask", &[None, Some(true), None]);
        let ca = ca.set(&mask, Some(5)).unwrap();
        assert_eq!(Vec::from(&ca), &[Some(1), Some(5), Some(3)]);

        let ca = Int32Chunked::new_from_slice("a", &[1, 2, 3]);
        let mask = BooleanChunked::new_from_opt_slice("mask", &[None, None, None]);
        let ca = ca.set(&mask, Some(5)).unwrap();
        assert_eq!(Vec::from(&ca), &[Some(1), Some(2), Some(3)]);

        let ca = Int32Chunked::new_from_slice("a", &[1, 2, 3]);
        let mask = BooleanChunked::new_from_opt_slice("mask", &[Some(true), Some(false), None]);
        let ca = ca.set(&mask, Some(5)).unwrap();
        assert_eq!(Vec::from(&ca), &[Some(5), Some(2), Some(3)]);

        let ca = ca.set_at_idx(vec![0, 1], Some(10)).unwrap();
        assert_eq!(Vec::from(&ca), &[Some(10), Some(10), Some(3)]);

        assert!(ca.set_at_idx(vec![0, 10], Some(0)).is_err());

        // test booleans
        let ca = BooleanChunked::new_from_slice("a", &[true, true, true]);
        let mask = BooleanChunked::new_from_slice("mask", &[false, true, false]);
        let ca = ca.set(&mask, None).unwrap();
        assert_eq!(Vec::from(&ca), &[Some(true), None, Some(true)]);

        // test utf8
        let ca = Utf8Chunked::new_from_slice("a", &["foo", "foo", "foo"]);
        let mask = BooleanChunked::new_from_slice("mask", &[false, true, false]);
        let ca = ca.set(&mask, Some("bar")).unwrap();
        assert_eq!(Vec::from(&ca), &[Some("foo"), Some("bar"), Some("foo")]);
    }

    #[test]
    fn test_set_null_values() {
        let ca = Int32Chunked::new_from_opt_slice("a", &[Some(1), None, Some(3)]);
        let mask = BooleanChunked::new_from_opt_slice("mask", &[Some(false), Some(true), None]);
        let ca = ca.set(&mask, Some(2)).unwrap();
        assert_eq!(Vec::from(&ca), &[Some(1), Some(2), Some(3)]);

        let ca = Utf8Chunked::new_from_opt_slice("a", &[Some("foo"), None, Some("bar")]);
        let ca = ca.set(&mask, Some("foo")).unwrap();
        assert_eq!(Vec::from(&ca), &[Some("foo"), Some("foo"), Some("bar")]);

        let ca = BooleanChunked::new_from_opt_slice("a", &[Some(false), None, Some(true)]);
        let ca = ca.set(&mask, Some(true)).unwrap();
        assert_eq!(Vec::from(&ca), &[Some(false), Some(true), Some(true)]);
    }
}
