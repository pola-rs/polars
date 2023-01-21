use arrow::bitmap::MutableBitmap;
use polars_arrow::array::ValueSize;
use polars_arrow::kernels::set::{set_at_idx_no_null, set_with_mask};
use polars_arrow::prelude::FromData;

use crate::prelude::*;
use crate::utils::{align_chunks_binary, CustomIterTools};

macro_rules! impl_set_at_idx_with {
    ($self:ident, $builder:ident, $idx:ident, $f:ident) => {{
        let mut idx_iter = $idx.into_iter();
        let mut ca_iter = $self.into_iter().enumerate();

        while let Some(current_idx) = idx_iter.next() {
            if current_idx as usize > $self.len() {
                return Err(PolarsError::ComputeError(
                    format!(
                        "index: {} outside of ChunkedArray with length: {}",
                        current_idx,
                        $self.len()
                    )
                    .into(),
                ));
            }
            while let Some((cnt_idx, opt_val)) = ca_iter.next() {
                if cnt_idx == current_idx as usize {
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
    fn set_at_idx<I: IntoIterator<Item = IdxSize>>(
        &'a self,
        idx: I,
        value: Option<T::Native>,
    ) -> PolarsResult<Self> {
        if !self.has_validity() {
            if let Some(value) = value {
                // fast path uses kernel
                if self.chunks.len() == 1 {
                    let arr = set_at_idx_no_null(
                        self.downcast_iter().next().unwrap(),
                        idx.into_iter(),
                        value,
                        T::get_dtype().to_arrow(),
                    )?;
                    return unsafe { Ok(Self::from_chunks(self.name(), vec![Box::new(arr)])) };
                }
                // Other fast path. Slightly slower as it does not do a memcpy
                else {
                    let mut av = self.into_no_null_iter().collect::<Vec<_>>();
                    let data = av.as_mut_slice();

                    idx.into_iter().try_for_each::<_, PolarsResult<_>>(|idx| {
                        let val = data.get_mut(idx as usize).ok_or_else(|| {
                            PolarsError::ComputeError(
                                format!("{} out of bounds on array of length: {}", idx, self.len())
                                    .into(),
                            )
                        })?;
                        *val = value;
                        Ok(())
                    })?;
                    return Ok(Self::from_vec(self.name(), av));
                }
            }
        }
        self.set_at_idx_with(idx, |_| value)
    }

    fn set_at_idx_with<I: IntoIterator<Item = IdxSize>, F>(
        &'a self,
        idx: I,
        f: F,
    ) -> PolarsResult<Self>
    where
        F: Fn(Option<T::Native>) -> Option<T::Native>,
    {
        let mut builder = PrimitiveChunkedBuilder::<T>::new(self.name(), self.len());
        impl_set_at_idx_with!(self, builder, idx, f)
    }

    fn set(&'a self, mask: &BooleanChunked, value: Option<T::Native>) -> PolarsResult<Self> {
        check_bounds!(self, mask);

        // Fast path uses the kernel in polars-arrow
        if let (Some(value), false) = (value, mask.has_validity()) {
            let (left, mask) = align_chunks_binary(self, mask);

            // apply binary kernel.
            let chunks = left
                .downcast_iter()
                .zip(mask.downcast_iter())
                .map(|(arr, mask)| {
                    let a = set_with_mask(arr, mask, value, T::get_dtype().to_arrow());
                    Box::new(a) as ArrayRef
                })
                .collect();
            Ok(unsafe { ChunkedArray::from_chunks(self.name(), chunks) })
        } else {
            // slow path, could be optimized.
            let ca = mask
                .into_iter()
                .zip(self.into_iter())
                .map(|(mask_val, opt_val)| match mask_val {
                    Some(true) => value,
                    _ => opt_val,
                })
                .collect_trusted();
            Ok(ca)
        }
    }
}

impl<'a> ChunkSet<'a, bool, bool> for BooleanChunked {
    fn set_at_idx<I: IntoIterator<Item = IdxSize>>(
        &'a self,
        idx: I,
        value: Option<bool>,
    ) -> PolarsResult<Self> {
        self.set_at_idx_with(idx, |_| value)
    }

    fn set_at_idx_with<I: IntoIterator<Item = IdxSize>, F>(
        &'a self,
        idx: I,
        f: F,
    ) -> PolarsResult<Self>
    where
        F: Fn(Option<bool>) -> Option<bool>,
    {
        let mut values = MutableBitmap::with_capacity(self.len());
        let mut validity = MutableBitmap::with_capacity(self.len());

        for a in self.downcast_iter() {
            values.extend_from_bitmap(a.values());
            if let Some(v) = a.validity() {
                validity.extend_from_bitmap(v)
            } else {
                validity.extend_constant(a.len(), true);
            }
        }

        for i in idx {
            let input = if validity.get(i as usize) {
                Some(values.get(i as usize))
            } else {
                None
            };
            match f(input) {
                None => validity.set(i as usize, false),
                Some(v) => values.set(i as usize, v),
            }
        }
        let arr = BooleanArray::from_data_default(values.into(), Some(validity.into()));

        Ok(unsafe { BooleanChunked::from_chunks(self.name(), vec![Box::new(arr)]) })
    }

    fn set(&'a self, mask: &BooleanChunked, value: Option<bool>) -> PolarsResult<Self> {
        check_bounds!(self, mask);
        let ca = mask
            .into_iter()
            .zip(self.into_iter())
            .map(|(mask_val, opt_val)| match mask_val {
                Some(true) => value,
                _ => opt_val,
            })
            .collect_trusted();
        Ok(ca)
    }
}

impl<'a> ChunkSet<'a, &'a str, String> for Utf8Chunked {
    fn set_at_idx<I: IntoIterator<Item = IdxSize>>(
        &'a self,
        idx: I,
        opt_value: Option<&'a str>,
    ) -> PolarsResult<Self>
    where
        Self: Sized,
    {
        let idx_iter = idx.into_iter();
        let mut ca_iter = self.into_iter().enumerate();
        let mut builder = Utf8ChunkedBuilder::new(self.name(), self.len(), self.get_values_size());

        for current_idx in idx_iter {
            if current_idx as usize > self.len() {
                return Err(PolarsError::ComputeError(
                    format!(
                        "index: {} outside of ChunkedArray with length: {}",
                        current_idx,
                        self.len()
                    )
                    .into(),
                ));
            }
            for (cnt_idx, opt_val_self) in &mut ca_iter {
                if cnt_idx == current_idx as usize {
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

    fn set_at_idx_with<I: IntoIterator<Item = IdxSize>, F>(
        &'a self,
        idx: I,
        f: F,
    ) -> PolarsResult<Self>
    where
        Self: Sized,
        F: Fn(Option<&'a str>) -> Option<String>,
    {
        let mut builder = Utf8ChunkedBuilder::new(self.name(), self.len(), self.get_values_size());
        impl_set_at_idx_with!(self, builder, idx, f)
    }

    fn set(&'a self, mask: &BooleanChunked, value: Option<&'a str>) -> PolarsResult<Self>
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
            .collect_trusted();
        Ok(ca)
    }
}

#[cfg(feature = "dtype-binary")]
impl<'a> ChunkSet<'a, &'a [u8], Vec<u8>> for BinaryChunked {
    fn set_at_idx<I: IntoIterator<Item = IdxSize>>(
        &'a self,
        idx: I,
        opt_value: Option<&'a [u8]>,
    ) -> PolarsResult<Self>
    where
        Self: Sized,
    {
        let idx_iter = idx.into_iter();
        let mut ca_iter = self.into_iter().enumerate();
        let mut builder =
            BinaryChunkedBuilder::new(self.name(), self.len(), self.get_values_size());

        for current_idx in idx_iter {
            if current_idx as usize > self.len() {
                return Err(PolarsError::ComputeError(
                    format!(
                        "index: {} outside of ChunkedArray with length: {}",
                        current_idx,
                        self.len()
                    )
                    .into(),
                ));
            }
            for (cnt_idx, opt_val_self) in &mut ca_iter {
                if cnt_idx == current_idx as usize {
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

    fn set_at_idx_with<I: IntoIterator<Item = IdxSize>, F>(
        &'a self,
        idx: I,
        f: F,
    ) -> PolarsResult<Self>
    where
        Self: Sized,
        F: Fn(Option<&'a [u8]>) -> Option<Vec<u8>>,
    {
        let mut builder =
            BinaryChunkedBuilder::new(self.name(), self.len(), self.get_values_size());
        impl_set_at_idx_with!(self, builder, idx, f)
    }

    fn set(&'a self, mask: &BooleanChunked, value: Option<&'a [u8]>) -> PolarsResult<Self>
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
            .collect_trusted();
        Ok(ca)
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    fn test_set() {
        let ca = Int32Chunked::new("a", &[1, 2, 3]);
        let mask = BooleanChunked::new("mask", &[false, true, false]);
        let ca = ca.set(&mask, Some(5)).unwrap();
        assert_eq!(Vec::from(&ca), &[Some(1), Some(5), Some(3)]);

        let ca = Int32Chunked::new("a", &[1, 2, 3]);
        let mask = BooleanChunked::new("mask", &[None, Some(true), None]);
        let ca = ca.set(&mask, Some(5)).unwrap();
        assert_eq!(Vec::from(&ca), &[Some(1), Some(5), Some(3)]);

        let ca = Int32Chunked::new("a", &[1, 2, 3]);
        let mask = BooleanChunked::new("mask", &[None, None, None]);
        let ca = ca.set(&mask, Some(5)).unwrap();
        assert_eq!(Vec::from(&ca), &[Some(1), Some(2), Some(3)]);

        let ca = Int32Chunked::new("a", &[1, 2, 3]);
        let mask = BooleanChunked::new("mask", &[Some(true), Some(false), None]);
        let ca = ca.set(&mask, Some(5)).unwrap();
        assert_eq!(Vec::from(&ca), &[Some(5), Some(2), Some(3)]);

        let ca = ca.set_at_idx(vec![0, 1], Some(10)).unwrap();
        assert_eq!(Vec::from(&ca), &[Some(10), Some(10), Some(3)]);

        assert!(ca.set_at_idx(vec![0, 10], Some(0)).is_err());

        // test booleans
        let ca = BooleanChunked::new("a", &[true, true, true]);
        let mask = BooleanChunked::new("mask", &[false, true, false]);
        let ca = ca.set(&mask, None).unwrap();
        assert_eq!(Vec::from(&ca), &[Some(true), None, Some(true)]);

        // test utf8
        let ca = Utf8Chunked::new("a", &["foo", "foo", "foo"]);
        let mask = BooleanChunked::new("mask", &[false, true, false]);
        let ca = ca.set(&mask, Some("bar")).unwrap();
        assert_eq!(Vec::from(&ca), &[Some("foo"), Some("bar"), Some("foo")]);
    }

    #[test]
    fn test_set_null_values() {
        let ca = Int32Chunked::new("a", &[Some(1), None, Some(3)]);
        let mask = BooleanChunked::new("mask", &[Some(false), Some(true), None]);
        let ca = ca.set(&mask, Some(2)).unwrap();
        assert_eq!(Vec::from(&ca), &[Some(1), Some(2), Some(3)]);

        let ca = Utf8Chunked::new("a", &[Some("foo"), None, Some("bar")]);
        let ca = ca.set(&mask, Some("foo")).unwrap();
        assert_eq!(Vec::from(&ca), &[Some("foo"), Some("foo"), Some("bar")]);

        let ca = BooleanChunked::new("a", &[Some(false), None, Some(true)]);
        let ca = ca.set(&mask, Some(true)).unwrap();
        assert_eq!(Vec::from(&ca), &[Some(false), Some(true), Some(true)]);
    }
}
