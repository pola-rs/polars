use arrow::bitmap::MutableBitmap;
use arrow::legacy::kernels::set::{scatter_single_non_null, set_with_mask};

use crate::prelude::*;
use crate::utils::align_chunks_binary;

macro_rules! impl_scatter_with {
    ($self:ident, $builder:ident, $idx:ident, $f:ident) => {{
        let mut ca_iter = $self.into_iter().enumerate();

        for current_idx in $idx.into_iter().map(|i| i as usize) {
            polars_ensure!(current_idx < $self.len(), oob = current_idx, $self.len());
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
        polars_ensure!(
            $self.len() == $mask.len(),
            ShapeMismatch: "invalid mask in `get` operation: shape doesn't match array's shape"
        );
    }};
}

impl<'a, T> ChunkSet<'a, T::Native, T::Native> for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn scatter_single<I: IntoIterator<Item = IdxSize>>(
        &'a self,
        idx: I,
        value: Option<T::Native>,
    ) -> PolarsResult<Self> {
        if !self.has_nulls() {
            if let Some(value) = value {
                // Fast path uses kernel.
                if self.chunks.len() == 1 {
                    let arr = scatter_single_non_null(
                        self.downcast_iter().next().unwrap(),
                        idx,
                        value,
                        T::get_dtype().to_arrow(CompatLevel::newest()),
                    )?;
                    return Ok(Self::with_chunk(self.name(), arr));
                }
                // Other fast path. Slightly slower as it does not do a memcpy.
                else {
                    let mut av = self.into_no_null_iter().collect::<Vec<_>>();
                    let data = av.as_mut_slice();

                    idx.into_iter().try_for_each::<_, PolarsResult<_>>(|idx| {
                        let val = data
                            .get_mut(idx as usize)
                            .ok_or_else(|| polars_err!(oob = idx as usize, self.len()))?;
                        *val = value;
                        Ok(())
                    })?;
                    return Ok(Self::from_vec(self.name(), av));
                }
            }
        }
        self.scatter_with(idx, |_| value)
    }

    fn scatter_with<I: IntoIterator<Item = IdxSize>, F>(
        &'a self,
        idx: I,
        f: F,
    ) -> PolarsResult<Self>
    where
        F: Fn(Option<T::Native>) -> Option<T::Native>,
    {
        let mut builder = PrimitiveChunkedBuilder::<T>::new(self.name(), self.len());
        impl_scatter_with!(self, builder, idx, f)
    }

    fn set(&'a self, mask: &BooleanChunked, value: Option<T::Native>) -> PolarsResult<Self> {
        check_bounds!(self, mask);

        // Fast path uses the kernel in polars-arrow.
        if let (Some(value), false) = (value, mask.has_nulls()) {
            let (left, mask) = align_chunks_binary(self, mask);

            // Apply binary kernel.
            let chunks = left
                .downcast_iter()
                .zip(mask.downcast_iter())
                .map(|(arr, mask)| {
                    set_with_mask(
                        arr,
                        mask,
                        value,
                        T::get_dtype().to_arrow(CompatLevel::newest()),
                    )
                });
            Ok(ChunkedArray::from_chunk_iter(self.name(), chunks))
        } else {
            // slow path, could be optimized.
            let ca = mask
                .into_iter()
                .zip(self)
                .map(|(mask_val, opt_val)| match mask_val {
                    Some(true) => value,
                    _ => opt_val,
                })
                .collect_trusted::<Self>()
                .with_name(self.name());
            Ok(ca)
        }
    }
}

impl<'a> ChunkSet<'a, bool, bool> for BooleanChunked {
    fn scatter_single<I: IntoIterator<Item = IdxSize>>(
        &'a self,
        idx: I,
        value: Option<bool>,
    ) -> PolarsResult<Self> {
        self.scatter_with(idx, |_| value)
    }

    fn scatter_with<I: IntoIterator<Item = IdxSize>, F>(
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

        for i in idx.into_iter().map(|i| i as usize) {
            let input = validity.get(i).then(|| values.get(i));
            validity.set(i, f(input).unwrap_or(false));
        }
        let arr = BooleanArray::from_data_default(values.into(), Some(validity.into()));
        Ok(BooleanChunked::with_chunk(self.name(), arr))
    }

    fn set(&'a self, mask: &BooleanChunked, value: Option<bool>) -> PolarsResult<Self> {
        check_bounds!(self, mask);
        let ca = mask
            .into_iter()
            .zip(self)
            .map(|(mask_val, opt_val)| match mask_val {
                Some(true) => value,
                _ => opt_val,
            })
            .collect_trusted::<Self>()
            .with_name(self.name());
        Ok(ca)
    }
}

impl<'a> ChunkSet<'a, &'a str, String> for StringChunked {
    fn scatter_single<I: IntoIterator<Item = IdxSize>>(
        &'a self,
        idx: I,
        opt_value: Option<&'a str>,
    ) -> PolarsResult<Self>
    where
        Self: Sized,
    {
        let idx_iter = idx.into_iter();
        let mut ca_iter = self.into_iter().enumerate();
        let mut builder = StringChunkedBuilder::new(self.name(), self.len());

        for current_idx in idx_iter.into_iter().map(|i| i as usize) {
            polars_ensure!(current_idx < self.len(), oob = current_idx, self.len());
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

    fn scatter_with<I: IntoIterator<Item = IdxSize>, F>(
        &'a self,
        idx: I,
        f: F,
    ) -> PolarsResult<Self>
    where
        Self: Sized,
        F: Fn(Option<&'a str>) -> Option<String>,
    {
        let mut builder = StringChunkedBuilder::new(self.name(), self.len());
        impl_scatter_with!(self, builder, idx, f)
    }

    fn set(&'a self, mask: &BooleanChunked, value: Option<&'a str>) -> PolarsResult<Self>
    where
        Self: Sized,
    {
        check_bounds!(self, mask);
        let ca = mask
            .into_iter()
            .zip(self)
            .map(|(mask_val, opt_val)| match mask_val {
                Some(true) => value,
                _ => opt_val,
            })
            .collect_trusted::<Self>()
            .with_name(self.name());
        Ok(ca)
    }
}

impl<'a> ChunkSet<'a, &'a [u8], Vec<u8>> for BinaryChunked {
    fn scatter_single<I: IntoIterator<Item = IdxSize>>(
        &'a self,
        idx: I,
        opt_value: Option<&'a [u8]>,
    ) -> PolarsResult<Self>
    where
        Self: Sized,
    {
        let mut ca_iter = self.into_iter().enumerate();
        let mut builder = BinaryChunkedBuilder::new(self.name(), self.len());

        for current_idx in idx.into_iter().map(|i| i as usize) {
            polars_ensure!(current_idx < self.len(), oob = current_idx, self.len());
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

    fn scatter_with<I: IntoIterator<Item = IdxSize>, F>(
        &'a self,
        idx: I,
        f: F,
    ) -> PolarsResult<Self>
    where
        Self: Sized,
        F: Fn(Option<&'a [u8]>) -> Option<Vec<u8>>,
    {
        let mut builder = BinaryChunkedBuilder::new(self.name(), self.len());
        impl_scatter_with!(self, builder, idx, f)
    }

    fn set(&'a self, mask: &BooleanChunked, value: Option<&'a [u8]>) -> PolarsResult<Self>
    where
        Self: Sized,
    {
        check_bounds!(self, mask);
        let ca = mask
            .into_iter()
            .zip(self)
            .map(|(mask_val, opt_val)| match mask_val {
                Some(true) => value,
                _ => opt_val,
            })
            .collect_trusted::<Self>()
            .with_name(self.name());
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

        let ca = ca.scatter_single(vec![0, 1], Some(10)).unwrap();
        assert_eq!(Vec::from(&ca), &[Some(10), Some(10), Some(3)]);

        assert!(ca.scatter_single(vec![0, 10], Some(0)).is_err());

        // test booleans
        let ca = BooleanChunked::new("a", &[true, true, true]);
        let mask = BooleanChunked::new("mask", &[false, true, false]);
        let ca = ca.set(&mask, None).unwrap();
        assert_eq!(Vec::from(&ca), &[Some(true), None, Some(true)]);

        // test string
        let ca = StringChunked::new("a", &["foo", "foo", "foo"]);
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

        let ca = StringChunked::new("a", &[Some("foo"), None, Some("bar")]);
        let ca = ca.set(&mask, Some("foo")).unwrap();
        assert_eq!(Vec::from(&ca), &[Some("foo"), Some("foo"), Some("bar")]);

        let ca = BooleanChunked::new("a", &[Some(false), None, Some(true)]);
        let ca = ca.set(&mask, Some(true)).unwrap();
        assert_eq!(Vec::from(&ca), &[Some(false), Some(true), Some(true)]);
    }
}
