use crate::prelude::*;

impl<T> ChunkSet<T::Native> for ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: Copy,
{
    fn set_at_idx<I: AsTakeIndex>(
        &mut self,
        idx: &I,
        value: Option<T::Native>,
    ) -> Result<&mut Self> {
        // TODO: implement fast path
        let mut idx_iter = idx.as_take_iter();
        let mut ca_iter = self.into_iter().enumerate();

        let mut builder = PrimitiveChunkedBuilder::<T>::new(self.name(), self.len());

        while let Some(current_idx) = idx_iter.next() {
            if current_idx > self.len() {
                return Err(PolarsError::OutOfBounds);
            }
            while let Some((cnt_idx, opt_val)) = ca_iter.next() {
                if cnt_idx == current_idx {
                    builder.append_option(value);
                    break;
                } else {
                    builder.append_option(opt_val);
                }
            }
        }
        // the last idx is probably not the last value so we finish the iterator
        while let Some((_, opt_val)) = ca_iter.next() {
            builder.append_option(opt_val);
        }

        let ca = builder.finish();
        self.chunks = ca.chunks;
        Ok(self)
    }

    fn set(&mut self, mask: &BooleanChunked, value: Option<T::Native>) -> Result<&mut Self> {
        if self.len() != mask.len() {
            return Err(PolarsError::ShapeMisMatch);
        }

        // TODO: could make faster by also checking the mask for a fast path.
        let ca: ChunkedArray<T> = match self.cont_slice() {
            // fast path
            Ok(slice) => slice
                .iter()
                .zip(mask)
                .map(|(&val, opt_mask)| match opt_mask {
                    None => Some(val),
                    Some(true) => value,
                    Some(false) => Some(val),
                })
                .collect(),
            // slower path
            Err(_) => self
                .into_iter()
                .zip(mask)
                .map(|(opt_val, opt_mask)| match opt_mask {
                    None => opt_val,
                    Some(true) => value,
                    Some(false) => opt_val,
                })
                .collect(),
        };

        self.chunks = ca.chunks;
        Ok(self)
    }
}

macro_rules! impl_chunkset {
    ($value_type:ty, $ca_type:ident, $builder:ident) => {
        impl ChunkSet<$value_type> for $ca_type {
            fn set_at_idx<I: AsTakeIndex>(
                &mut self,
                idx: &I,
                value: Option<$value_type>,
            ) -> Result<&mut Self> {
                let mut idx_iter = idx.as_take_iter();
                let mut ca_iter = self.into_iter().enumerate();

                let mut builder = $builder::new(self.name(), self.len());

                while let Some(current_idx) = idx_iter.next() {
                    if current_idx > self.len() {
                        return Err(PolarsError::OutOfBounds);
                    }
                    while let Some((cnt_idx, opt_val)) = ca_iter.next() {
                        if cnt_idx == current_idx {
                            builder.append_option(value);
                            break;
                        } else {
                            builder.append_option(opt_val);
                        }
                    }
                }
                // the last idx is probably not the last value so we finish the iterator
                while let Some((_, opt_val)) = ca_iter.next() {
                    builder.append_option(opt_val);
                }

                let ca = builder.finish();
                self.chunks = ca.chunks;
                Ok(self)
            }

            fn set(
                &mut self,
                mask: &BooleanChunked,
                value: Option<$value_type>,
            ) -> Result<&mut Self> {
                if self.len() != mask.len() {
                    return Err(PolarsError::ShapeMisMatch);
                }

                let ca: $ca_type = self
                    .into_iter()
                    .zip(mask)
                    .map(|(opt_val, opt_mask)| match opt_mask {
                        None => opt_val,
                        Some(true) => value,
                        Some(false) => opt_val,
                    })
                    .collect();

                self.chunks = ca.chunks;
                Ok(self)
            }
        }
    };
}

impl_chunkset!(&str, Utf8Chunked, Utf8ChunkedBuilder);
impl_chunkset!(bool, BooleanChunked, BooleanChunkedBuilder);

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    fn test_set() {
        let mut ca = Int32Chunked::new_from_slice("a", &[1, 2, 3]);
        let mask = BooleanChunked::new_from_slice("mask", &[false, true, false]);
        ca.set(&mask, Some(5)).unwrap();
        assert_eq!(Vec::from(&ca), &[Some(1), Some(5), Some(3)]);
        let mut ca = Int32Chunked::new_from_slice("a", &[1, 2, 3]);
        let mask = BooleanChunked::new_from_opt_slice("mask", &[None, Some(true), None]);
        ca.set(&mask, Some(5)).unwrap();
        assert_eq!(Vec::from(&ca), &[Some(1), Some(5), Some(3)]);

        ca.set_at_idx(&[0, 1], Some(10)).unwrap();
        assert_eq!(Vec::from(&ca), &[Some(10), Some(10), Some(3)]);

        // check that out of bounds error doesn't modify original
        assert!(ca.set_at_idx(&[0, 10], Some(0)).is_err());
        assert_eq!(Vec::from(&ca), &[Some(10), Some(10), Some(3)]);

        // test booleans
        let mut ca = BooleanChunked::new_from_slice("a", &[true, true, true]);
        let mask = BooleanChunked::new_from_slice("mask", &[false, true, false]);
        ca.set(&mask, None).unwrap();
        assert_eq!(Vec::from(&ca), &[Some(true), None, Some(true)]);

        // test utf8
        let mut ca = Utf8Chunked::new_from_slice("a", &["foo", "foo", "foo"]);
        let mask = BooleanChunked::new_from_slice("mask", &[false, true, false]);
        ca.set(&mask, Some("bar")).unwrap();
        assert_eq!(Vec::from(&ca), &[Some("foo"), Some("bar"), Some("foo")]);
    }
}
