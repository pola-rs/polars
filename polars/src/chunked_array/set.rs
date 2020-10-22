use crate::prelude::*;
use crate::{chunked_array::kernels::set::set_with_value, utils::Xob};
use arrow::array::ArrayRef;
use std::sync::Arc;

macro_rules! impl_set_at_idx_with {
    ($self:ident, $builder:ident, $idx:ident, $f:ident) => {{
        let mut idx_iter = $idx.as_take_iter();
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
    fn set_at_idx<I: AsTakeIndex>(&'a self, idx: &I, value: Option<T::Native>) -> Result<Self> {
        self.set_at_idx_with(idx, |_| value)
    }

    fn set_at_idx_with<I: AsTakeIndex, F>(&'a self, idx: &I, f: F) -> Result<Self>
    where
        F: Fn(Option<T::Native>) -> Option<T::Native>,
    {
        // TODO: implement fast path
        let mut builder = PrimitiveChunkedBuilder::<T>::new(self.name(), self.len());
        impl_set_at_idx_with!(self, builder, idx, f)
    }

    fn set(&'a self, mask: &BooleanChunked, value: Option<T::Native>) -> Result<Self> {
        if T::get_data_type() != ArrowDataType::Boolean
            && value.is_some()
            && self.chunk_id() == mask.chunk_id()
        {
            let value = value.unwrap();
            let chunks = self
                .downcast_chunks()
                .into_iter()
                .zip(mask.downcast_chunks())
                .map(|(arr, mask)| {
                    let a = set_with_value(mask, arr, value);
                    Arc::new(a) as ArrayRef
                })
                .collect();
            return Ok(ChunkedArray::new_from_chunks(self.name(), chunks));
        }
        // all the code does practically the same but has different fast paths. We do this
        // again because we don't need the return option in all cases. Otherwise we would have called
        // set_with
        match value {
            Some(new_value) => {
                // fast path because self has no nulls
                let mut ca = if self.is_optimal_aligned() {
                    // fast path because mask has no nulls
                    if mask.is_optimal_aligned() {
                        let ca: Xob<_> = self
                            .into_no_null_iter()
                            .zip(mask.into_no_null_iter())
                            .map(|(val, mask)| match mask {
                                true => new_value,
                                false => val,
                            })
                            .collect();
                        ca.into_inner()
                    // slower path, mask has null values
                    } else {
                        let ca: Xob<_> = self
                            .into_no_null_iter()
                            .zip(mask)
                            .map(|(val, opt_mask)| match opt_mask {
                                None => val,
                                Some(true) => new_value,
                                Some(false) => val,
                            })
                            .collect();
                        ca.into_inner()
                    }
                } else {
                    // mask has no nulls, self has
                    if mask.is_optimal_aligned() {
                        self.into_iter()
                            .zip(mask.into_no_null_iter())
                            .map(|(opt_val, mask)| match mask {
                                true => Some(new_value),
                                false => opt_val,
                            })
                            .collect()
                    } else {
                        // slowest path, mask and self have null values
                        self.into_iter()
                            .zip(mask)
                            .map(|(opt_val, opt_mask)| match opt_mask {
                                None => opt_val,
                                Some(true) => Some(new_value),
                                Some(false) => opt_val,
                            })
                            .collect()
                    }
                };
                ca.rename(self.name());
                Ok(ca)
            }
            None => self.set_with(mask, |_| value),
        }
    }

    fn set_with<F>(&'a self, mask: &BooleanChunked, f: F) -> Result<Self>
    where
        F: Fn(Option<T::Native>) -> Option<T::Native>,
    {
        check_bounds!(self, mask);

        // fast path because self has no nulls
        let mut ca: ChunkedArray<T> = if self.is_optimal_aligned() {
            // fast path because mask has no nulls
            if mask.is_optimal_aligned() {
                self.into_no_null_iter()
                    .zip(mask.into_no_null_iter())
                    .map(|(val, mask)| match mask {
                        true => f(Some(val)),
                        false => Some(val),
                    })
                    .collect()

            // slower path, mask has null values
            } else {
                self.into_no_null_iter()
                    .zip(mask)
                    .map(|(val, opt_mask)| match opt_mask {
                        None => Some(val),
                        Some(true) => f(Some(val)),
                        Some(false) => Some(val),
                    })
                    .collect()
            }
        } else {
            // mask has no nulls, self has
            if mask.is_optimal_aligned() {
                self.into_iter()
                    .zip(mask.into_no_null_iter())
                    .map(|(opt_val, mask)| match mask {
                        true => f(opt_val),
                        false => opt_val,
                    })
                    .collect()
            } else {
                // slowest path, mask and self have null values
                self.into_iter()
                    .zip(mask)
                    .map(|(opt_val, opt_mask)| match opt_mask {
                        None => opt_val,
                        Some(true) => f(opt_val),
                        Some(false) => opt_val,
                    })
                    .collect()
            }
        };

        ca.rename(self.name());
        Ok(ca)
    }
}

impl<'a> ChunkSet<'a, bool, bool> for BooleanChunked {
    fn set_at_idx<I: AsTakeIndex>(&'a self, idx: &I, value: Option<bool>) -> Result<Self> {
        self.set_at_idx_with(idx, |_| value)
    }

    fn set_at_idx_with<I: AsTakeIndex, F>(&'a self, idx: &I, f: F) -> Result<Self>
    where
        F: Fn(Option<bool>) -> Option<bool>,
    {
        // TODO: implement fast path
        let mut builder = BooleanChunkedBuilder::new(self.name(), self.len());
        impl_set_at_idx_with!(self, builder, idx, f)
    }

    fn set(&'a self, mask: &BooleanChunked, value: Option<bool>) -> Result<Self> {
        // all the code does practically the same but has different fast paths. We do this
        // again because we don't need the return option in all cases. Otherwise we would have called
        // set_with
        match value {
            Some(new_value) => {
                // fast path because self has no nulls
                let mut ca = if self.is_optimal_aligned() {
                    // fast path because mask has no nulls
                    if mask.is_optimal_aligned() {
                        let ca: Xob<_> = self
                            .into_no_null_iter()
                            .zip(mask.into_no_null_iter())
                            .map(|(val, mask)| match mask {
                                true => new_value,
                                false => val,
                            })
                            .collect();
                        ca.into_inner()
                    // slower path, mask has null values
                    } else {
                        let ca: Xob<_> = self
                            .into_no_null_iter()
                            .zip(mask)
                            .map(|(val, opt_mask)| match opt_mask {
                                None => val,
                                Some(true) => new_value,
                                Some(false) => val,
                            })
                            .collect();
                        ca.into_inner()
                    }
                } else {
                    // mask has no nulls, self has
                    if mask.is_optimal_aligned() {
                        self.into_iter()
                            .zip(mask.into_no_null_iter())
                            .map(|(opt_val, mask)| match mask {
                                true => Some(new_value),
                                false => opt_val,
                            })
                            .collect()
                    } else {
                        // slowest path, mask and self have null values
                        self.into_iter()
                            .zip(mask)
                            .map(|(opt_val, opt_mask)| match opt_mask {
                                None => opt_val,
                                Some(true) => Some(new_value),
                                Some(false) => opt_val,
                            })
                            .collect()
                    }
                };
                ca.rename(self.name());
                Ok(ca)
            }
            None => self.set_with(mask, |_| value),
        }
    }

    fn set_with<F>(&'a self, mask: &BooleanChunked, f: F) -> Result<Self>
    where
        F: Fn(Option<bool>) -> Option<bool>,
    {
        check_bounds!(self, mask);

        // fast path because self has no nulls
        let mut ca: BooleanChunked = if self.is_optimal_aligned() {
            // fast path because mask has no nulls
            if mask.is_optimal_aligned() {
                self.into_no_null_iter()
                    .zip(mask.into_no_null_iter())
                    .map(|(val, mask)| match mask {
                        true => f(Some(val)),
                        false => Some(val),
                    })
                    .collect()

            // slower path, mask has null values
            } else {
                self.into_no_null_iter()
                    .zip(mask)
                    .map(|(val, opt_mask)| match opt_mask {
                        None => Some(val),
                        Some(true) => f(Some(val)),
                        Some(false) => Some(val),
                    })
                    .collect()
            }
        } else {
            // mask has no nulls, self has
            if mask.is_optimal_aligned() {
                self.into_iter()
                    .zip(mask.into_no_null_iter())
                    .map(|(opt_val, mask)| match mask {
                        true => f(opt_val),
                        false => opt_val,
                    })
                    .collect()
            } else {
                // slowest path, mask and self have null values
                self.into_iter()
                    .zip(mask)
                    .map(|(opt_val, opt_mask)| match opt_mask {
                        None => opt_val,
                        Some(true) => f(opt_val),
                        Some(false) => opt_val,
                    })
                    .collect()
            }
        };

        ca.rename(self.name());
        Ok(ca)
    }
}

impl<'a> ChunkSet<'a, &'a str, String> for Utf8Chunked {
    fn set_at_idx<T: AsTakeIndex>(&'a self, idx: &T, opt_value: Option<&'a str>) -> Result<Self>
    where
        Self: Sized,
    {
        let mut idx_iter = idx.as_take_iter();
        let mut ca_iter = self.into_iter().enumerate();
        let mut builder = Utf8ChunkedBuilder::new(self.name(), self.len());

        while let Some(current_idx) = idx_iter.next() {
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
            while let Some((cnt_idx, opt_val_self)) = ca_iter.next() {
                if cnt_idx == current_idx {
                    builder.append_option(opt_value);
                    break;
                } else {
                    builder.append_option(opt_val_self);
                }
            }
        }
        // the last idx is probably not the last value so we finish the iterator
        while let Some((_, opt_val_self)) = ca_iter.next() {
            builder.append_option(opt_val_self);
        }

        let ca = builder.finish();
        Ok(ca)
    }

    fn set_at_idx_with<T: AsTakeIndex, F>(&'a self, idx: &T, f: F) -> Result<Self>
    where
        Self: Sized,
        F: Fn(Option<&'a str>) -> Option<String>,
    {
        let mut builder = Utf8ChunkedBuilder::new(self.name(), self.len());
        impl_set_at_idx_with!(self, builder, idx, f)
    }

    fn set(&'a self, mask: &BooleanChunked, opt_value: Option<&'a str>) -> Result<Self>
    where
        Self: Sized,
    {
        check_bounds!(self, mask);
        let mut builder = Utf8ChunkedBuilder::new(self.name(), self.len());
        self.into_iter()
            .zip(mask)
            .for_each(|(opt_val_self, opt_mask)| match opt_mask {
                None => builder.append_option(opt_val_self),
                Some(true) => builder.append_option(opt_value),
                Some(false) => builder.append_option(opt_val_self),
            });
        Ok(builder.finish())
    }

    fn set_with<F>(&'a self, mask: &BooleanChunked, f: F) -> Result<Self>
    where
        Self: Sized,
        F: Fn(Option<&'a str>) -> Option<String>,
    {
        check_bounds!(self, mask);
        let mut builder = Utf8ChunkedBuilder::new(self.name(), self.len());
        self.into_iter()
            .zip(mask)
            .for_each(|(opt_val, opt_mask)| match opt_mask {
                None => builder.append_option(opt_val),
                Some(true) => builder.append_option(f(opt_val)),
                Some(false) => builder.append_option(opt_val),
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

        let ca = ca.set_at_idx(&[0, 1], Some(10)).unwrap();
        assert_eq!(Vec::from(&ca), &[Some(10), Some(10), Some(3)]);

        assert!(ca.set_at_idx(&[0, 10], Some(0)).is_err());

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
}
