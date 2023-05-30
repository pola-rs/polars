use std::cmp::Ordering;

use super::*;

impl<T> ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn primitive_compare_scalar<Rhs: ToPrimitive>(
        &self,
        rhs: Rhs,
        f: impl Fn(&PrimitiveArray<T::Native>, &dyn Scalar) -> BooleanArray,
    ) -> BooleanChunked {
        let rhs: T::Native = NumCast::from(rhs).unwrap();
        let scalar = PrimitiveScalar::new(T::get_dtype().to_arrow(), Some(rhs));
        self.apply_kernel_cast(&|arr| Box::new(f(arr, &scalar)))
    }
}

fn binary_search<T: PolarsNumericType, F>(
    ca: &ChunkedArray<T>,
    // lhs part of mask will be set to boolean
    // rhs part of mask will be set to !boolean
    lower_part: bool,
    cmp_fn: F,
) -> BooleanChunked
where
    F: Fn(&T::Native) -> Ordering + Copy,
{
    let chunks = ca
        .downcast_iter()
        .map(|arr| {
            let values = arr.values();
            let mask = match values.binary_search_by(cmp_fn) {
                Err(mut idx) => {
                    if idx == 0 || idx == arr.len() {
                        let mut mask = MutableBitmap::with_capacity(arr.len());
                        let fill_value = if idx == 0 { !lower_part } else { lower_part };
                        mask.extend_constant(arr.len(), fill_value);
                        BooleanArray::from_data_default(mask.into(), None)
                    } else {
                        let found_ordering = cmp_fn(&values[idx]);

                        idx = idx.saturating_sub(1);
                        loop {
                            let current_value = unsafe { values.get_unchecked(idx) };
                            let current_output = cmp_fn(current_value);

                            if current_output != found_ordering || idx == 0 {
                                break;
                            }

                            idx = idx.saturating_sub(1);
                        }
                        idx += 1;
                        let mut mask = MutableBitmap::with_capacity(arr.len());
                        mask.extend_constant(idx, lower_part);
                        mask.extend_constant(arr.len() - idx, !lower_part);
                        BooleanArray::from_data_default(mask.into(), None)
                    }
                }
                Ok(_) => {
                    unreachable!()
                }
            };
            Box::new(mask) as ArrayRef
        })
        .collect();
    unsafe { BooleanChunked::from_chunks(ca.name(), chunks) }
}

impl<T, Rhs> ChunkCompare<Rhs> for ChunkedArray<T>
where
    T: PolarsNumericType,
    Rhs: ToPrimitive,
{
    type Item = BooleanChunked;
    fn equal(&self, rhs: Rhs) -> BooleanChunked {
        self.primitive_compare_scalar(rhs, |l, rhs| comparison::eq_scalar(l, rhs))
    }

    fn equal_missing(&self, rhs: Rhs) -> BooleanChunked {
        self.primitive_compare_scalar(rhs, |l, rhs| comparison::eq_scalar_and_validity(l, rhs))
    }

    fn not_equal(&self, rhs: Rhs) -> BooleanChunked {
        self.primitive_compare_scalar(rhs, |l, rhs| comparison::neq_scalar(l, rhs))
    }

    fn not_equal_missing(&self, rhs: Rhs) -> BooleanChunked {
        self.primitive_compare_scalar(rhs, |l, rhs| comparison::neq_scalar_and_validity(l, rhs))
    }

    fn gt(&self, rhs: Rhs) -> BooleanChunked {
        match (self.is_sorted_flag(), self.null_count()) {
            (IsSorted::Ascending, 0) => {
                let rhs: T::Native = NumCast::from(rhs).unwrap();

                let cmp_fn = |a: &T::Native| match compare_fn_nan_max(a, &rhs) {
                    Ordering::Equal | Ordering::Less => Ordering::Less,
                    _ => Ordering::Greater,
                };
                let mut ca = binary_search(self, false, cmp_fn);
                ca.set_sorted_flag(IsSorted::Ascending);
                ca
            }
            _ => self.primitive_compare_scalar(rhs, |l, rhs| comparison::gt_scalar(l, rhs)),
        }
    }

    fn gt_eq(&self, rhs: Rhs) -> BooleanChunked {
        match (self.is_sorted_flag(), self.null_count()) {
            (IsSorted::Ascending, 0) => {
                let rhs: T::Native = NumCast::from(rhs).unwrap();

                let cmp_fn = |a: &T::Native| match compare_fn_nan_max(a, &rhs) {
                    Ordering::Equal | Ordering::Greater => Ordering::Greater,
                    Ordering::Less => Ordering::Less,
                };
                let mut ca = binary_search(self, false, cmp_fn);
                ca.set_sorted_flag(IsSorted::Ascending);
                ca
            }
            _ => self.primitive_compare_scalar(rhs, |l, rhs| comparison::gt_eq_scalar(l, rhs)),
        }
    }

    fn lt(&self, rhs: Rhs) -> BooleanChunked {
        match (self.is_sorted_flag(), self.null_count()) {
            (IsSorted::Ascending, 0) => {
                let rhs: T::Native = NumCast::from(rhs).unwrap();

                let cmp_fn = |a: &T::Native| match compare_fn_nan_max(a, &rhs) {
                    Ordering::Equal | Ordering::Greater => Ordering::Greater,
                    Ordering::Less => Ordering::Less,
                };
                let mut ca = binary_search(self, true, cmp_fn);
                ca.set_sorted_flag(IsSorted::Ascending);
                ca
            }
            _ => self.primitive_compare_scalar(rhs, |l, rhs| comparison::lt_scalar(l, rhs)),
        }
    }

    fn lt_eq(&self, rhs: Rhs) -> BooleanChunked {
        match (self.is_sorted_flag(), self.null_count()) {
            (IsSorted::Ascending, 0) => {
                let rhs: T::Native = NumCast::from(rhs).unwrap();

                let cmp_fn = |a: &T::Native| match compare_fn_nan_max(a, &rhs) {
                    Ordering::Greater => Ordering::Greater,
                    Ordering::Equal | Ordering::Less => Ordering::Less,
                };
                let mut ca = binary_search(self, true, cmp_fn);
                ca.set_sorted_flag(IsSorted::Ascending);
                ca
            }
            _ => self.primitive_compare_scalar(rhs, |l, rhs| comparison::lt_eq_scalar(l, rhs)),
        }
    }
}

impl Utf8Chunked {
    pub(super) fn utf8_compare_scalar(
        &self,
        rhs: &str,
        f: impl Fn(&Utf8Array<i64>, &dyn Scalar) -> BooleanArray,
    ) -> BooleanChunked {
        let scalar = Utf8Scalar::<i64>::new(Some(rhs));
        self.apply_kernel_cast(&|arr| Box::new(f(arr, &scalar)))
    }
}

impl BinaryChunked {
    fn binary_compare_scalar(
        &self,
        rhs: &[u8],
        f: impl Fn(&BinaryArray<i64>, &dyn Scalar) -> BooleanArray,
    ) -> BooleanChunked {
        let scalar = BinaryScalar::<i64>::new(Some(rhs));
        self.apply_kernel_cast(&|arr| Box::new(f(arr, &scalar)))
    }
}

impl ChunkCompare<&[u8]> for BinaryChunked {
    type Item = BooleanChunked;
    fn equal(&self, rhs: &[u8]) -> BooleanChunked {
        self.binary_compare_scalar(rhs, |l, rhs| comparison::eq_scalar(l, rhs))
    }

    fn equal_missing(&self, rhs: &[u8]) -> BooleanChunked {
        self.binary_compare_scalar(rhs, |l, rhs| comparison::eq_scalar_and_validity(l, rhs))
    }

    fn not_equal(&self, rhs: &[u8]) -> BooleanChunked {
        self.binary_compare_scalar(rhs, |l, rhs| comparison::neq_scalar(l, rhs))
    }

    fn not_equal_missing(&self, rhs: &[u8]) -> BooleanChunked {
        self.binary_compare_scalar(rhs, |l, rhs| comparison::neq_scalar_and_validity(l, rhs))
    }

    fn gt(&self, rhs: &[u8]) -> BooleanChunked {
        self.binary_compare_scalar(rhs, |l, rhs| comparison::gt_scalar(l, rhs))
    }

    fn gt_eq(&self, rhs: &[u8]) -> BooleanChunked {
        self.binary_compare_scalar(rhs, |l, rhs| comparison::gt_eq_scalar(l, rhs))
    }

    fn lt(&self, rhs: &[u8]) -> BooleanChunked {
        self.binary_compare_scalar(rhs, |l, rhs| comparison::lt_scalar(l, rhs))
    }

    fn lt_eq(&self, rhs: &[u8]) -> BooleanChunked {
        self.binary_compare_scalar(rhs, |l, rhs| comparison::lt_eq_scalar(l, rhs))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_binary_search_cmp() {
        let mut s = Series::new("", &[1, 1, 2, 2, 4, 8]);
        s.set_sorted_flag(IsSorted::Ascending);
        let out = s.gt(10).unwrap();
        assert!(!out.any());

        let out = s.gt(0).unwrap();
        assert!(out.all());

        let out = s.gt(2).unwrap();
        assert_eq!(
            out.into_series(),
            Series::new("", [false, false, false, false, true, true])
        );
        let out = s.gt(3).unwrap();
        assert_eq!(
            out.into_series(),
            Series::new("", [false, false, false, false, true, true])
        );

        let out = s.gt_eq(10).unwrap();
        assert!(!out.any());
        let out = s.gt_eq(0).unwrap();
        assert!(out.all());

        let out = s.gt_eq(2).unwrap();
        assert_eq!(
            out.into_series(),
            Series::new("", [false, false, true, true, true, true])
        );
        let out = s.gt_eq(3).unwrap();
        assert_eq!(
            out.into_series(),
            Series::new("", [false, false, false, false, true, true])
        );

        let out = s.lt(10).unwrap();
        assert!(out.all());
        let out = s.lt(0).unwrap();
        assert!(!out.any());

        let out = s.lt(2).unwrap();
        assert_eq!(
            out.into_series(),
            Series::new("", [true, true, false, false, false, false])
        );
        let out = s.lt(3).unwrap();
        assert_eq!(
            out.into_series(),
            Series::new("", [true, true, true, true, false, false])
        );

        let out = s.lt_eq(10).unwrap();
        assert!(out.all());
        let out = s.lt_eq(0).unwrap();
        assert!(!out.any());

        let out = s.lt_eq(2).unwrap();
        assert_eq!(
            out.into_series(),
            Series::new("", [true, true, true, true, false, false])
        );
        let out = s.lt(3).unwrap();
        assert_eq!(
            out.into_series(),
            Series::new("", [true, true, true, true, false, false])
        );
    }
}
