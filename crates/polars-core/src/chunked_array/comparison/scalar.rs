use super::*;

// Given two monotonic functions f_a and f_d where f_a is ascending
// (f_a(x[0]) <= f_a(x[1]) <= .. <= f_a(x[n-1])) and f_d is descending
// (f_d(x[0]) >= f_d(x[1]) >= .. >= f_d(x[n-1])),
// outputs a mask where both are true.
//
// If a function is not given it is always assumed to be true. If invert is
// true the output mask is inverted.
fn bitonic_mask<T: PolarsNumericType, FA, FD>(
    ca: &ChunkedArray<T>,
    f_a: Option<FA>,
    f_d: Option<FD>,
    invert: bool,
) -> BooleanChunked
where
    FA: Fn(T::Native) -> bool,
    FD: Fn(T::Native) -> bool,
{
    let mut output_order: Option<IsSorted> = None;
    let mut last_value: Option<bool> = None;
    let mut logical_extend = |len: usize, val: bool| {
        if len != 0 {
            if let Some(last_value) = last_value {
                output_order = match (last_value, val, output_order) {
                    (false, true, None) => Some(IsSorted::Ascending),
                    (false, true, _) => Some(IsSorted::Not),
                    (true, false, None) => Some(IsSorted::Descending),
                    (true, false, _) => Some(IsSorted::Not),
                    _ => output_order,
                };
            }
            last_value = Some(val);
        }
    };

    let chunks = ca.downcast_iter().map(|arr| {
        let values = arr.values();
        let true_range_start = if let Some(f_a) = f_a.as_ref() {
            values.partition_point(|x| !f_a(*x))
        } else {
            0
        };
        let true_range_end = if let Some(f_d) = f_d.as_ref() {
            true_range_start + values[true_range_start..].partition_point(|x| f_d(*x))
        } else {
            values.len()
        };
        let mut mask = MutableBitmap::with_capacity(arr.len());
        mask.extend_constant(true_range_start, invert);
        mask.extend_constant(true_range_end - true_range_start, !invert);
        mask.extend_constant(arr.len() - true_range_end, invert);
        logical_extend(true_range_start, invert);
        logical_extend(true_range_end - true_range_start, !invert);
        logical_extend(arr.len() - true_range_end, invert);
        BooleanArray::from_data_default(mask.into(), None)
    });

    let mut ca = BooleanChunked::from_chunk_iter(ca.name(), chunks);
    ca.set_sorted_flag(output_order.unwrap_or(IsSorted::Ascending));
    ca
}

impl<T, Rhs> ChunkCompare<Rhs> for ChunkedArray<T>
where
    T: PolarsNumericType,
    Rhs: ToPrimitive,
    T::Array: TotalOrdKernel<Scalar = T::Native> + TotalEqKernel<Scalar = T::Native>,
{
    type Item = BooleanChunked;
    fn equal(&self, rhs: Rhs) -> BooleanChunked {
        let rhs: T::Native = NumCast::from(rhs).unwrap();
        let fa = Some(|x: T::Native| x.tot_ge(&rhs));
        let fd = Some(|x: T::Native| x.tot_le(&rhs));
        match (self.is_sorted_flag(), self.null_count()) {
            (IsSorted::Ascending, 0) => bitonic_mask(self, fa, fd, false),
            (IsSorted::Descending, 0) => bitonic_mask(self, fd, fa, false),
            _ => arity::unary_mut_values(self, |arr| arr.tot_eq_kernel_broadcast(&rhs).into()),
        }
    }

    fn equal_missing(&self, rhs: Rhs) -> BooleanChunked {
        if self.null_count() == 0 {
            self.equal(rhs)
        } else {
            let rhs: T::Native = NumCast::from(rhs).unwrap();
            arity::unary_mut_with_options(self, |arr| {
                arr.tot_eq_missing_kernel_broadcast(&rhs).into()
            })
        }
    }

    fn not_equal(&self, rhs: Rhs) -> BooleanChunked {
        let rhs: T::Native = NumCast::from(rhs).unwrap();
        let fa = Some(|x: T::Native| x.tot_ge(&rhs));
        let fd = Some(|x: T::Native| x.tot_le(&rhs));
        match (self.is_sorted_flag(), self.null_count()) {
            (IsSorted::Ascending, 0) => bitonic_mask(self, fa, fd, true),
            (IsSorted::Descending, 0) => bitonic_mask(self, fd, fa, true),
            _ => arity::unary_mut_values(self, |arr| arr.tot_ne_kernel_broadcast(&rhs).into()),
        }
    }

    fn not_equal_missing(&self, rhs: Rhs) -> BooleanChunked {
        if self.null_count() == 0 {
            self.not_equal(rhs)
        } else {
            let rhs: T::Native = NumCast::from(rhs).unwrap();
            arity::unary_mut_with_options(self, |arr| {
                arr.tot_ne_missing_kernel_broadcast(&rhs).into()
            })
        }
    }

    fn gt(&self, rhs: Rhs) -> BooleanChunked {
        let rhs: T::Native = NumCast::from(rhs).unwrap();
        let fa = Some(|x: T::Native| x.tot_gt(&rhs));
        let fd: Option<fn(_) -> _> = None;
        match (self.is_sorted_flag(), self.null_count()) {
            (IsSorted::Ascending, 0) => bitonic_mask(self, fa, fd, false),
            (IsSorted::Descending, 0) => bitonic_mask(self, fd, fa, false),
            _ => arity::unary_mut_values(self, |arr| arr.tot_gt_kernel_broadcast(&rhs).into()),
        }
    }

    fn gt_eq(&self, rhs: Rhs) -> BooleanChunked {
        let rhs: T::Native = NumCast::from(rhs).unwrap();
        let fa = Some(|x: T::Native| x.tot_ge(&rhs));
        let fd: Option<fn(_) -> _> = None;
        match (self.is_sorted_flag(), self.null_count()) {
            (IsSorted::Ascending, 0) => bitonic_mask(self, fa, fd, false),
            (IsSorted::Descending, 0) => bitonic_mask(self, fd, fa, false),
            _ => arity::unary_mut_values(self, |arr| arr.tot_ge_kernel_broadcast(&rhs).into()),
        }
    }

    fn lt(&self, rhs: Rhs) -> BooleanChunked {
        let rhs: T::Native = NumCast::from(rhs).unwrap();
        let fa: Option<fn(_) -> _> = None;
        let fd = Some(|x: T::Native| x.tot_lt(&rhs));
        match (self.is_sorted_flag(), self.null_count()) {
            (IsSorted::Ascending, 0) => bitonic_mask(self, fa, fd, false),
            (IsSorted::Descending, 0) => bitonic_mask(self, fd, fa, false),
            _ => arity::unary_mut_values(self, |arr| arr.tot_lt_kernel_broadcast(&rhs).into()),
        }
    }

    fn lt_eq(&self, rhs: Rhs) -> BooleanChunked {
        let rhs: T::Native = NumCast::from(rhs).unwrap();
        let fa: Option<fn(_) -> _> = None;
        let fd = Some(|x: T::Native| x.tot_le(&rhs));
        match (self.is_sorted_flag(), self.null_count()) {
            (IsSorted::Ascending, 0) => bitonic_mask(self, fa, fd, false),
            (IsSorted::Descending, 0) => bitonic_mask(self, fd, fa, false),
            _ => arity::unary_mut_values(self, |arr| arr.tot_le_kernel_broadcast(&rhs).into()),
        }
    }
}

impl ChunkCompare<&[u8]> for BinaryChunked {
    type Item = BooleanChunked;

    fn equal(&self, rhs: &[u8]) -> BooleanChunked {
        arity::unary_mut_values(self, |arr| arr.tot_eq_kernel_broadcast(rhs).into())
    }

    fn equal_missing(&self, rhs: &[u8]) -> BooleanChunked {
        arity::unary_mut_with_options(self, |arr| arr.tot_eq_missing_kernel_broadcast(rhs).into())
    }

    fn not_equal(&self, rhs: &[u8]) -> BooleanChunked {
        arity::unary_mut_values(self, |arr| arr.tot_ne_kernel_broadcast(rhs).into())
    }

    fn not_equal_missing(&self, rhs: &[u8]) -> BooleanChunked {
        arity::unary_mut_with_options(self, |arr| arr.tot_ne_missing_kernel_broadcast(rhs).into())
    }

    fn gt(&self, rhs: &[u8]) -> BooleanChunked {
        arity::unary_mut_values(self, |arr| arr.tot_gt_kernel_broadcast(rhs).into())
    }

    fn gt_eq(&self, rhs: &[u8]) -> BooleanChunked {
        arity::unary_mut_values(self, |arr| arr.tot_ge_kernel_broadcast(rhs).into())
    }

    fn lt(&self, rhs: &[u8]) -> BooleanChunked {
        arity::unary_mut_values(self, |arr| arr.tot_lt_kernel_broadcast(rhs).into())
    }

    fn lt_eq(&self, rhs: &[u8]) -> BooleanChunked {
        arity::unary_mut_values(self, |arr| arr.tot_le_kernel_broadcast(rhs).into())
    }
}

impl ChunkCompare<&str> for StringChunked {
    type Item = BooleanChunked;

    fn equal(&self, rhs: &str) -> BooleanChunked {
        arity::unary_mut_values(self, |arr| arr.tot_eq_kernel_broadcast(rhs).into())
    }

    fn equal_missing(&self, rhs: &str) -> BooleanChunked {
        arity::unary_mut_with_options(self, |arr| arr.tot_eq_missing_kernel_broadcast(rhs).into())
    }

    fn not_equal(&self, rhs: &str) -> BooleanChunked {
        arity::unary_mut_values(self, |arr| arr.tot_ne_kernel_broadcast(rhs).into())
    }

    fn not_equal_missing(&self, rhs: &str) -> BooleanChunked {
        arity::unary_mut_with_options(self, |arr| arr.tot_ne_missing_kernel_broadcast(rhs).into())
    }

    fn gt(&self, rhs: &str) -> BooleanChunked {
        arity::unary_mut_values(self, |arr| arr.tot_gt_kernel_broadcast(rhs).into())
    }

    fn gt_eq(&self, rhs: &str) -> BooleanChunked {
        arity::unary_mut_values(self, |arr| arr.tot_ge_kernel_broadcast(rhs).into())
    }

    fn lt(&self, rhs: &str) -> BooleanChunked {
        arity::unary_mut_values(self, |arr| arr.tot_lt_kernel_broadcast(rhs).into())
    }

    fn lt_eq(&self, rhs: &str) -> BooleanChunked {
        arity::unary_mut_values(self, |arr| arr.tot_le_kernel_broadcast(rhs).into())
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
