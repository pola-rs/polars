use super::*;

#[derive(Clone, Copy)]
enum CmpOp {
    Lt,
    Le,
    Gt,
    Ge,
}

// Given two monotonic functions f_a and f_d where f_a is ascending
// (f_a(x[0]) <= f_a(x[1]) <= .. <= f_a(x[n-1])) and f_d is descending
// (f_d(x[0]) >= f_d(x[1]) >= .. >= f_d(x[n-1])),
// outputs a mask where both are true.
//
// If a function is not given it is always assumed to be true. If invert is
// true the output mask is inverted.
fn bitonic_mask<T: PolarsNumericType>(
    ca: &ChunkedArray<T>,
    f_a: Option<CmpOp>,
    f_d: Option<CmpOp>,
    rhs: &T::Native,
    invert: bool,
) -> BooleanChunked {
    fn apply<T: PolarsNumericType>(op: CmpOp, x: T::Native, rhs: &T::Native) -> bool {
        match op {
            CmpOp::Lt => x.tot_lt(rhs),
            CmpOp::Le => x.tot_le(rhs),
            CmpOp::Gt => x.tot_gt(rhs),
            CmpOp::Ge => x.tot_ge(rhs),
        }
    }
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
        let length = arr.len();

        // Where the run of elements the two functions both hold at starts and ends. Every element
        // of a chunk whose values are stored in the scalar representation is the one value it
        // repeats — `full` builds exactly such a chunk and flags it sorted — so the two bounds
        // are read off that one value rather than searched for over values written out first.
        let (true_range_start, true_range_end) = match arr.scalar_values() {
            Some(value) => {
                let holds = f_a.is_none_or(|f_a| apply::<T>(f_a, value, rhs))
                    && f_d.is_none_or(|f_d| apply::<T>(f_d, value, rhs));

                if holds { (0, length) } else { (length, length) }
            },
            None => {
                let values = arr.flat_values().expect("the values are not repeated");
                let start = match f_a {
                    Some(f_a) => values.partition_point(|x| !apply::<T>(f_a, *x, rhs)),
                    None => 0,
                };
                let end = match f_d {
                    Some(f_d) => {
                        start + values[start..].partition_point(|x| apply::<T>(f_d, *x, rhs))
                    },
                    None => length,
                };

                (start, end)
            },
        };

        logical_extend(true_range_start, invert);
        logical_extend(true_range_end - true_range_start, !invert);
        logical_extend(length - true_range_end, invert);

        // The mask is three runs at most, so a chunk that falls entirely inside or entirely
        // outside the range says the same of every element: that is the single bit it repeats,
        // and it is not written out one bit per element.
        if true_range_start == 0 && true_range_end == length {
            return PlBooleanArray::new_scalar(!invert, length);
        }
        if true_range_start == true_range_end {
            return PlBooleanArray::new_scalar(invert, length);
        }

        let mut mask = BitmapBuilder::with_capacity(length);
        mask.extend_constant(true_range_start, invert);
        mask.extend_constant(true_range_end - true_range_start, !invert);
        mask.extend_constant(length - true_range_end, invert);
        PlBooleanArray::from_values(mask.freeze())
    });

    let mut ca = BooleanChunked::from_chunk_iter(ca.name().clone(), chunks);
    ca.set_sorted_flag(output_order.unwrap_or(IsSorted::Ascending));
    ca
}

impl<T, Rhs> ChunkCompareEq<Rhs> for ChunkedArray<T>
where
    T: PolarsNumericType,
    Rhs: ToPrimitive,
    Flat<T::Array>: TotalOrdKernel<Scalar = T::Native> + TotalEqKernel<Scalar = T::Native>,
{
    type Item = BooleanChunked;

    fn equal(&self, rhs: Rhs) -> BooleanChunked {
        let rhs: T::Native = NumCast::from(rhs).unwrap();
        let fa = Some(CmpOp::Ge);
        let fd = Some(CmpOp::Le);
        match (self.is_sorted_flag(), self.null_count()) {
            (IsSorted::Ascending, 0) => bitonic_mask(self, fa, fd, &rhs, false),
            (IsSorted::Descending, 0) => bitonic_mask(self, fd, fa, &rhs, false),
            _ => arity::unary_elementwise_mut_values_flat(self, |arr| {
                arr.tot_eq_kernel_broadcast(&rhs).into()
            }),
        }
    }

    fn equal_missing(&self, rhs: Rhs) -> BooleanChunked {
        if self.null_count() == 0 {
            self.equal(rhs)
        } else {
            let rhs: T::Native = NumCast::from(rhs).unwrap();
            arity::unary_elementwise_mut_with_options_flat(self, |arr| {
                arr.tot_eq_missing_kernel_broadcast(&rhs).into()
            })
        }
    }

    fn not_equal(&self, rhs: Rhs) -> BooleanChunked {
        let rhs: T::Native = NumCast::from(rhs).unwrap();
        let fa = Some(CmpOp::Ge);
        let fd = Some(CmpOp::Le);
        match (self.is_sorted_flag(), self.null_count()) {
            (IsSorted::Ascending, 0) => bitonic_mask(self, fa, fd, &rhs, true),
            (IsSorted::Descending, 0) => bitonic_mask(self, fd, fa, &rhs, true),
            _ => arity::unary_elementwise_mut_values_flat(self, |arr| {
                arr.tot_ne_kernel_broadcast(&rhs).into()
            }),
        }
    }

    fn not_equal_missing(&self, rhs: Rhs) -> BooleanChunked {
        if self.null_count() == 0 {
            self.not_equal(rhs)
        } else {
            let rhs: T::Native = NumCast::from(rhs).unwrap();
            arity::unary_elementwise_mut_with_options_flat(self, |arr| {
                arr.tot_ne_missing_kernel_broadcast(&rhs).into()
            })
        }
    }
}

impl<T, Rhs> ChunkCompareIneq<Rhs> for ChunkedArray<T>
where
    T: PolarsNumericType,
    Rhs: ToPrimitive,
    Flat<T::Array>: TotalOrdKernel<Scalar = T::Native> + TotalEqKernel<Scalar = T::Native>,
{
    type Item = BooleanChunked;

    fn gt(&self, rhs: Rhs) -> BooleanChunked {
        let rhs: T::Native = NumCast::from(rhs).unwrap();
        let fa = Some(CmpOp::Gt);
        let fd = None;
        match (self.is_sorted_flag(), self.null_count()) {
            (IsSorted::Ascending, 0) => bitonic_mask(self, fa, fd, &rhs, false),
            (IsSorted::Descending, 0) => bitonic_mask(self, fd, fa, &rhs, false),
            _ => arity::unary_elementwise_mut_values_flat(self, |arr| {
                arr.tot_gt_kernel_broadcast(&rhs).into()
            }),
        }
    }

    fn gt_eq(&self, rhs: Rhs) -> BooleanChunked {
        let rhs: T::Native = NumCast::from(rhs).unwrap();
        let fa = Some(CmpOp::Ge);
        let fd = None;
        match (self.is_sorted_flag(), self.null_count()) {
            (IsSorted::Ascending, 0) => bitonic_mask(self, fa, fd, &rhs, false),
            (IsSorted::Descending, 0) => bitonic_mask(self, fd, fa, &rhs, false),
            _ => arity::unary_elementwise_mut_values_flat(self, |arr| {
                arr.tot_ge_kernel_broadcast(&rhs).into()
            }),
        }
    }

    fn lt(&self, rhs: Rhs) -> BooleanChunked {
        let rhs: T::Native = NumCast::from(rhs).unwrap();
        let fa = None;
        let fd = Some(CmpOp::Lt);
        match (self.is_sorted_flag(), self.null_count()) {
            (IsSorted::Ascending, 0) => bitonic_mask(self, fa, fd, &rhs, false),
            (IsSorted::Descending, 0) => bitonic_mask(self, fd, fa, &rhs, false),
            _ => arity::unary_elementwise_mut_values_flat(self, |arr| {
                arr.tot_lt_kernel_broadcast(&rhs).into()
            }),
        }
    }

    fn lt_eq(&self, rhs: Rhs) -> BooleanChunked {
        let rhs: T::Native = NumCast::from(rhs).unwrap();
        let fa = None;
        let fd = Some(CmpOp::Le);
        match (self.is_sorted_flag(), self.null_count()) {
            (IsSorted::Ascending, 0) => bitonic_mask(self, fa, fd, &rhs, false),
            (IsSorted::Descending, 0) => bitonic_mask(self, fd, fa, &rhs, false),
            _ => arity::unary_elementwise_mut_values_flat(self, |arr| {
                arr.tot_le_kernel_broadcast(&rhs).into()
            }),
        }
    }
}

macro_rules! binary_eq_ineq_impl {
    ($($ca:ident),+) => {
        $(
        impl ChunkCompareEq<&[u8]> for $ca {
            type Item = BooleanChunked;

            fn equal(&self, rhs: &[u8]) -> BooleanChunked {
                arity::unary_elementwise_mut_values_flat(self, |arr| arr.tot_eq_kernel_broadcast(rhs).into())
            }

            fn equal_missing(&self, rhs: &[u8]) -> BooleanChunked {
                arity::unary_elementwise_mut_with_options_flat(self, |arr| arr.tot_eq_missing_kernel_broadcast(rhs).into())
            }

            fn not_equal(&self, rhs: &[u8]) -> BooleanChunked {
                arity::unary_elementwise_mut_values_flat(self, |arr| arr.tot_ne_kernel_broadcast(rhs).into())
            }

            fn not_equal_missing(&self, rhs: &[u8]) -> BooleanChunked {
                arity::unary_elementwise_mut_with_options_flat(self, |arr| arr.tot_ne_missing_kernel_broadcast(rhs).into())
            }
        }

        impl ChunkCompareIneq<&[u8]> for $ca {
            type Item = BooleanChunked;

            fn gt(&self, rhs: &[u8]) -> BooleanChunked {
                arity::unary_elementwise_mut_values_flat(self, |arr| arr.tot_gt_kernel_broadcast(rhs).into())
            }

            fn gt_eq(&self, rhs: &[u8]) -> BooleanChunked {
                arity::unary_elementwise_mut_values_flat(self, |arr| arr.tot_ge_kernel_broadcast(rhs).into())
            }

            fn lt(&self, rhs: &[u8]) -> BooleanChunked {
                arity::unary_elementwise_mut_values_flat(self, |arr| arr.tot_lt_kernel_broadcast(rhs).into())
            }

            fn lt_eq(&self, rhs: &[u8]) -> BooleanChunked {
                arity::unary_elementwise_mut_values_flat(self, |arr| arr.tot_le_kernel_broadcast(rhs).into())
            }
        }
        )+
    };
}

binary_eq_ineq_impl!(BinaryChunked, BinaryOffsetChunked);

impl ChunkCompareEq<&str> for StringChunked {
    type Item = BooleanChunked;

    fn equal(&self, rhs: &str) -> BooleanChunked {
        arity::unary_elementwise_mut_values_flat(self, |arr| {
            arr.tot_eq_kernel_broadcast(rhs).into()
        })
    }

    fn equal_missing(&self, rhs: &str) -> BooleanChunked {
        arity::unary_elementwise_mut_with_options_flat(self, |arr| {
            arr.tot_eq_missing_kernel_broadcast(rhs).into()
        })
    }

    fn not_equal(&self, rhs: &str) -> BooleanChunked {
        arity::unary_elementwise_mut_values_flat(self, |arr| {
            arr.tot_ne_kernel_broadcast(rhs).into()
        })
    }

    fn not_equal_missing(&self, rhs: &str) -> BooleanChunked {
        arity::unary_elementwise_mut_with_options_flat(self, |arr| {
            arr.tot_ne_missing_kernel_broadcast(rhs).into()
        })
    }
}

impl ChunkCompareIneq<&str> for StringChunked {
    type Item = BooleanChunked;

    fn gt(&self, rhs: &str) -> BooleanChunked {
        arity::unary_elementwise_mut_values_flat(self, |arr| {
            arr.tot_gt_kernel_broadcast(rhs).into()
        })
    }

    fn gt_eq(&self, rhs: &str) -> BooleanChunked {
        arity::unary_elementwise_mut_values_flat(self, |arr| {
            arr.tot_ge_kernel_broadcast(rhs).into()
        })
    }

    fn lt(&self, rhs: &str) -> BooleanChunked {
        arity::unary_elementwise_mut_values_flat(self, |arr| {
            arr.tot_lt_kernel_broadcast(rhs).into()
        })
    }

    fn lt_eq(&self, rhs: &str) -> BooleanChunked {
        arity::unary_elementwise_mut_values_flat(self, |arr| {
            arr.tot_le_kernel_broadcast(rhs).into()
        })
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_binary_search_cmp() {
        let mut s = Series::new(PlSmallStr::EMPTY, &[1, 1, 2, 2, 4, 8]);
        s.set_sorted_flag(IsSorted::Ascending);
        let out = s.gt(10).unwrap();
        assert!(!out.any());

        let out = s.gt(0).unwrap();
        assert!(out.all());

        let out = s.gt(2).unwrap();
        assert_eq!(
            out.into_series(),
            Series::new(PlSmallStr::EMPTY, [false, false, false, false, true, true])
        );
        let out = s.gt(3).unwrap();
        assert_eq!(
            out.into_series(),
            Series::new(PlSmallStr::EMPTY, [false, false, false, false, true, true])
        );

        let out = s.gt_eq(10).unwrap();
        assert!(!out.any());
        let out = s.gt_eq(0).unwrap();
        assert!(out.all());

        let out = s.gt_eq(2).unwrap();
        assert_eq!(
            out.into_series(),
            Series::new(PlSmallStr::EMPTY, [false, false, true, true, true, true])
        );
        let out = s.gt_eq(3).unwrap();
        assert_eq!(
            out.into_series(),
            Series::new(PlSmallStr::EMPTY, [false, false, false, false, true, true])
        );

        let out = s.lt(10).unwrap();
        assert!(out.all());
        let out = s.lt(0).unwrap();
        assert!(!out.any());

        let out = s.lt(2).unwrap();
        assert_eq!(
            out.into_series(),
            Series::new(PlSmallStr::EMPTY, [true, true, false, false, false, false])
        );
        let out = s.lt(3).unwrap();
        assert_eq!(
            out.into_series(),
            Series::new(PlSmallStr::EMPTY, [true, true, true, true, false, false])
        );

        let out = s.lt_eq(10).unwrap();
        assert!(out.all());
        let out = s.lt_eq(0).unwrap();
        assert!(!out.any());

        let out = s.lt_eq(2).unwrap();
        assert_eq!(
            out.into_series(),
            Series::new(PlSmallStr::EMPTY, [true, true, true, true, false, false])
        );
        let out = s.lt(3).unwrap();
        assert_eq!(
            out.into_series(),
            Series::new(PlSmallStr::EMPTY, [true, true, true, true, false, false])
        );
    }
}
