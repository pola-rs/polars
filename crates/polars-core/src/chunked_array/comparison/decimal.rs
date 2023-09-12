use crate::prelude::*;

impl ChunkCompare<&DecimalChunked> for DecimalChunked
{
    type Item = BooleanChunked;

    fn equal(&self, rhs: &DecimalChunked) -> Self::Item {
        self.0.equal(&rhs.0)
    }

    fn equal_missing(&self, rhs: &DecimalChunked) -> Self::Item {
        self.0.equal_missing(&rhs.0)
    }

    fn not_equal(&self, rhs: &DecimalChunked) -> Self::Item {
        self.0.not_equal(&rhs.0)
    }

    fn not_equal_missing(&self, rhs: &DecimalChunked) -> Self::Item {
        self.0.not_equal_missing(&rhs.0)
    }

    fn gt(&self, rhs: &DecimalChunked) -> Self::Item {
        self.0.gt(&rhs.0)
    }

    fn gt_eq(&self, rhs: &DecimalChunked) -> Self::Item {
        self.0.gt_eq(&rhs.0)
    }

    fn lt(&self, rhs: &DecimalChunked) -> Self::Item {
        self.0.lt(&rhs.0)
    }

    fn lt_eq(&self, rhs: &DecimalChunked) -> Self::Item {
        self.0.lt_eq(&rhs.0)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    macro_rules! assert_cmp {
        ($precision:expr, $scale:expr, $a:expr, $b:expr, $cmp:expr, $chunk_cmp:expr) => {
            {
                // apply the decimal chunked comparison
                let actual_chunked = $chunk_cmp(&to_chunk($a, $precision, $scale), &to_chunk($b, $precision, $scale));
                
                let (a, b) = if $a.len() < $b.len() {
                    assert_eq!($a.len(), 1);
                    (vec![$a[0]; $b.len()], $b.to_vec()) // duplicate the unique `a` element b.len() times
                } else if $a.len() > $b.len() {
                    assert_eq!($b.len(), 1);
                    ($a.to_vec(), vec![$b[0]; $a.len()]) // duplicate the unique `b` element a.len() times
                } else {
                    ($a.to_vec(), $b.to_vec())
                };

                // apply the primitive i128 comparison
                let expected_values: Vec<Option<bool>> = a.into_iter().zip(b.into_iter()).map($cmp).map(|v| v.into()).collect();

                // both primitive and decimal chunked comparisons should yield the same result
                assert_eq!(Vec::from(&actual_chunked), expected_values);
            }
        };
    }

    fn to_chunk(v: &[i128], precision: usize, scale: usize) -> DecimalChunked {
        Int128Chunked::from_vec("", v.into())
                .into_decimal_unchecked(Some(precision), scale)
    }

    fn sample_decimal_chunked(precision: usize) -> [Vec<i128>; 3] {
        let max_prec = 10_i128.pow(precision as u32) - 1;
        let a = vec![-7_i128];
        let b = vec![max_prec, -max_prec, 1_i128, 0, -7, 9];
        let c = vec![max_prec, max_prec, -1, 0, 1, max_prec];

        [a, b, c]
    }

    #[test]
    fn test_decimal_equal() {
        let (precision, scale) = (38, 0);
        let [a, b, c] = sample_decimal_chunked(precision);
        let i128_cmp = |(x, y)| x == y;
        let decimal_cmp = |chunk_x: &DecimalChunked, chunk_y: &DecimalChunked| chunk_x.equal(chunk_y);
        let decimal_missing_cmp = |chunk_x: &DecimalChunked, chunk_y: &DecimalChunked| chunk_x.equal_missing(chunk_y);
        for (x, y) in [(&a, &b), (&a, &c), (&b, &c)] {
            assert_cmp!(precision, scale, x, y, i128_cmp, decimal_cmp);
            assert_cmp!(precision, scale, y, x, i128_cmp, decimal_cmp);
            assert_cmp!(precision, scale, x, y, i128_cmp, decimal_missing_cmp);
            assert_cmp!(precision, scale, y, x, i128_cmp, decimal_missing_cmp);
        }
    }

    #[test]
    fn test_decimal_not_equal() {
        let (precision, scale) = (5, 1);
        let [a, b, c] = sample_decimal_chunked(precision);
        let i128_cmp = |(x, y)| x != y;
        let decimal_cmp = |chunk_x: &DecimalChunked, chunk_y: &DecimalChunked| chunk_x.not_equal(chunk_y);
        let decimal_missing_cmp = |chunk_x: &DecimalChunked, chunk_y: &DecimalChunked| chunk_x.not_equal_missing(chunk_y);
        for (x, y) in [(&a, &b), (&a, &c), (&b, &c)] {
            assert_cmp!(precision, scale, x, y, i128_cmp, decimal_cmp);
            assert_cmp!(precision, scale, y, x, i128_cmp, decimal_cmp);
            assert_cmp!(precision, scale, x, y, i128_cmp, decimal_missing_cmp);
            assert_cmp!(precision, scale, y, x, i128_cmp, decimal_missing_cmp);
        }
    }

    #[test]
    fn test_decimal_lt() {
        let (precision, scale) = (38, 0);
        let [a, b, c] = sample_decimal_chunked(precision);
        let i128_cmp = |(x, y)| x < y;
        let decimal_cmp = |chunk_x: &DecimalChunked, chunk_y: &DecimalChunked| chunk_x.lt(chunk_y);
        for (x, y) in [(&a, &b), (&a, &c), (&b, &c)] {
            assert_cmp!(precision, scale, x, y, i128_cmp, decimal_cmp);
            assert_cmp!(precision, scale, y, x, i128_cmp, decimal_cmp);
        }
    }

    #[test]
    fn test_decimal_lt_eq() {
        let (precision, scale) = (32, 5);
        let [a, b, c] = sample_decimal_chunked(precision);
        let i128_cmp = |(x, y)| x <= y;
        let decimal_cmp = |chunk_x: &DecimalChunked, chunk_y: &DecimalChunked| chunk_x.lt_eq(chunk_y);
        for (x, y) in [(&a, &b), (&a, &c), (&b, &c)] {
            assert_cmp!(precision, scale, x, y, i128_cmp, decimal_cmp);
            assert_cmp!(precision, scale, y, x, i128_cmp, decimal_cmp);
        }
    }

    #[test]
    fn test_decimal_gt() {
        let (precision, scale) = (25, 3);
        let [a, b, c] = sample_decimal_chunked(precision);
        let i128_cmp = |(x, y)| x > y;
        let decimal_cmp = |chunk_x: &DecimalChunked, chunk_y: &DecimalChunked| chunk_x.gt(chunk_y);
        for (x, y) in [(&a, &b), (&a, &c), (&b, &c)] {
            assert_cmp!(precision, scale, x, y, i128_cmp, decimal_cmp);
            assert_cmp!(precision, scale, y, x, i128_cmp, decimal_cmp);
        }
    }

    #[test]
    fn test_decimal_gt_eq() {
        let (precision, scale) = (15, 3);
        let [a, b, c] = sample_decimal_chunked(precision);
        let i128_cmp = |(x, y)| x >= y;
        let decimal_cmp = |chunk_x: &DecimalChunked, chunk_y: &DecimalChunked| chunk_x.gt_eq(chunk_y);
        for (x, y) in [(&a, &b), (&a, &c), (&b, &c)] {
            assert_cmp!(precision, scale, x, y, i128_cmp, decimal_cmp);
            assert_cmp!(precision, scale, y, x, i128_cmp, decimal_cmp);
        }
    }

    #[test]
    #[should_panic]
    fn comparison_with_non_unitary_arrays_of_different_sizes_will_fail() {
        let (precision, scale) = (15, 3);
        to_chunk(&[1, 2, 3], precision, scale).gt(&to_chunk(&[1, 2], precision, scale));
    }

    #[test]
    #[should_panic]
    fn comparison_with_decimals_having_different_precision_will_fail() {
        let (precision0, precision1, scale) = (15, 14, 3);
        to_chunk(&[1, 2, 3], precision0, scale).gt(&to_chunk(&[1, 2], precision1, scale));
    }

    #[test]
    #[should_panic]
    fn comparison_with_decimals_having_different_scale_will_fail() {
        let (precision, scale0, scale1) = (20, 0, 3);
        to_chunk(&[1, 2], precision, scale0).gt(&to_chunk(&[1, 2], precision, scale1));
    }
}
