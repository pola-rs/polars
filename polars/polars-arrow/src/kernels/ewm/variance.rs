use std::ops::AddAssign;

use arrow::array::PrimitiveArray;
use arrow::types::NativeType;
use num::{Float, One};

use crate::trusted_len::TrustedLen;
use crate::utils::CustomIterTools;

pub fn ewm_std<I, T>(
    xs: I,
    alpha: T,
    adjust: bool,
    bias: bool,
    min_periods: usize,
) -> PrimitiveArray<T>
where
    I: IntoIterator<Item = Option<T>>,
    I::IntoIter: TrustedLen,
    T: Float + NativeType + AddAssign,
{
    let one_sub_alpha = T::one() - alpha;
    let two = T::one() + T::one();

    let mut opt_mean = None;
    let mut opt_var = None;
    let mut non_null_cnt = 0usize;

    let wgt = alpha;

    let (mut wgt_sum, mut wgt_sum_sqr) = if adjust {
        (T::zero(), T::zero())
    } else {
        // NOTE: we must ensure `wgt_sum` and `wgt_sum_sqr` are equal
        // to 1 during the first iteration
        (T::one(), (T::one() + alpha) / one_sub_alpha)
    };

    xs.into_iter()
        .map(|opt_x| {
            if let Some(x) = opt_x {
                non_null_cnt += 1;

                let prev_mean = opt_mean.unwrap_or(x);
                let prev_var = opt_var.unwrap_or_else(T::zero);

                wgt_sum = one_sub_alpha * wgt_sum + wgt;
                wgt_sum_sqr = one_sub_alpha.powf(two) * wgt_sum_sqr + wgt.powf(two);

                let curr_mean = prev_mean + (x - prev_mean) * wgt / wgt_sum;
                let curr_var = (T::one() - wgt / wgt_sum)
                    * (prev_var + wgt / wgt_sum * (x - prev_mean).powf(two));

                opt_mean = Some(curr_mean);
                opt_var = Some(curr_var);
            }
            match non_null_cnt < min_periods {
                true => None,
                false => opt_var.map(|var| {
                    // NOTE: the `non_null_cnt.is_one()` condition prevents a NaN
                    // from appearing in the first entry (it prevents a zero division)
                    let correction = if bias || non_null_cnt.is_one() {
                        T::one()
                    } else {
                        T::one() - wgt_sum_sqr / wgt_sum.powf(two)
                    };
                    (var / correction).sqrt()
                }),
            }
        })
        .collect_trusted()
}

pub fn ewm_var<I, T>(
    xs: I,
    alpha: T,
    adjust: bool,
    bias: bool,
    min_periods: usize,
) -> PrimitiveArray<T>
where
    I: IntoIterator<Item = Option<T>>,
    I::IntoIter: TrustedLen,
    T: Float + NativeType + AddAssign,
{
    let one_sub_alpha = T::one() - alpha;
    let two = T::one() + T::one();

    let mut opt_mean = None;
    let mut opt_var = None;
    let mut non_null_cnt = 0usize;

    let wgt = alpha;

    let (mut wgt_sum, mut wgt_sum_sqr) = if adjust {
        (T::zero(), T::zero())
    } else {
        // NOTE: we must ensure `wgt_sum` and `wgt_sum_sqr` are equal
        // to 1 during the first iteration
        (T::one(), (T::one() + alpha) / one_sub_alpha)
    };

    xs.into_iter()
        .map(|opt_x| {
            if let Some(x) = opt_x {
                non_null_cnt += 1;

                let prev_mean = opt_mean.unwrap_or(x);
                let prev_var = opt_var.unwrap_or_else(T::zero);

                wgt_sum = one_sub_alpha * wgt_sum + wgt;
                wgt_sum_sqr = one_sub_alpha.powf(two) * wgt_sum_sqr + wgt.powf(two);

                let curr_mean = prev_mean + (x - prev_mean) * wgt / wgt_sum;
                let curr_var = (T::one() - wgt / wgt_sum)
                    * (prev_var + wgt / wgt_sum * (x - prev_mean).powf(two));

                opt_mean = Some(curr_mean);
                opt_var = Some(curr_var);
            }
            match non_null_cnt < min_periods {
                true => None,
                false => opt_var.map(|var| {
                    // NOTE: the `non_null_cnt.is_one()` condition prevents a NaN
                    // from appearing in the first entry (it prevents a zero division)
                    let correction = if bias || non_null_cnt.is_one() {
                        T::one()
                    } else {
                        T::one() - wgt_sum_sqr / wgt_sum.powf(two)
                    };
                    var / correction
                }),
            }
        })
        .collect_trusted()
}

#[cfg(test)]
mod test {
    use super::*;

    const XS: [Option<f64>; 7] = [
        Some(1.0),
        Some(5.0),
        Some(7.0),
        Some(1.0),
        Some(2.0),
        Some(1.0),
        Some(4.0),
    ];
    const ALPHA: f64 = 0.5;

    #[test]
    fn test_emw_var_adjusted_biased() {
        let xs = Vec::from(XS);
        let polars_result = ewm_var(xs, ALPHA, true, true, 0);
        let pandas_result = PrimitiveArray::from([
            Some(0.0),
            Some(3.555555555555556),
            Some(4.244897959183674),
            Some(7.182222222222221),
            Some(3.796045785639958),
            Some(2.4671201814058956), // <-- pandas: 2.467120181405896
            Some(2.4760369520739043),
        ]);
        assert_eq!(polars_result, pandas_result);
    }

    #[test]
    fn test_emw_var_adjusted_unbiased() {
        let xs = Vec::from(XS);
        let polars_result = ewm_var(xs, ALPHA, true, false, 0);
        // NOTE: pandas actually returns `nan` for the first entry here, but that
        // is inconsistent with the other var calculations.
        let pandas_result = PrimitiveArray::from([
            Some(0.0),
            Some(8.000000000000002),  // <-- pandas: 8.0
            Some(7.42857142857143),   // <-- pandas: 7.428571428571429
            Some(11.542857142857141), // <-- pandas: 11.542857142857143
            Some(5.8838709677419345),
            Some(3.76036866359447),  // <-- pandas: 3.7603686635944706
            Some(3.743532058492689), // <-- pandas: 3.7435320584926886
        ]);
        assert_eq!(polars_result, pandas_result);
    }

    #[test]
    fn test_emw_var_unadjusted_biased() {
        let xs = Vec::from(XS);
        let polars_result = ewm_var(xs, ALPHA, false, true, 0);
        let pandas_result = PrimitiveArray::from([
            Some(0.0),
            Some(4.0),
            Some(6.0),
            Some(7.0),
            Some(3.75),
            Some(2.4375),
            Some(2.484375),
        ]);
        assert_eq!(polars_result, pandas_result);
    }

    #[test]
    fn test_emw_var_unadjusted_unbiased() {
        let xs = Vec::from(XS);
        let polars_result = ewm_var(xs, ALPHA, false, false, 0);
        // NOTE: pandas actually returns `nan` for the first entry here, but that
        // is inconsistent with the other var calculations.
        let pandas_result = PrimitiveArray::from([
            Some(0.0),
            Some(8.0),
            Some(9.6), // <-- pandas: 9.600000000000001
            Some(10.666666666666666),
            Some(5.647058823529412), // <-- pandas: 5.647058823529411
            Some(3.659824046920821),
            Some(3.7274725274725276),
        ]);
        assert_eq!(polars_result, pandas_result);
    }
}
