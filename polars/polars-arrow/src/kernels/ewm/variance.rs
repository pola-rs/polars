use std::ops::{AddAssign, DivAssign};
use arrow::array::PrimitiveArray;
use arrow::types::NativeType;
use num::{Float, Zero};
use crate::utils::CustomIterTools;

// See: https://stats.stackexchange.com/a/111912/147321

pub fn ewm_std<T: Float>(x_vals: &[T], ewma_vals: &mut [T], alpha: T) {
    if !ewma_vals.is_empty() {
        let mut emwa_prev = ewma_vals[0];
        let mut emvar_prev = T::zero();
        ewma_vals[0] = T::zero();

        let one_sub_alpha = T::one() - alpha;
        let two = T::one() + T::one();

        let mut x_iter = x_vals.iter();
        x_iter.next();

        for (xi, ewma_i) in x_iter.zip(ewma_vals[1..].iter_mut()) {
            let delta_i = *xi - emwa_prev;
            emwa_prev = *ewma_i;

            emvar_prev = one_sub_alpha * (emvar_prev + alpha * delta_i.powf(two));
            *ewma_i = emvar_prev.sqrt();
        }
    }
}

pub fn ewm_var<T>(
    xs: &PrimitiveArray<T>,
    alpha: T,
    adjust: bool,
    bias: bool,
    min_periods: usize,
) -> PrimitiveArray<T>
where
    T: Float + NativeType + AddAssign + DivAssign,
{
    let one_sub_alpha = T::one() - alpha;

    let mut opt_mean_var = None;
    let mut non_null_cnt = 0usize;
    let mut wgt_sum = T::zero();
    let mut wgt_sum_sqr = T::zero();

    xs.iter()
        .map(|opt_x| {
            if let Some(&x) = opt_x {
                non_null_cnt += 1;

                let (prev_mean, prev_var) = opt_mean_var.unwrap_or((x, T::zero()));

                let pow = T::from(non_null_cnt - 1).unwrap();
                // TODO: introduce `ignore_na` parameter. We currently default to
                //  `ignore_na = True` but we can achieve `ignore_na = False` (for
                //  the adjusted case, at least) by setting `pow = T::from(i).unwrap();`
                let wgt = one_sub_alpha.powf(pow);

                wgt_sum += wgt;
                wgt_sum_sqr += wgt * wgt;

                let curr_mean = if adjust {
                    prev_mean + (x - prev_mean) / wgt_sum
                } else {
                    prev_mean + alpha * (x - prev_mean)
                };

                // NOTE: this is correct for unadjusted_biased -- ref: https://fanf2.user.srcf.net/hermes/doc/antiforgery/stats.pdf
                // let curr_var = one_sub_alpha * (prev_var + alpha * (x - prev_mean).powf(T::one() + T::one()));

                // NOTE: this is correct for adjusted_biased -- ref: personal notes
                let theta = wgt_sum;
                let theta_sub_one = theta - T::one();
                let mut curr_var = theta_sub_one / theta * prev_var + (theta_sub_one / theta / theta + theta_sub_one * theta_sub_one / theta / theta) * (x - prev_mean) * (x - prev_mean) / theta;

                opt_mean_var = Some((curr_mean, curr_var));
            }
            // NOTE: this is correct for adjusted_unbiased -- ref: personal notes
            let bias_correction = T::one() - wgt_sum_sqr / (wgt_sum * wgt_sum);
            match non_null_cnt < min_periods {
                true => None,
                false => opt_mean_var.map(|(_, var)| var / bias_correction),
            }
        })
        .collect_trusted()
}

#[cfg(test)]
mod test {
    use super::*;

    const XS: [Option<f64>; 7] = [Some(1.0), Some(5.0), Some(7.0), Some(1.0), Some(2.0), Some(1.0), Some(4.0)];
    const ALPHA: f64 = 0.5;

    #[test]
    fn test_emw_var_adjusted_biased() {
        let xs = PrimitiveArray::from(XS);
        let polars_result = ewm_var(&xs, ALPHA, true, true, 0);
        let pandas_result = PrimitiveArray::from([
            Some(0.0),
            Some(3.555555555555556),
            Some(4.244897959183674),
            Some(7.182222222222221),
            Some(3.796045785639958),
            Some(2.467120181405896),
            Some(2.4760369520739043),
        ]);
        assert_eq!(polars_result, pandas_result);
    }

    #[test]
    fn test_emw_var_adjusted_unbiased() {
        let xs = PrimitiveArray::from(XS);
        let polars_result = ewm_var(&xs, ALPHA, true, false, 0);
        // NOTE: pandas actually returns `nan` for the first entry here, but that
        // is inconsistent with the other var calculations.
        let pandas_result = PrimitiveArray::from([
            Some(0.0),
            Some(8.0),
            Some(7.428571428571429),
            Some(11.542857142857143),
            Some(5.8838709677419345),
            Some(3.7603686635944706),
            Some(3.7435320584926886),
        ]);
        assert_eq!(polars_result, pandas_result);
    }

        #[test]
    fn test_emw_var_unadjusted_biased() {
        let xs = PrimitiveArray::from(XS);
        let polars_result = ewm_var(&xs, ALPHA, false, true, 0);
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
        let xs = PrimitiveArray::from(XS);
        let polars_result = ewm_var(&xs, ALPHA, false, false, 0);
        // NOTE: pandas actually returns `nan` for the first entry here, but that
        // is inconsistent with the other var calculations.
        let pandas_result = PrimitiveArray::from([
            Some(0.0),
            Some(8.0),
            Some(9.600000000000001),
            Some(10.666666666666666),
            Some(5.647058823529411),
            Some(3.659824046920821),
            Some(3.7274725274725276),
        ]);
        assert_eq!(polars_result, pandas_result);
    }
}
