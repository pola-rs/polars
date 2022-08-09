use num::Float;


pub fn ewm_std<T: Float>(xs: &[T], result: &mut [T], alpha: T, adjust: bool, bias: bool) {
    if adjust {
        ewm_var_adjusted(xs, result, alpha, bias, true)
    } else {
        ewm_var_unadjusted(xs, result, alpha, bias, true)
    }
}

pub fn ewm_var<T: Float>(xs: &[T], result: &mut [T], alpha: T, adjust: bool, bias: bool) {
    if adjust {
        ewm_var_adjusted(xs, result, alpha, bias, false)
    } else {
        ewm_var_unadjusted(xs, result, alpha, bias, false)
    }
}

/// Compute an adjusted, exponentially-weighted moving variance or standard deviation.
///
/// Sources:
///  - https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Weighted_incremental_algorithm
///  - https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average
fn ewm_var_adjusted<T: Float>(xs: &[T], result: &mut [T], alpha: T, bias: bool, take_sqrt: bool) {
    let mut wgt_sum = T::zero();
    let mut wgt_sum_sqr = T::zero();
    let mut var = T::zero();
    let mut mean = T::zero();

    let one_sub_alpha = T::one() - alpha;

    let n = T::from(xs.len()).unwrap() - T::one();
    let mut i = T::zero();

    for &x in xs.iter() {
        let wgt = alpha * one_sub_alpha.powf(n - i);
        wgt_sum = wgt_sum + wgt;
        wgt_sum_sqr = wgt_sum_sqr + wgt * wgt;
        let mean_old = mean;
        mean = mean_old + (wgt / wgt_sum) * (x - mean_old);
        var = var + wgt * (x - mean_old) * (x - mean);

        let bias_correction = if bias {
            wgt_sum
        } else if i == T::zero() {
            // Prevent a NaN from cropping up in the first entry
            T::one()
        } else {
            wgt_sum - wgt_sum_sqr / wgt_sum
        };

        result[i.to_usize().unwrap()] = if take_sqrt {
            (var / bias_correction).sqrt()
        } else {
            var / bias_correction
        };

        i = i + T::one();
    }
}


/// Compute an unadjusted, exponentially-weighted moving variance or standard deviation
///
/// Sources:
///  - https://web.archive.org/web/20181222175223/http://people.ds.cam.ac.uk/fanf2/hermes/doc/antiforgery/stats.pdf
///  - https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_variance
fn ewm_var_unadjusted<T: Float>(xs: &[T], result: &mut [T], alpha: T, bias: bool, take_sqrt: bool) {
    if !xs.is_empty() {
        let mut mean_prev = xs[0];
        let mut var_prev = T::zero();
        result[0] = T::zero();

        let mut wgt = T::one();
        let mut wgt_sum_sqr = T::one();

        let one_sub_alpha = T::one() - alpha;
        let two = T::one() + T::one();

        for (i, &x_i) in xs.iter().skip(1).enumerate() {
            let delta = x_i - mean_prev;
            let var_i = one_sub_alpha * (var_prev + alpha * delta.powf(two));

            let bias_correction = if bias {
                T::one()
            } else {
                wgt = wgt * one_sub_alpha;
                let correction = T::one() - (alpha * alpha * wgt_sum_sqr + wgt * wgt);
                wgt_sum_sqr = wgt_sum_sqr + wgt * wgt;
                correction
            };

            result[i + 1] = if take_sqrt {
                (var_i / bias_correction).sqrt()
            } else {
                var_i / bias_correction
            };
            var_prev = var_i;
            mean_prev = one_sub_alpha * mean_prev + alpha * x_i;
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    static XS: [f64; 7] = [1.0, 5.0, 7.0, 1.0, 2.0, 1.0, 4.0];
    static ALPHA: f64 = 0.5;

    #[test]
    fn test_emw_var_adjusted_unbiased() {
        let mut polars_result: [f64; 7] = Default::default();
        ewm_var(&XS, &mut polars_result, ALPHA, true, false);
        let pandas_result = [
            0.0,
            8.0,
            7.428571428571429,
            11.542857142857143,
            5.8838709677419345,
            3.7603686635944706,
            3.7435320584926886,
        ];
        assert_eq!(polars_result, pandas_result);
    }

    #[test]
    fn test_emw_var_adjusted_biased() {
        let mut polars_result: [f64; 7] = Default::default();
        ewm_var(&XS, &mut polars_result, ALPHA, true, true);
        let pandas_result = [
            0.0,
            3.555555555555556,
            4.244897959183674,
            7.182222222222221,
            3.796045785639958,
            2.467120181405896,
            2.4760369520739043,
        ];
        assert_eq!(polars_result, pandas_result);
    }

    #[test]
    fn test_ewm_var_unadjusted_unbiased() {
        let mut polars_result: [f64; 7] = Default::default();
        ewm_var(&XS, &mut polars_result, ALPHA, false, false);
        let pandas_result = [
            0.0,
            8.0,
            9.600000000000001,
            10.666666666666666,
            5.647058823529411,
            3.659824046920821,
            3.7274725274725276,
        ];
        assert_eq!(polars_result, pandas_result);
    }

    #[test]
    fn test_ewm_var_unadjusted_biased() {
        let mut polars_result: [f64; 7] = Default::default();
        ewm_var(&XS, &mut polars_result, ALPHA, false, true);
        let pandas_result = [
            0.0,
            4.0,
            6.0,
            7.0,
            3.75,
            2.4375,
            2.484375,
        ];
        assert_eq!(polars_result, pandas_result);
    }
}
