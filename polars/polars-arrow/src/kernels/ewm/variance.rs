use crate::kernels::ewm::EWMOptions;
use num::{Float, Zero};

pub fn ewm_std<T: Float>(xs: &[T], options: EWMOptions) -> Vec<T> {
    if options.adjust {
        ewm_var_adjusted(xs, options, true)
    } else {
        ewm_var_unadjusted(xs, options, true)
    }
}

pub fn ewm_var<T: Float>(xs: &[T], options: EWMOptions) -> Vec<T> {
    if options.adjust {
        ewm_var_adjusted(xs, options, false)
    } else {
        ewm_var_unadjusted(xs, options, false)
    }
}

/// Compute an adjusted, exponentially-weighted moving variance or standard deviation.
///
/// Sources:
///  - https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Weighted_incremental_algorithm
///  - https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average
fn ewm_var_adjusted<T: Float>(xs: &[T], options: EWMOptions, take_sqrt: bool) -> Vec<T> {
    let mut wgt_sum = T::zero();
    let mut wgt_sum_sqr = T::zero();

    let mut sigma_sqr = T::zero();
    let mut mu = T::zero();

    let alpha = T::from(options.alpha).unwrap();
    let one_sub_alpha = T::one() - alpha;

    let mut results = Vec::new();

    for (i, &x) in xs.iter().enumerate() {
        let pow = T::from(xs.len() - i - 1).unwrap();
        let wgt = alpha * one_sub_alpha.powf(pow);
        wgt_sum = wgt_sum + wgt;
        wgt_sum_sqr = wgt_sum_sqr + wgt * wgt;
        let mu_old = mu;
        mu = mu_old + (wgt / wgt_sum) * (x - mu_old);
        sigma_sqr = sigma_sqr + wgt * (x - mu_old) * (x - mu);

        let bias_correction = if options.bias {
            wgt_sum
        } else if i.is_zero() {
            // Prevent a NaN from cropping up in the first entry
            T::one()
        } else {
            wgt_sum - wgt_sum_sqr / wgt_sum
        };

        let mut result = sigma_sqr / bias_correction;
        if take_sqrt {
            result = result.sqrt();
        }
        results.push(result);
    }
    results
}

/// Compute an unadjusted, exponentially-weighted moving variance or standard deviation
///
/// Sources:
///  - https://web.archive.org/web/20181222175223/http://people.ds.cam.ac.uk/fanf2/hermes/doc/antiforgery/stats.pdf
///  - https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_variance
fn ewm_var_unadjusted<T: Float>(xs: &[T], options: EWMOptions, take_sqrt: bool) -> Vec<T> {
    let mut results = Vec::new();

    if !xs.is_empty() {
        let mut mean_prev = xs[0];
        let mut var_prev = T::zero();

        results.push(T::zero());

        let mut wgt = T::one();
        let mut wgt_sum_sqr = T::one();

        let alpha = T::from(options.alpha).unwrap();
        let one_sub_alpha = T::one() - alpha;
        let two = T::one() + T::one();

        for &x_i in xs.iter().skip(1) {
            let delta = x_i - mean_prev;
            let var_i = one_sub_alpha * (var_prev + alpha * delta.powf(two));

            let bias_correction = if options.bias {
                T::one()
            } else {
                wgt = wgt * one_sub_alpha;
                let correction = T::one() - (alpha * alpha * wgt_sum_sqr + wgt * wgt);
                wgt_sum_sqr = wgt_sum_sqr + wgt * wgt;
                correction
            };

            let mut result = var_i / bias_correction;
            if take_sqrt {
                result = result.sqrt()
            }
            results.push(result);

            var_prev = var_i;
            mean_prev = one_sub_alpha * mean_prev + alpha * x_i;
        }
    }
    results
}

#[cfg(test)]
mod test {
    use super::*;

    static XS: [f64; 7] = [1.0, 5.0, 7.0, 1.0, 2.0, 1.0, 4.0];
    static ALPHA: f64 = 0.5;
    static RTOL: f64 = 1e-12;

    /// `rtol` measures percent difference.
    fn _assert_approx_eq(x: &[f64], y: &[f64], rtol: f64) {
        if x.len() != y.len() {
            assert!(false, "x and y are different lengths")
        }

        for (xi, yi) in x.iter().zip(y) {
            if (xi - yi) / xi * 100. > rtol {
                // call assert_eq on the the whole slice so they get printed
                assert_eq!(x, y)
            }
        }
    }

    #[test]
    fn test_emw_var_adjusted_unbiased() {
        let options = EWMOptions {
            alpha: ALPHA,
            adjust: true,
            bias: false,
            min_periods: 0,
        };
        let polars_result = ewm_var(&XS, options);
        let pandas_result = [
            0.0,
            8.0,
            7.428571428571429,
            11.542857142857143,
            5.8838709677419345,
            3.7603686635944706,
            3.7435320584926886,
        ];
        _assert_approx_eq(polars_result.as_slice(), pandas_result.as_slice(), RTOL);
    }

    #[test]
    fn test_emw_var_adjusted_biased() {
        let options = EWMOptions {
            alpha: ALPHA,
            adjust: true,
            bias: true,
            min_periods: 0,
        };
        let polars_result = ewm_var(&XS, options);
        let pandas_result = [
            0.0,
            3.555555555555556,
            4.244897959183674,
            7.182222222222221,
            3.796045785639958,
            2.467120181405896,
            2.4760369520739043,
        ];
        _assert_approx_eq(polars_result.as_slice(), pandas_result.as_slice(), RTOL);
    }

    #[test]
    fn test_ewm_var_unadjusted_unbiased() {
        let options = EWMOptions {
            alpha: ALPHA,
            adjust: false,
            bias: false,
            min_periods: 0,
        };
        let polars_result = ewm_var(&XS, options);
        let pandas_result = [
            0.0,
            8.0,
            9.600000000000001,
            10.666666666666666,
            5.647058823529411,
            3.659824046920821,
            3.7274725274725276,
        ];
        _assert_approx_eq(polars_result.as_slice(), pandas_result.as_slice(), RTOL);
    }

    #[test]
    fn test_ewm_var_unadjusted_biased() {
        let options = EWMOptions {
            alpha: ALPHA,
            adjust: false,
            bias: true,
            min_periods: 0,
        };
        let polars_result = ewm_var(&XS, options);
        let pandas_result = [0.0, 4.0, 6.0, 7.0, 3.75, 2.4375, 2.484375];
        _assert_approx_eq(polars_result.as_slice(), pandas_result.as_slice(), RTOL);
    }

    #[test]
    fn test_ewm_var_all_zeros_f32() {
        let xs: [f32; 15] = [0.0; 15];
        // let xs: [f32; 8] = [0.0; 8];
        // let xs: [f64; 15] = [0.0; 15];
        for &adjust in [true, false].iter() {
            for &bias in [true, false].iter() {
                let options = EWMOptions {
                    alpha: 0.99951171875,
                    adjust,
                    bias,
                    min_periods: 0
                };
                let result = ewm_var(&xs, options);
                assert_eq!(&xs, result.as_slice())
            }
        }
    }
}
