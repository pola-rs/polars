use std::ops::{AddAssign, DivAssign, MulAssign};

use arrow::array::PrimitiveArray;
use arrow::types::NativeType;
use num::Float;

use crate::trusted_len::TrustedLen;
use crate::utils::CustomIterTools;

#[allow(clippy::too_many_arguments)]
fn ewm_cov_internal<I, T>(
    xs: I,
    ys: I,
    alpha: T,
    adjust: bool,
    bias: bool,
    min_periods: usize,
    ignore_nulls: bool,
    do_sqrt: bool,
) -> PrimitiveArray<T>
where
    I: IntoIterator<Item = Option<T>>,
    I::IntoIter: TrustedLen,
    T: Float + NativeType + AddAssign + MulAssign + DivAssign,
{
    let old_wt_factor = T::one() - alpha;
    let new_wt = if adjust { T::one() } else { alpha };
    let mut sum_wt = T::one();
    let mut sum_wt2 = T::one();
    let mut old_wt = T::one();

    let mut opt_mean_x = None;
    let mut opt_mean_y = None;
    let mut cov = T::zero();
    let mut non_na_cnt = 0usize;
    let min_periods_fixed = if min_periods == 0 { 1 } else { min_periods };

    let res = xs
        .into_iter()
        .zip(ys.into_iter())
        .enumerate()
        .map(|(i, (opt_x, opt_y))| {
            let is_observation = opt_x.is_some() && opt_y.is_some();
            if is_observation {
                non_na_cnt += 1;
            }
            match (i, opt_mean_x, opt_mean_y) {
                (0, _, _) => {
                    if is_observation {
                        opt_mean_x = opt_x;
                        opt_mean_y = opt_y;
                    }
                }
                (_, Some(mean_x), Some(mean_y)) => {
                    if is_observation || !ignore_nulls {
                        sum_wt *= old_wt_factor;
                        sum_wt2 *= old_wt_factor * old_wt_factor;
                        old_wt *= old_wt_factor;
                        if is_observation {
                            let x = opt_x.unwrap();
                            let y = opt_y.unwrap();
                            let old_mean_x = mean_x;
                            let old_mean_y = mean_y;

                            // avoid numerical errors on constant series
                            if mean_x != x {
                                opt_mean_x =
                                    Some((old_wt * old_mean_x + new_wt * x) / (old_wt + new_wt));
                            }

                            // avoid numerical errors on constant series
                            if mean_y != y {
                                opt_mean_y =
                                    Some((old_wt * old_mean_y + new_wt * y) / (old_wt + new_wt));
                            }

                            cov = ((old_wt
                                * (cov
                                    + ((old_mean_x - opt_mean_x.unwrap())
                                        * (old_mean_y - opt_mean_y.unwrap()))))
                                + (new_wt
                                    * ((x - opt_mean_x.unwrap()) * (y - opt_mean_y.unwrap()))))
                                / (old_wt + new_wt);

                            sum_wt += new_wt;
                            sum_wt2 += new_wt * new_wt;
                            old_wt += new_wt;
                            if !adjust {
                                sum_wt /= old_wt;
                                sum_wt2 /= old_wt * old_wt;
                                old_wt = T::one();
                            }
                        }
                    }
                }
                _ => {
                    if is_observation {
                        opt_mean_x = opt_x;
                        opt_mean_y = opt_y;
                    }
                }
            }
            match (non_na_cnt >= min_periods_fixed, bias) {
                (false, _) => None,
                (true, false) => {
                    if non_na_cnt == 1 {
                        Some(cov)
                    } else {
                        let numerator = sum_wt * sum_wt;
                        let denominator = numerator - sum_wt2;
                        if denominator > T::zero() {
                            Some((numerator / denominator) * cov)
                        } else {
                            None
                        }
                    }
                }
                (true, true) => Some(cov),
            }
        });

    if do_sqrt {
        res.map(|opt_x| opt_x.map(|x| x.sqrt())).collect_trusted()
    } else {
        res.collect_trusted()
    }
}

pub fn ewm_cov<I, T>(
    xs: I,
    ys: I,
    alpha: T,
    adjust: bool,
    bias: bool,
    min_periods: usize,
    ignore_nulls: bool,
) -> PrimitiveArray<T>
where
    I: IntoIterator<Item = Option<T>>,
    I::IntoIter: TrustedLen,
    T: Float + NativeType + AddAssign + MulAssign + DivAssign,
{
    ewm_cov_internal(
        xs,
        ys,
        alpha,
        adjust,
        bias,
        min_periods,
        ignore_nulls,
        false,
    )
}

pub fn ewm_var<I, T>(
    xs: I,
    alpha: T,
    adjust: bool,
    bias: bool,
    min_periods: usize,
    ignore_nulls: bool,
) -> PrimitiveArray<T>
where
    I: IntoIterator<Item = Option<T>> + Clone,
    I::IntoIter: TrustedLen,
    T: Float + NativeType + AddAssign + MulAssign + DivAssign,
{
    ewm_cov_internal(
        xs.clone(),
        xs,
        alpha,
        adjust,
        bias,
        min_periods,
        ignore_nulls,
        false,
    )
}

pub fn ewm_std<I, T>(
    xs: I,
    alpha: T,
    adjust: bool,
    bias: bool,
    min_periods: usize,
    ignore_nulls: bool,
) -> PrimitiveArray<T>
where
    I: IntoIterator<Item = Option<T>> + Clone,
    I::IntoIter: TrustedLen,
    T: Float + NativeType + AddAssign + MulAssign + DivAssign,
{
    ewm_cov_internal(
        xs.clone(),
        xs,
        alpha,
        adjust,
        bias,
        min_periods,
        ignore_nulls,
        true,
    )
}

#[cfg(test)]
mod test {
    use super::super::assert_allclose;
    use super::*;
    const ALPHA: f64 = 0.5;
    const EPS: f64 = 1e-15;

    const XS: [Option<f64>; 7] = [
        Some(1.0),
        Some(5.0),
        Some(7.0),
        Some(1.0),
        Some(2.0),
        Some(1.0),
        Some(4.0),
    ];
    const YS: [Option<f64>; 7] = [None, Some(5.0), Some(7.0), None, None, Some(1.0), Some(4.0)];

    #[test]
    fn test_ewm_var() {
        assert_allclose!(
            ewm_var(XS.to_vec(), ALPHA, true, true, 0, true),
            PrimitiveArray::from([
                Some(0.00000000000000000000),
                Some(3.55555555555555580227),
                Some(4.24489795918367374128),
                Some(7.18222222222222139720),
                Some(3.79604578563995787022),
                Some(2.46712018140589606219),
                Some(2.47603695207390428479),
            ]),
            EPS
        );
        assert_allclose!(
            ewm_var(XS.to_vec(), ALPHA, true, true, 0, false),
            PrimitiveArray::from([
                Some(0.00000000000000000000),
                Some(3.55555555555555580227),
                Some(4.24489795918367374128),
                Some(7.18222222222222139720),
                Some(3.79604578563995787022),
                Some(2.46712018140589606219),
                Some(2.47603695207390428479),
            ]),
            EPS
        );
        assert_allclose!(
            ewm_var(XS.to_vec(), ALPHA, true, false, 0, true),
            PrimitiveArray::from([
                Some(0.00000000000000000000),
                Some(8.00000000000000000000),
                Some(7.42857142857142882519),
                Some(11.54285714285714270488),
                Some(5.88387096774193452120),
                Some(3.76036866359447063957),
                Some(3.74353205849268855232),
            ]),
            EPS
        );
        assert_allclose!(
            ewm_var(XS.to_vec(), ALPHA, true, false, 0, false),
            PrimitiveArray::from([
                Some(0.00000000000000000000),
                Some(8.00000000000000000000),
                Some(7.42857142857142882519),
                Some(11.54285714285714270488),
                Some(5.88387096774193452120),
                Some(3.76036866359447063957),
                Some(3.74353205849268855232),
            ]),
            EPS
        );
        assert_allclose!(
            ewm_var(XS.to_vec(), ALPHA, false, true, 0, true),
            PrimitiveArray::from([
                Some(0.00000000000000000000),
                Some(4.00000000000000000000),
                Some(6.00000000000000000000),
                Some(7.00000000000000000000),
                Some(3.75000000000000000000),
                Some(2.43750000000000000000),
                Some(2.48437500000000000000),
            ]),
            EPS
        );
        assert_allclose!(
            ewm_var(XS.to_vec(), ALPHA, false, true, 0, false),
            PrimitiveArray::from([
                Some(0.00000000000000000000),
                Some(4.00000000000000000000),
                Some(6.00000000000000000000),
                Some(7.00000000000000000000),
                Some(3.75000000000000000000),
                Some(2.43750000000000000000),
                Some(2.48437500000000000000),
            ]),
            EPS
        );
        assert_allclose!(
            ewm_var(XS.to_vec(), ALPHA, false, true, 0, false),
            PrimitiveArray::from([
                Some(0.00000000000000000000),
                Some(4.00000000000000000000),
                Some(6.00000000000000000000),
                Some(7.00000000000000000000),
                Some(3.75000000000000000000),
                Some(2.43750000000000000000),
                Some(2.48437500000000000000),
            ]),
            EPS
        );
        assert_allclose!(
            ewm_var(XS.to_vec(), ALPHA, false, false, 0, true),
            PrimitiveArray::from([
                Some(0.00000000000000000000),
                Some(8.00000000000000000000),
                Some(9.60000000000000142109),
                Some(10.66666666666666607455),
                Some(5.64705882352941124225),
                Some(3.65982404692082097242),
                Some(3.72747252747252755256),
            ]),
            EPS
        );
        assert_allclose!(
            ewm_var(XS.to_vec(), ALPHA, false, false, 0, false),
            PrimitiveArray::from([
                Some(0.00000000000000000000),
                Some(8.00000000000000000000),
                Some(9.60000000000000142109),
                Some(10.66666666666666607455),
                Some(5.64705882352941124225),
                Some(3.65982404692082097242),
                Some(3.72747252747252755256),
            ]),
            EPS
        );
        assert_allclose!(
            ewm_var(YS.to_vec(), ALPHA, true, true, 0, true),
            PrimitiveArray::from([
                None,
                Some(0.00000000000000000000),
                Some(0.88888888888888895057),
                Some(0.88888888888888895057),
                Some(0.88888888888888895057),
                Some(7.34693877551020335659),
                Some(3.55555555555555535818),
            ]),
            EPS
        );
        assert_allclose!(
            ewm_var(YS.to_vec(), ALPHA, true, true, 0, false),
            PrimitiveArray::from([
                None,
                Some(0.00000000000000000000),
                Some(0.88888888888888895057),
                Some(0.88888888888888895057),
                Some(0.88888888888888895057),
                Some(3.92243767313019331411),
                Some(2.54978854286812728347),
            ]),
            EPS
        );
        assert_allclose!(
            ewm_var(YS.to_vec(), ALPHA, true, false, 0, true),
            PrimitiveArray::from([
                None,
                Some(0.00000000000000000000),
                Some(2.00000000000000000000),
                Some(2.00000000000000000000),
                Some(2.00000000000000000000),
                Some(12.85714285714285587403),
                Some(5.71428571428571441260),
            ]),
            EPS
        );
        assert_allclose!(
            ewm_var(YS.to_vec(), ALPHA, true, false, 0, false),
            PrimitiveArray::from([
                None,
                Some(0.00000000000000000000),
                Some(2.00000000000000000000),
                Some(2.00000000000000000000),
                Some(2.00000000000000000000),
                Some(14.15999999999999658939),
                Some(5.03951367781154946357),
            ]),
            EPS
        );
        assert_allclose!(
            ewm_var(YS.to_vec(), ALPHA, false, true, 0, true),
            PrimitiveArray::from([
                None,
                Some(0.00000000000000000000),
                Some(1.00000000000000000000),
                Some(1.00000000000000000000),
                Some(1.00000000000000000000),
                Some(6.75000000000000000000),
                Some(3.43750000000000000000),
            ]),
            EPS
        );
        assert_allclose!(
            ewm_var(YS.to_vec(), ALPHA, false, true, 0, false),
            PrimitiveArray::from([
                None,
                Some(0.00000000000000000000),
                Some(1.00000000000000000000),
                Some(1.00000000000000000000),
                Some(1.00000000000000000000),
                Some(4.20000000000000017764),
                Some(3.10000000000000008882),
            ]),
            EPS
        );
        assert_allclose!(
            ewm_var(YS.to_vec(), ALPHA, false, false, 0, true),
            PrimitiveArray::from([
                None,
                Some(0.00000000000000000000),
                Some(2.00000000000000000000),
                Some(2.00000000000000000000),
                Some(2.00000000000000000000),
                Some(10.80000000000000071054),
                Some(5.23809523809523813753),
            ]),
            EPS
        );
        assert_allclose!(
            ewm_var(YS.to_vec(), ALPHA, false, false, 0, false),
            PrimitiveArray::from([
                None,
                Some(0.00000000000000000000),
                Some(2.00000000000000000000),
                Some(2.00000000000000000000),
                Some(2.00000000000000000000),
                Some(12.35294117647058875775),
                Some(5.29914529914529985888),
            ]),
            EPS
        );
    }

    #[test]
    fn test_ewm_cov() {
        assert_allclose!(
            ewm_cov(XS.to_vec(), YS.to_vec(), ALPHA, true, true, 0, true),
            PrimitiveArray::from([
                None,
                Some(0.00000000000000000000),
                Some(0.88888888888888895057),
                Some(0.88888888888888895057),
                Some(0.88888888888888895057),
                Some(7.34693877551020335659),
                Some(3.55555555555555535818)
            ]),
            EPS
        );
        assert_allclose!(
            ewm_cov(XS.to_vec(), YS.to_vec(), ALPHA, true, true, 0, false),
            PrimitiveArray::from([
                None,
                Some(0.00000000000000000000),
                Some(0.88888888888888895057),
                Some(0.88888888888888895057),
                Some(0.88888888888888895057),
                Some(3.92243767313019331411),
                Some(2.54978854286812728347)
            ]),
            EPS
        );
        assert_allclose!(
            ewm_cov(XS.to_vec(), YS.to_vec(), ALPHA, true, false, 0, true),
            PrimitiveArray::from([
                None,
                Some(0.00000000000000000000),
                Some(2.00000000000000000000),
                Some(2.00000000000000000000),
                Some(2.00000000000000000000),
                Some(12.85714285714285587403),
                Some(5.71428571428571441260)
            ]),
            EPS
        );
        assert_allclose!(
            ewm_cov(XS.to_vec(), YS.to_vec(), ALPHA, true, false, 0, false),
            PrimitiveArray::from([
                None,
                Some(0.00000000000000000000),
                Some(2.00000000000000000000),
                Some(2.00000000000000000000),
                Some(2.00000000000000000000),
                Some(14.15999999999999658939),
                Some(5.03951367781154946357)
            ]),
            EPS
        );
        assert_allclose!(
            ewm_cov(XS.to_vec(), YS.to_vec(), ALPHA, false, true, 0, true),
            PrimitiveArray::from([
                None,
                Some(0.00000000000000000000),
                Some(1.00000000000000000000),
                Some(1.00000000000000000000),
                Some(1.00000000000000000000),
                Some(6.75000000000000000000),
                Some(3.43750000000000000000)
            ]),
            EPS
        );
        assert_allclose!(
            ewm_cov(XS.to_vec(), YS.to_vec(), ALPHA, false, true, 0, false),
            PrimitiveArray::from([
                None,
                Some(0.00000000000000000000),
                Some(1.00000000000000000000),
                Some(1.00000000000000000000),
                Some(1.00000000000000000000),
                Some(4.20000000000000017764),
                Some(3.10000000000000008882)
            ]),
            EPS
        );
        assert_allclose!(
            ewm_cov(XS.to_vec(), YS.to_vec(), ALPHA, false, false, 0, true),
            PrimitiveArray::from([
                None,
                Some(0.00000000000000000000),
                Some(2.00000000000000000000),
                Some(2.00000000000000000000),
                Some(2.00000000000000000000),
                Some(10.80000000000000071054),
                Some(5.23809523809523813753)
            ]),
            EPS
        );
        assert_allclose!(
            ewm_cov(XS.to_vec(), YS.to_vec(), ALPHA, false, false, 0, false),
            PrimitiveArray::from([
                None,
                Some(0.00000000000000000000),
                Some(2.00000000000000000000),
                Some(2.00000000000000000000),
                Some(2.00000000000000000000),
                Some(12.35294117647058875775),
                Some(5.29914529914529985888)
            ]),
            EPS
        );
    }

    #[test]
    fn test_ewm_std() {
        assert_allclose!(
            ewm_std(XS.to_vec(), ALPHA, true, true, 0, true),
            PrimitiveArray::from([
                Some(0.00000000000000000000),
                Some(1.88561808316412671260),
                Some(2.06031501455085130914),
                Some(2.67996683229890386713),
                Some(1.94834437039245145229),
                Some(1.57070690499720422295),
                Some(1.57354280274605318191),
            ]),
            EPS
        );
        assert_allclose!(
            ewm_std(XS.to_vec(), ALPHA, true, true, 0, false),
            PrimitiveArray::from([
                Some(0.00000000000000000000),
                Some(1.88561808316412671260),
                Some(2.06031501455085130914),
                Some(2.67996683229890386713),
                Some(1.94834437039245145229),
                Some(1.57070690499720422295),
                Some(1.57354280274605318191),
            ]),
            EPS
        );
        assert_allclose!(
            ewm_std(XS.to_vec(), ALPHA, true, false, 0, true),
            PrimitiveArray::from([
                Some(0.00000000000000000000),
                Some(2.82842712474619029095),
                Some(2.72554057547698747044),
                Some(3.39747805627308530063),
                Some(2.42566917936925907640),
                Some(1.93916700250248452697),
                Some(1.93482093706179658632),
            ]),
            EPS
        );
        assert_allclose!(
            ewm_std(XS.to_vec(), ALPHA, true, false, 0, false),
            PrimitiveArray::from([
                Some(0.00000000000000000000),
                Some(2.82842712474619029095),
                Some(2.72554057547698747044),
                Some(3.39747805627308530063),
                Some(2.42566917936925907640),
                Some(1.93916700250248452697),
                Some(1.93482093706179658632),
            ]),
            EPS
        );
        assert_allclose!(
            ewm_std(XS.to_vec(), ALPHA, false, true, 0, true),
            PrimitiveArray::from([
                Some(0.00000000000000000000),
                Some(2.00000000000000000000),
                Some(2.44948974278317788134),
                Some(2.64575131106459071617),
                Some(1.93649167310370851069),
                Some(1.56124949959959957724),
                Some(1.57619002661481144578),
            ]),
            EPS
        );
        assert_allclose!(
            ewm_std(XS.to_vec(), ALPHA, false, true, 0, false),
            PrimitiveArray::from([
                Some(0.00000000000000000000),
                Some(2.00000000000000000000),
                Some(2.44948974278317788134),
                Some(2.64575131106459071617),
                Some(1.93649167310370851069),
                Some(1.56124949959959957724),
                Some(1.57619002661481144578),
            ]),
            EPS
        );
        assert_allclose!(
            ewm_std(XS.to_vec(), ALPHA, false, false, 0, true),
            PrimitiveArray::from([
                Some(0.00000000000000000000),
                Some(2.82842712474619029095),
                Some(3.09838667696593361711),
                Some(3.26598632371090413784),
                Some(2.37635410314401829268),
                Some(1.91306666034428118905),
                Some(1.93066634286521066066),
            ]),
            EPS
        );
        assert_allclose!(
            ewm_std(XS.to_vec(), ALPHA, false, false, 0, false),
            PrimitiveArray::from([
                Some(0.00000000000000000000),
                Some(2.82842712474619029095),
                Some(3.09838667696593361711),
                Some(3.26598632371090413784),
                Some(2.37635410314401829268),
                Some(1.91306666034428118905),
                Some(1.93066634286521066066),
            ]),
            EPS
        );
        assert_allclose!(
            ewm_std(YS.to_vec(), ALPHA, true, true, 0, true),
            PrimitiveArray::from([
                None,
                Some(0.00000000000000000000),
                Some(0.94280904158206335630),
                Some(0.94280904158206335630),
                Some(0.94280904158206335630),
                Some(2.71052370871575343259),
                Some(1.88561808316412671260),
            ]),
            EPS
        );
        assert_allclose!(
            ewm_std(YS.to_vec(), ALPHA, true, true, 0, false),
            PrimitiveArray::from([
                None,
                Some(0.00000000000000000000),
                Some(0.94280904158206335630),
                Some(0.94280904158206335630),
                Some(0.94280904158206335630),
                Some(1.98051449707650295551),
                Some(1.59680573109822199207),
            ]),
            EPS
        );
        assert_allclose!(
            ewm_std(YS.to_vec(), ALPHA, true, false, 0, true),
            PrimitiveArray::from([
                None,
                Some(0.00000000000000000000),
                Some(1.41421356237309514547),
                Some(1.41421356237309514547),
                Some(1.41421356237309514547),
                Some(3.58568582800318091941),
                Some(2.39045721866878713158),
            ]),
            EPS
        );
        assert_allclose!(
            ewm_std(YS.to_vec(), ALPHA, true, false, 0, false),
            PrimitiveArray::from([
                None,
                Some(0.00000000000000000000),
                Some(1.41421356237309514547),
                Some(1.41421356237309514547),
                Some(1.41421356237309514547),
                Some(3.76297754444535526019),
                Some(2.24488611689135586502),
            ]),
            EPS
        );
        assert_allclose!(
            ewm_std(YS.to_vec(), ALPHA, false, true, 0, true),
            PrimitiveArray::from([
                None,
                Some(0.00000000000000000000),
                Some(1.00000000000000000000),
                Some(1.00000000000000000000),
                Some(1.00000000000000000000),
                Some(2.59807621135331601181),
                Some(1.85404962177391574585),
            ]),
            EPS
        );
        assert_allclose!(
            ewm_std(YS.to_vec(), ALPHA, false, true, 0, false),
            PrimitiveArray::from([
                None,
                Some(0.00000000000000000000),
                Some(1.00000000000000000000),
                Some(1.00000000000000000000),
                Some(1.00000000000000000000),
                Some(2.04939015319191986109),
                Some(1.76068168616590092768),
            ]),
            EPS
        );
        assert_allclose!(
            ewm_std(YS.to_vec(), ALPHA, false, false, 0, true),
            PrimitiveArray::from([
                None,
                Some(0.00000000000000000000),
                Some(1.41421356237309514547),
                Some(1.41421356237309514547),
                Some(1.41421356237309514547),
                Some(3.28633534503099689061),
                Some(2.28868854108531749603),
            ]),
            EPS
        );
        assert_allclose!(
            ewm_std(YS.to_vec(), ALPHA, false, false, 0, false),
            PrimitiveArray::from([
                None,
                Some(0.00000000000000000000),
                Some(1.41421356237309514547),
                Some(1.41421356237309514547),
                Some(1.41421356237309514547),
                Some(3.51467511677403665615),
                Some(2.30198724999625037313),
            ]),
            EPS
        );
    }

    #[test]
    fn test_ewm_min_periods() {
        assert_allclose!(
            ewm_var(YS.to_vec(), ALPHA, true, true, 0, false),
            PrimitiveArray::from([
                None,
                Some(0.00000000000000000000),
                Some(0.88888888888888895057),
                Some(0.88888888888888895057),
                Some(0.88888888888888895057),
                Some(3.92243767313019331411),
                Some(2.54978854286812728347),
            ]),
            EPS
        );
        assert_allclose!(
            ewm_var(YS.to_vec(), ALPHA, true, true, 1, false),
            PrimitiveArray::from([
                None,
                Some(0.00000000000000000000),
                Some(0.88888888888888895057),
                Some(0.88888888888888895057),
                Some(0.88888888888888895057),
                Some(3.92243767313019331411),
                Some(2.54978854286812728347),
            ]),
            EPS
        );
        assert_allclose!(
            ewm_var(YS.to_vec(), ALPHA, true, true, 2, false),
            PrimitiveArray::from([
                None,
                None,
                Some(0.88888888888888895057),
                Some(0.88888888888888895057),
                Some(0.88888888888888895057),
                Some(3.92243767313019331411),
                Some(2.54978854286812728347),
            ]),
            EPS
        );
    }
}
