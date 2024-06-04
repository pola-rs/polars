use std::ops::{AddAssign, DivAssign, MulAssign};

use num_traits::Float;

use crate::array::PrimitiveArray;
use crate::legacy::utils::CustomIterTools;
use crate::trusted_len::TrustedLen;
use crate::types::NativeType;

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
        .zip(ys)
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
                },
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
                },
                _ => {
                    if is_observation {
                        opt_mean_x = opt_x;
                        opt_mean_y = opt_y;
                    }
                },
            }
            match (non_na_cnt >= min_periods_fixed, bias, is_observation) {
                (_, _, false) => None,
                (false, _, true) => None,
                (true, false, true) => {
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
                },
                (true, true, true) => Some(cov),
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
    use std::f64::consts::SQRT_2;

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
                Some(0.0),
                Some(3.555_555_555_555_556),
                Some(4.244_897_959_183_674),
                Some(7.182_222_222_222_221),
                Some(3.796_045_785_639_958),
                Some(2.467_120_181_405_896),
                Some(2.476_036_952_073_904_3),
            ]),
            EPS
        );
        assert_allclose!(
            ewm_var(XS.to_vec(), ALPHA, true, true, 0, false),
            PrimitiveArray::from([
                Some(0.0),
                Some(3.555_555_555_555_556),
                Some(4.244_897_959_183_674),
                Some(7.182_222_222_222_221),
                Some(3.796_045_785_639_958),
                Some(2.467_120_181_405_896),
                Some(2.476_036_952_073_904_3),
            ]),
            EPS
        );
        assert_allclose!(
            ewm_var(XS.to_vec(), ALPHA, true, false, 0, true),
            PrimitiveArray::from([
                Some(0.0),
                Some(8.0),
                Some(7.428_571_428_571_429),
                Some(11.542_857_142_857_143),
                Some(5.883_870_967_741_934_5),
                Some(3.760_368_663_594_470_6),
                Some(3.743_532_058_492_688_6),
            ]),
            EPS
        );
        assert_allclose!(
            ewm_var(XS.to_vec(), ALPHA, true, false, 0, false),
            PrimitiveArray::from([
                Some(0.0),
                Some(8.0),
                Some(7.428_571_428_571_429),
                Some(11.542_857_142_857_143),
                Some(5.883_870_967_741_934_5),
                Some(3.760_368_663_594_470_6),
                Some(3.743_532_058_492_688_6),
            ]),
            EPS
        );
        assert_allclose!(
            ewm_var(XS.to_vec(), ALPHA, false, true, 0, true),
            PrimitiveArray::from([
                Some(0.0),
                Some(4.0),
                Some(6.0),
                Some(7.0),
                Some(3.75),
                Some(2.437_5),
                Some(2.484_375),
            ]),
            EPS
        );
        assert_allclose!(
            ewm_var(XS.to_vec(), ALPHA, false, true, 0, false),
            PrimitiveArray::from([
                Some(0.0),
                Some(4.0),
                Some(6.0),
                Some(7.0),
                Some(3.75),
                Some(2.437_5),
                Some(2.484_375),
            ]),
            EPS
        );
        assert_allclose!(
            ewm_var(XS.to_vec(), ALPHA, false, true, 0, false),
            PrimitiveArray::from([
                Some(0.0),
                Some(4.0),
                Some(6.0),
                Some(7.0),
                Some(3.75),
                Some(2.437_5),
                Some(2.484_375),
            ]),
            EPS
        );
        assert_allclose!(
            ewm_var(XS.to_vec(), ALPHA, false, false, 0, true),
            PrimitiveArray::from([
                Some(0.0),
                Some(8.0),
                Some(9.600_000_000_000_001),
                Some(10.666_666_666_666_666),
                Some(5.647_058_823_529_411),
                Some(3.659_824_046_920_821),
                Some(3.727_472_527_472_527_6),
            ]),
            EPS
        );
        assert_allclose!(
            ewm_var(XS.to_vec(), ALPHA, false, false, 0, false),
            PrimitiveArray::from([
                Some(0.0),
                Some(8.0),
                Some(9.600_000_000_000_001),
                Some(10.666_666_666_666_666),
                Some(5.647_058_823_529_411),
                Some(3.659_824_046_920_821),
                Some(3.727_472_527_472_527_6),
            ]),
            EPS
        );
        assert_allclose!(
            ewm_var(YS.to_vec(), ALPHA, true, true, 0, true),
            PrimitiveArray::from([
                None,
                Some(0.0),
                Some(0.888_888_888_888_889),
                None,
                None,
                Some(7.346_938_775_510_203),
                Some(3.555_555_555_555_555_4),
            ]),
            EPS
        );
        assert_allclose!(
            ewm_var(YS.to_vec(), ALPHA, true, true, 0, false),
            PrimitiveArray::from([
                None,
                Some(0.0),
                Some(0.888_888_888_888_889),
                None,
                None,
                Some(3.922_437_673_130_193_3),
                Some(2.549_788_542_868_127_3),
            ]),
            EPS
        );
        assert_allclose!(
            ewm_var(YS.to_vec(), ALPHA, true, false, 0, true),
            PrimitiveArray::from([
                None,
                Some(0.0),
                Some(2.0),
                None,
                None,
                Some(12.857_142_857_142_856),
                Some(5.714_285_714_285_714),
            ]),
            EPS
        );
        assert_allclose!(
            ewm_var(YS.to_vec(), ALPHA, true, false, 0, false),
            PrimitiveArray::from([
                None,
                Some(0.0),
                Some(2.0),
                None,
                None,
                Some(14.159_999_999_999_997),
                Some(5.039_513_677_811_549_5),
            ]),
            EPS
        );
        assert_allclose!(
            ewm_var(YS.to_vec(), ALPHA, false, true, 0, true),
            PrimitiveArray::from([
                None,
                Some(0.0),
                Some(1.0),
                None,
                None,
                Some(6.75),
                Some(3.437_5),
            ]),
            EPS
        );
        assert_allclose!(
            ewm_var(YS.to_vec(), ALPHA, false, true, 0, false),
            PrimitiveArray::from([None, Some(0.0), Some(1.0), None, None, Some(4.2), Some(3.1),]),
            EPS
        );
        assert_allclose!(
            ewm_var(YS.to_vec(), ALPHA, false, false, 0, true),
            PrimitiveArray::from([
                None,
                Some(0.0),
                Some(2.0),
                None,
                None,
                Some(10.8),
                Some(5.238_095_238_095_238),
            ]),
            EPS
        );
        assert_allclose!(
            ewm_var(YS.to_vec(), ALPHA, false, false, 0, false),
            PrimitiveArray::from([
                None,
                Some(0.0),
                Some(2.0),
                None,
                None,
                Some(12.352_941_176_470_589),
                Some(5.299_145_299_145_3),
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
                Some(0.0),
                Some(0.888_888_888_888_889),
                None,
                None,
                Some(7.346_938_775_510_203),
                Some(3.555_555_555_555_555_4)
            ]),
            EPS
        );
        assert_allclose!(
            ewm_cov(XS.to_vec(), YS.to_vec(), ALPHA, true, true, 0, false),
            PrimitiveArray::from([
                None,
                Some(0.0),
                Some(0.888_888_888_888_889),
                None,
                None,
                Some(3.922_437_673_130_193_3),
                Some(2.549_788_542_868_127_3)
            ]),
            EPS
        );
        assert_allclose!(
            ewm_cov(XS.to_vec(), YS.to_vec(), ALPHA, true, false, 0, true),
            PrimitiveArray::from([
                None,
                Some(0.0),
                Some(2.0),
                None,
                None,
                Some(12.857_142_857_142_856),
                Some(5.714_285_714_285_714)
            ]),
            EPS
        );
        assert_allclose!(
            ewm_cov(XS.to_vec(), YS.to_vec(), ALPHA, true, false, 0, false),
            PrimitiveArray::from([
                None,
                Some(0.0),
                Some(2.0),
                None,
                None,
                Some(14.159_999_999_999_997),
                Some(5.039_513_677_811_549_5)
            ]),
            EPS
        );
        assert_allclose!(
            ewm_cov(XS.to_vec(), YS.to_vec(), ALPHA, false, true, 0, true),
            PrimitiveArray::from([
                None,
                Some(0.0),
                Some(1.0),
                None,
                None,
                Some(6.75),
                Some(3.437_5)
            ]),
            EPS
        );
        assert_allclose!(
            ewm_cov(XS.to_vec(), YS.to_vec(), ALPHA, false, true, 0, false),
            PrimitiveArray::from([None, Some(0.0), Some(1.0), None, None, Some(4.2), Some(3.1)]),
            EPS
        );
        assert_allclose!(
            ewm_cov(XS.to_vec(), YS.to_vec(), ALPHA, false, false, 0, true),
            PrimitiveArray::from([
                None,
                Some(0.0),
                Some(2.0),
                None,
                None,
                Some(10.8),
                Some(5.238_095_238_095_238)
            ]),
            EPS
        );
        assert_allclose!(
            ewm_cov(XS.to_vec(), YS.to_vec(), ALPHA, false, false, 0, false),
            PrimitiveArray::from([
                None,
                Some(0.0),
                Some(2.0),
                None,
                None,
                Some(12.352_941_176_470_589),
                Some(5.299_145_299_145_3)
            ]),
            EPS
        );
    }

    #[test]
    fn test_ewm_std() {
        assert_allclose!(
            ewm_std(XS.to_vec(), ALPHA, true, true, 0, true),
            PrimitiveArray::from([
                Some(0.0),
                Some(1.885_618_083_164_126_7),
                Some(2.060_315_014_550_851_3),
                Some(2.679_966_832_298_904),
                Some(1.948_344_370_392_451_5),
                Some(1.570_706_904_997_204_2),
                Some(1.573_542_802_746_053_2),
            ]),
            EPS
        );
        assert_allclose!(
            ewm_std(XS.to_vec(), ALPHA, true, true, 0, false),
            PrimitiveArray::from([
                Some(0.0),
                Some(1.885_618_083_164_126_7),
                Some(2.060_315_014_550_851_3),
                Some(2.679_966_832_298_904),
                Some(1.948_344_370_392_451_5),
                Some(1.570_706_904_997_204_2),
                Some(1.573_542_802_746_053_2),
            ]),
            EPS
        );
        assert_allclose!(
            ewm_std(XS.to_vec(), ALPHA, true, false, 0, true),
            PrimitiveArray::from([
                Some(0.0),
                Some(2.828_427_124_746_190_3),
                Some(2.725_540_575_476_987_5),
                Some(3.397_478_056_273_085_3),
                Some(2.425_669_179_369_259),
                Some(1.939_167_002_502_484_5),
                Some(1.934_820_937_061_796_6),
            ]),
            EPS
        );
        assert_allclose!(
            ewm_std(XS.to_vec(), ALPHA, true, false, 0, false),
            PrimitiveArray::from([
                Some(0.0),
                Some(2.828_427_124_746_190_3),
                Some(2.725_540_575_476_987_5),
                Some(3.397_478_056_273_085_3),
                Some(2.425_669_179_369_259),
                Some(1.939_167_002_502_484_5),
                Some(1.934_820_937_061_796_6),
            ]),
            EPS
        );
        assert_allclose!(
            ewm_std(XS.to_vec(), ALPHA, false, true, 0, true),
            PrimitiveArray::from([
                Some(0.0),
                Some(2.0),
                Some(2.449_489_742_783_178),
                Some(2.645_751_311_064_590_7),
                Some(1.936_491_673_103_708_5),
                Some(1.561_249_499_599_599_6),
                Some(1.576_190_026_614_811_4),
            ]),
            EPS
        );
        assert_allclose!(
            ewm_std(XS.to_vec(), ALPHA, false, true, 0, false),
            PrimitiveArray::from([
                Some(0.0),
                Some(2.0),
                Some(2.449_489_742_783_178),
                Some(2.645_751_311_064_590_7),
                Some(1.936_491_673_103_708_5),
                Some(1.561_249_499_599_599_6),
                Some(1.576_190_026_614_811_4),
            ]),
            EPS
        );
        assert_allclose!(
            ewm_std(XS.to_vec(), ALPHA, false, false, 0, true),
            PrimitiveArray::from([
                Some(0.0),
                Some(2.828_427_124_746_190_3),
                Some(3.098_386_676_965_933_6),
                Some(3.265_986_323_710_904),
                Some(2.376_354_103_144_018_3),
                Some(1.913_066_660_344_281_2),
                Some(1.930_666_342_865_210_7),
            ]),
            EPS
        );
        assert_allclose!(
            ewm_std(XS.to_vec(), ALPHA, false, false, 0, false),
            PrimitiveArray::from([
                Some(0.0),
                Some(2.828_427_124_746_190_3),
                Some(3.098_386_676_965_933_6),
                Some(3.265_986_323_710_904),
                Some(2.376_354_103_144_018_3),
                Some(1.913_066_660_344_281_2),
                Some(1.930_666_342_865_210_7),
            ]),
            EPS
        );
        assert_allclose!(
            ewm_std(YS.to_vec(), ALPHA, true, true, 0, true),
            PrimitiveArray::from([
                None,
                Some(0.0),
                Some(0.942_809_041_582_063_4),
                None,
                None,
                Some(2.710_523_708_715_753_4),
                Some(1.885_618_083_164_126_7),
            ]),
            EPS
        );
        assert_allclose!(
            ewm_std(YS.to_vec(), ALPHA, true, true, 0, false),
            PrimitiveArray::from([
                None,
                Some(0.0),
                Some(0.942_809_041_582_063_4),
                None,
                None,
                Some(1.980_514_497_076_503),
                Some(1.596_805_731_098_222),
            ]),
            EPS
        );
        assert_allclose!(
            ewm_std(YS.to_vec(), ALPHA, true, false, 0, true),
            PrimitiveArray::from([
                None,
                Some(0.0),
                Some(SQRT_2),
                None,
                None,
                Some(3.585_685_828_003_181),
                Some(2.390_457_218_668_787),
            ]),
            EPS
        );
        assert_allclose!(
            ewm_std(YS.to_vec(), ALPHA, true, false, 0, false),
            PrimitiveArray::from([
                None,
                Some(0.0),
                Some(SQRT_2),
                None,
                None,
                Some(3.762_977_544_445_355_3),
                Some(2.244_886_116_891_356),
            ]),
            EPS
        );
        assert_allclose!(
            ewm_std(YS.to_vec(), ALPHA, false, true, 0, true),
            PrimitiveArray::from([
                None,
                Some(0.0),
                Some(1.0),
                None,
                None,
                Some(2.598_076_211_353_316),
                Some(1.854_049_621_773_915_7),
            ]),
            EPS
        );
        assert_allclose!(
            ewm_std(YS.to_vec(), ALPHA, false, true, 0, false),
            PrimitiveArray::from([
                None,
                Some(0.0),
                Some(1.0),
                None,
                None,
                Some(2.049_390_153_191_92),
                Some(1.760_681_686_165_901),
            ]),
            EPS
        );
        assert_allclose!(
            ewm_std(YS.to_vec(), ALPHA, false, false, 0, true),
            PrimitiveArray::from([
                None,
                Some(0.0),
                Some(SQRT_2),
                None,
                None,
                Some(3.286_335_345_030_997),
                Some(2.288_688_541_085_317_5),
            ]),
            EPS
        );
        assert_allclose!(
            ewm_std(YS.to_vec(), ALPHA, false, false, 0, false),
            PrimitiveArray::from([
                None,
                Some(0.0),
                Some(SQRT_2),
                None,
                None,
                Some(3.514_675_116_774_036_7),
                Some(2.301_987_249_996_250_4),
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
                Some(0.0),
                Some(0.888_888_888_888_889),
                None,
                None,
                Some(3.922_437_673_130_193_3),
                Some(2.549_788_542_868_127_3),
            ]),
            EPS
        );
        assert_allclose!(
            ewm_var(YS.to_vec(), ALPHA, true, true, 1, false),
            PrimitiveArray::from([
                None,
                Some(0.0),
                Some(0.888_888_888_888_889),
                None,
                None,
                Some(3.922_437_673_130_193_3),
                Some(2.549_788_542_868_127_3),
            ]),
            EPS
        );
        assert_allclose!(
            ewm_var(YS.to_vec(), ALPHA, true, true, 2, false),
            PrimitiveArray::from([
                None,
                None,
                Some(0.888_888_888_888_889),
                None,
                None,
                Some(3.922_437_673_130_193_3),
                Some(2.549_788_542_868_127_3),
            ]),
            EPS
        );
    }
}
