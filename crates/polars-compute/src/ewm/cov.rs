use std::ops::{AddAssign, DivAssign, MulAssign};

use arrow::array::{Array, PrimitiveArray};
use arrow::trusted_len::TrustedLen;
use arrow::types::NativeType;
use num_traits::Float;

use crate::ewm::EwmStateUpdate;

pub struct EwmCovState<T> {
    weight: T,
    mean_x: T,
    mean_y: T,
    cov: T,
    weight_sum: T,
    weight_square_sum: T,
    alpha: T,
    non_null_count: usize,
    adjust: bool,
    bias: bool,
    min_periods: usize,
    ignore_nulls: bool,
}

impl<T> EwmCovState<T>
where
    T: num_traits::Float,
{
    pub fn new(alpha: T, adjust: bool, bias: bool, min_periods: usize, ignore_nulls: bool) -> Self {
        Self {
            mean_x: T::zero(),
            mean_y: T::zero(),
            weight: T::zero(),
            cov: T::zero(),
            weight_sum: T::zero(),
            weight_square_sum: T::zero(),
            alpha,
            non_null_count: 0,
            adjust,
            bias,
            min_periods: min_periods.max(1),
            ignore_nulls,
        }
    }
}

impl<T> EwmCovState<T>
where
    T: NativeType
        + num_traits::Float
        + std::ops::AddAssign
        + std::ops::DivAssign
        + std::ops::MulAssign,
{
    pub fn update_iter<I>(&mut self, values: I) -> impl Iterator<Item = Option<T>>
    where
        I: IntoIterator<Item = Option<(T, T)>>,
    {
        let other_weight = if self.adjust { T::one() } else { self.alpha };

        values.into_iter().map(move |opt_xy| {
            if self.non_null_count == 0
                && let Some((x, y)) = opt_xy
            {
                // Initialize
                self.non_null_count = 1;
                self.mean_x = x;
                self.mean_y = y;
                self.weight = T::one();
                self.weight_sum = T::one();
                self.weight_square_sum = T::one();
            } else {
                if opt_xy.is_some() || !self.ignore_nulls {
                    self.weight_sum *= T::one() - self.alpha;
                    self.weight_square_sum *= (T::one() - self.alpha) * (T::one() - self.alpha);
                    self.weight *= T::one() - self.alpha;
                }

                if let Some((other_x, other_y)) = opt_xy {
                    self.non_null_count += 1;

                    let new_weight = self.weight + other_weight;
                    let other_weight_frac = other_weight / new_weight;
                    let delta_mean_x = other_x - self.mean_x;
                    let delta_mean_y = other_y - self.mean_y;

                    let new_mean_x = self.mean_x + delta_mean_x * other_weight_frac;
                    let new_mean_y = self.mean_y + delta_mean_y * other_weight_frac;

                    let cov = ((self.weight
                        * (self.cov + (self.mean_x - new_mean_x) * (self.mean_y - new_mean_y)))
                        + other_weight * (other_x - new_mean_x) * (other_y - new_mean_y))
                        / new_weight;

                    self.cov = cov;
                    self.weight = new_weight;
                    self.mean_x = new_mean_x;
                    self.mean_y = new_mean_y;

                    self.weight_sum += other_weight;
                    self.weight_square_sum += other_weight * other_weight;

                    if !self.adjust {
                        self.weight_sum /= new_weight;
                        self.weight_square_sum /= new_weight * new_weight;
                        self.weight = T::one();
                    }
                }
            }

            (opt_xy.is_some() && self.non_null_count >= self.min_periods)
                .then_some(self.cov)
                .and_then(|cov| {
                    if self.bias || self.non_null_count == 1 {
                        Some(cov)
                    } else {
                        let numerator = self.weight_sum * self.weight_sum;
                        let denominator = numerator - self.weight_square_sum;
                        if denominator > T::zero() {
                            Some((numerator / denominator) * cov)
                        } else {
                            None
                        }
                    }
                })
        })
    }
}

pub struct EwmVarState<T>(EwmCovState<T>);

impl<T> EwmVarState<T> {
    pub fn new(cov_state: EwmCovState<T>) -> Self {
        Self(cov_state)
    }
}

impl<T> EwmStateUpdate for EwmVarState<T>
where
    T: NativeType
        + num_traits::Float
        + std::ops::AddAssign
        + std::ops::DivAssign
        + std::ops::MulAssign,
{
    fn ewm_state_update(&mut self, values: &dyn Array) -> Box<dyn Array> {
        let values: &PrimitiveArray<T> = values.as_any().downcast_ref().unwrap();

        let out: PrimitiveArray<T> = self
            .0
            .update_iter(values.iter().map(|x| x.map(|x| (*x, *x))))
            .collect();

        out.boxed()
    }
}

pub struct EwmStdState<T>(EwmCovState<T>);

impl<T> EwmStdState<T> {
    pub fn new(cov_state: EwmCovState<T>) -> Self {
        Self(cov_state)
    }
}

impl<T> EwmStateUpdate for EwmStdState<T>
where
    T: NativeType
        + num_traits::Float
        + std::ops::AddAssign
        + std::ops::DivAssign
        + std::ops::MulAssign,
{
    fn ewm_state_update(&mut self, values: &dyn Array) -> Box<dyn Array> {
        let values: &PrimitiveArray<T> = values.as_any().downcast_ref().unwrap();

        let out: PrimitiveArray<T> = self
            .0
            .update_iter(values.iter().map(|x| x.map(|x| (*x, *x))))
            .map(|x| x.map(|x| x.sqrt()))
            .collect();

        out.boxed()
    }
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
    let mut state = EwmCovState::new(alpha, adjust, bias, min_periods, ignore_nulls);
    let iter = state.update_iter(xs.into_iter().map(|x| x.map(|x| (x, x))));

    iter.collect()
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
    I: IntoIterator<Item = Option<T>>,
    T: NativeType
        + num_traits::Float
        + std::ops::AddAssign
        + std::ops::DivAssign
        + std::ops::MulAssign,
{
    let mut state = EwmCovState::new(alpha, adjust, bias, min_periods, ignore_nulls);
    let iter = state.update_iter(xs.into_iter().map(|x| x.map(|x| (x, x))));

    iter.map(|opt_x| opt_x.map(|x| x.sqrt())).collect()
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

    fn ewm_cov<I, T>(
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
        let mut state = EwmCovState::new(alpha, adjust, bias, min_periods, ignore_nulls);
        let iter = state.update_iter(xs.into_iter().zip(ys).map(|(x, y)| x.zip(y)));

        iter.collect()
    }

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
            PrimitiveArray::from([None, Some(0.0), Some(1.0), None, None, Some(4.2), Some(3.1)]),
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
