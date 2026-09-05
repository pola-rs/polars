// Some formulae:
//     mean_x = sum(weight[i] * x[i]) / sum(weight)
//     dp_xy = weighted sum of deviation products of variables x, y, written in
//             the paper as simply XY.
//     dp_xy = sum(weight[i] * (x[i] - mean_x) * (y[i] - mean_y))
//
//     cov(x, y) = dp_xy / sum(weight)
//     var(x) = cov(x, x)
//
// Algorithms from:
// Numerically stable parallel computation of (co-)variance.
// Schubert, E. & Gertz, M. (2018).
//
// Key equations from the paper:
// (17) for mean update, (23) for dp update (and also Table 1).
//
//
// For higher moments we refer to:
// Numerically Stable, Scalable Formulas for Parallel and Online Computation of
// Higher-Order Multivariate Central Moments with Arbitrary Weights.
// Pébay, P. & Terriberry, T. B. & Kolla, H. & Bennett J. (2016)
//
// Key equations from paper:
// (3.26) mean update, (3.27) moment update.
//
// Here we use mk to mean the weighted kth central moment:
//    mk = sum(weight[i] * (x[i] - mean_x)**k)
// Note that we'll use the terms m2 = dp = dp_xx if unambiguous.

#![allow(clippy::collapsible_else_if)]

use arrow::types::NativeType;
use num_traits::AsPrimitive;
use polars_array::PlPrimitiveArray;
use polars_array::bitmap::combine_validities_and;
use polars_utils::algebraic_ops::*;

const CHUNK_SIZE: usize = 128;

/// The weight of `arr`, which is its number of non-null elements.
fn weight_of<T: NativeType>(arr: &PlPrimitiveArray<T>) -> f64 {
    (arr.len() - arr.null_count()) as f64
}

/// The weight of `x` and `y` taken together, which is the number of elements at which neither is
/// null.
fn joint_weight_of<T: NativeType, U: NativeType>(
    x: &PlPrimitiveArray<T>,
    y: &PlPrimitiveArray<U>,
) -> f64 {
    let nulls = combine_validities_and(x.validity(), y.validity())
        .map_or(0, |validity| validity.unset_bits());
    (x.len() - nulls) as f64
}

/// How far a repeated value deviates from the mean of the chunk that repeats it: zero, since that
/// value *is* the mean.
///
/// The subtraction is carried out rather than assumed to vanish because a NaN deviates from
/// itself, and so does an infinity. Either one has to reach the moments the way it would have had
/// the chunk been walked element by element, which is what subtracting the value from itself does.
#[inline]
#[expect(clippy::eq_op)]
fn deviation_of(mean: f64) -> f64 {
    mean - mean
}

#[derive(Default, Clone)]
#[repr(C)] // For serialization, don't change struct member order.
pub struct VarState {
    weight: f64,
    mean: f64,
    dp: f64,
}

#[derive(Default, Clone)]
#[repr(C)] // For serialization, don't change struct member order.
pub struct CovState {
    weight: f64,
    mean_x: f64,
    mean_y: f64,
    dp_xy: f64,
}

#[derive(Default, Clone)]
#[repr(C)] // For serialization, don't change struct member order.
pub struct PearsonState {
    weight: f64,
    mean_x: f64,
    mean_y: f64,
    dp_xx: f64,
    dp_xy: f64,
    dp_yy: f64,
}

impl VarState {
    /// The state of `weight` copies of `mean`.
    ///
    /// A chunk that repeats one value has that value as its mean exactly, and no element of it
    /// deviates from that mean, so the deviation product is nothing but the repeated deviation of
    /// the value from itself -- see [`deviation_of`].
    fn repeated(mean: f64, weight: f64) -> Self {
        let deviation = deviation_of(mean);
        let mut state = Self {
            weight,
            mean,
            dp: deviation * deviation * weight,
        };
        // A chunk whose every element is null weighs nothing and has no mean at all.
        state.clear_zero_weight_nan();
        state
    }

    fn new(x: &[f64]) -> Self {
        if x.is_empty() {
            return Self::default();
        }

        let weight = x.len() as f64;
        let mean = alg_sum_f64(x.iter().copied()) / weight;
        Self {
            weight,
            mean,
            dp: alg_sum_f64(x.iter().map(|&xi| (xi - mean) * (xi - mean))),
        }
    }

    fn clear_zero_weight_nan(&mut self) {
        // Clear NaNs due to division by zero.
        if self.weight == 0.0 {
            self.mean = 0.0;
            self.dp = 0.0;
        }
    }

    pub fn insert_one(&mut self, x: f64) {
        // Just a specialized version of
        // self.combine(&Self { weight: 1.0, mean: x, dp: 0.0 })
        let new_weight = self.weight + 1.0;
        let delta_mean = x - self.mean;
        let new_mean = self.mean + delta_mean / new_weight;
        self.dp += (x - new_mean) * delta_mean;
        self.weight = new_weight;
        self.mean = new_mean;
        self.clear_zero_weight_nan();
    }

    pub fn combine(&mut self, other: &Self) {
        if other.weight == 0.0 {
            return;
        }

        let new_weight = self.weight + other.weight;
        let other_weight_frac = other.weight / new_weight;
        let delta_mean = other.mean - self.mean;
        let new_mean = self.mean + delta_mean * other_weight_frac;
        self.dp += other.dp + other.weight * (other.mean - new_mean) * delta_mean;
        self.weight = new_weight;
        self.mean = new_mean;
        self.clear_zero_weight_nan();
    }

    pub fn finalize(&self, ddof: u8) -> Option<f64> {
        if self.weight <= ddof as f64 {
            None
        } else {
            let var = self.dp / (self.weight - ddof as f64);
            Some(if var < 0.0 {
                // Variance can't be negative, except through numerical instability.
                // We don't use f64::max here so we propagate nans.
                0.0
            } else {
                var
            })
        }
    }
}

impl CovState {
    pub fn weight(&self) -> f64 {
        self.weight
    }

    /// The state of `weight` copies of the pair `(mean_x, mean_y)`; see [`VarState::repeated`].
    fn repeated(mean_x: f64, mean_y: f64, weight: f64) -> Self {
        if weight == 0.0 {
            return Self::default();
        }

        Self {
            weight,
            mean_x,
            mean_y,
            dp_xy: deviation_of(mean_x) * deviation_of(mean_y) * weight,
        }
    }

    fn new(x: &[f64], y: &[f64]) -> Self {
        assert!(x.len() == y.len());
        if x.is_empty() {
            return Self::default();
        }

        let weight = x.len() as f64;
        let inv_weight = 1.0 / weight;
        let mean_x = alg_sum_f64(x.iter().copied()) * inv_weight;
        let mean_y = alg_sum_f64(y.iter().copied()) * inv_weight;
        Self {
            weight,
            mean_x,
            mean_y,
            dp_xy: alg_sum_f64(
                x.iter()
                    .zip(y)
                    .map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y)),
            ),
        }
    }

    pub fn insert_one(&mut self, x: f64, y: f64) {
        let new_weight = self.weight + 1.0;
        let new_weight_frac = 1.0 / new_weight;
        let delta_mean_x = x - self.mean_x;
        let delta_mean_y = y - self.mean_y;
        let new_mean_x = self.mean_x + delta_mean_x * new_weight_frac;
        let new_mean_y = self.mean_y + delta_mean_y * new_weight_frac;
        self.dp_xy += (x - new_mean_x) * delta_mean_y;
        self.weight = new_weight;
        self.mean_x = new_mean_x;
        self.mean_y = new_mean_y;
    }

    pub fn combine(&mut self, other: &Self) {
        if other.weight == 0.0 {
            return;
        } else if self.weight == 0.0 {
            *self = other.clone();
            return;
        }

        let new_weight = self.weight + other.weight;
        let other_weight_frac = other.weight / new_weight;
        let delta_mean_x = other.mean_x - self.mean_x;
        let delta_mean_y = other.mean_y - self.mean_y;
        let new_mean_x = self.mean_x + delta_mean_x * other_weight_frac;
        let new_mean_y = self.mean_y + delta_mean_y * other_weight_frac;
        self.dp_xy += other.dp_xy + other.weight * (other.mean_x - new_mean_x) * delta_mean_y;
        self.weight = new_weight;
        self.mean_x = new_mean_x;
        self.mean_y = new_mean_y;
    }

    pub fn finalize(&self, ddof: u8) -> Option<f64> {
        if self.weight <= ddof as f64 {
            None
        } else {
            Some(self.dp_xy / (self.weight - ddof as f64))
        }
    }
}

impl PearsonState {
    pub fn weight(&self) -> f64 {
        self.weight
    }

    /// The state of `weight` copies of the pair `(mean_x, mean_y)`; see [`VarState::repeated`].
    fn repeated(mean_x: f64, mean_y: f64, weight: f64) -> Self {
        if weight == 0.0 {
            return Self::default();
        }

        let dx = deviation_of(mean_x);
        let dy = deviation_of(mean_y);
        Self {
            weight,
            mean_x,
            mean_y,
            dp_xx: dx * dx * weight,
            dp_xy: dx * dy * weight,
            dp_yy: dy * dy * weight,
        }
    }

    fn new(x: &[f64], y: &[f64]) -> Self {
        assert!(x.len() == y.len());
        if x.is_empty() {
            return Self::default();
        }

        let weight = x.len() as f64;
        let inv_weight = 1.0 / weight;
        let mean_x = alg_sum_f64(x.iter().copied()) * inv_weight;
        let mean_y = alg_sum_f64(y.iter().copied()) * inv_weight;
        let mut dp_xx = 0.0;
        let mut dp_xy = 0.0;
        let mut dp_yy = 0.0;
        for (xi, yi) in x.iter().zip(y.iter()) {
            dp_xx = alg_add_f64(dp_xx, (xi - mean_x) * (xi - mean_x));
            dp_xy = alg_add_f64(dp_xy, (xi - mean_x) * (yi - mean_y));
            dp_yy = alg_add_f64(dp_yy, (yi - mean_y) * (yi - mean_y));
        }
        Self {
            weight,
            mean_x,
            mean_y,
            dp_xx,
            dp_xy,
            dp_yy,
        }
    }

    pub fn insert_one(&mut self, x: f64, y: f64) {
        let new_weight = self.weight + 1.0;
        let new_weight_frac = 1.0 / new_weight;
        let delta_mean_x = x - self.mean_x;
        let delta_mean_y = y - self.mean_y;
        let new_mean_x = self.mean_x + delta_mean_x * new_weight_frac;
        let new_mean_y = self.mean_y + delta_mean_y * new_weight_frac;
        self.dp_xx += (x - new_mean_x) * delta_mean_x;
        self.dp_xy += (x - new_mean_x) * delta_mean_y;
        self.dp_yy += (y - new_mean_y) * delta_mean_y;
        self.weight = new_weight;
        self.mean_x = new_mean_x;
        self.mean_y = new_mean_y;
    }

    pub fn combine(&mut self, other: &Self) {
        if other.weight == 0.0 {
            return;
        } else if self.weight == 0.0 {
            *self = other.clone();
            return;
        }

        let new_weight = self.weight + other.weight;
        let other_weight_frac = other.weight / new_weight;
        let delta_mean_x = other.mean_x - self.mean_x;
        let delta_mean_y = other.mean_y - self.mean_y;
        let new_mean_x = self.mean_x + delta_mean_x * other_weight_frac;
        let new_mean_y = self.mean_y + delta_mean_y * other_weight_frac;
        self.dp_xx += other.dp_xx + other.weight * (other.mean_x - new_mean_x) * delta_mean_x;
        self.dp_xy += other.dp_xy + other.weight * (other.mean_x - new_mean_x) * delta_mean_y;
        self.dp_yy += other.dp_yy + other.weight * (other.mean_y - new_mean_y) * delta_mean_y;
        self.weight = new_weight;
        self.mean_x = new_mean_x;
        self.mean_y = new_mean_y;
    }

    pub fn finalize(&self) -> f64 {
        let denom_sq = self.dp_xx * self.dp_yy;
        if denom_sq > 0.0 {
            self.dp_xy / denom_sq.sqrt()
        } else {
            f64::NAN
        }
    }
}

#[derive(Default, Clone)]
#[repr(C)] // For serialization, don't change struct member order.
pub struct SkewState {
    weight: f64,
    mean: f64,
    m2: f64,
    m3: f64,
}

impl SkewState {
    /// The state of `weight` copies of `mean`; see [`VarState::repeated`].
    fn repeated(mean: f64, weight: f64) -> Self {
        let d = deviation_of(mean);
        let d2 = d * d;
        let mut state = Self {
            weight,
            mean,
            m2: d2 * weight,
            m3: d * d2 * weight,
        };
        state.clear_zero_weight_nan();
        state
    }

    fn new(x: &[f64]) -> Self {
        Self::from_iter(x.iter().copied(), x.len())
    }

    fn from_iter(iter: impl Iterator<Item = f64> + Clone, length: usize) -> Self {
        if length == 0 {
            return Self::default();
        }

        let weight = length as f64;
        let mean = alg_sum_f64(iter.clone()) / weight;
        let mut m2 = 0.0;
        let mut m3 = 0.0;
        for xi in iter {
            let d = xi - mean;
            let d2 = d * d;
            let d3 = d * d2;
            m2 = alg_add_f64(m2, d2);
            m3 = alg_add_f64(m3, d3);
        }
        Self {
            weight,
            mean,
            m2,
            m3,
        }
    }

    fn clear_zero_weight_nan(&mut self) {
        // Clear NaNs due to division by zero.
        if self.weight == 0.0 {
            self.mean = 0.0;
            self.m2 = 0.0;
            self.m3 = 0.0;
        }
    }

    /// The state of the `length` elements of `arr` starting at `start`, folded in one pass.
    ///
    /// # Panics
    /// Panics if `start + length` exceeds the length of `arr`.
    pub fn from_array(arr: &PlPrimitiveArray<f64>, start: usize, length: usize) -> Self {
        // Slicing preserves the representation, so a range of a chunk that repeats one value
        // repeats it too and is read in `O(1)` below.
        let arr = arr.clone().sliced(start, length);

        // Every element of a chunk that repeats one value is that value, whatever the range's
        // length, so the whole range weighs in at once.
        if let Some(value) = arr.scalar_values() {
            return Self::repeated(value, weight_of(&arr));
        }

        if arr.has_nulls() {
            Self::from_iter(arr.iter().flatten(), arr.len() - arr.null_count())
        } else {
            Self::from_iter(arr.values_iter(), arr.len())
        }
    }

    pub fn insert_one(&mut self, x: f64) {
        // Specialization of self.combine(&SkewState { weight: 1.0, mean: x, m2: 0.0, m3: 0.0 });
        let new_weight = self.weight + 1.0;
        let delta_mean = x - self.mean;
        let delta_mean_weight = delta_mean / new_weight;
        let new_mean = self.mean + delta_mean_weight;

        let weight_diff = self.weight - 1.0;
        let m2_update = (x - new_mean) * delta_mean;
        let new_m2 = self.m2 + m2_update;
        let new_m3 = self.m3 + delta_mean_weight * (m2_update * weight_diff - 3.0 * self.m2);

        self.weight = new_weight;
        self.mean = new_mean;
        self.m2 = new_m2;
        self.m3 = new_m3;
        self.clear_zero_weight_nan();
    }

    pub fn combine(&mut self, other: &Self) {
        if other.weight == 0.0 {
            return;
        } else if self.weight == 0.0 {
            *self = other.clone();
            return;
        }

        let new_weight = self.weight + other.weight;
        let delta_mean = other.mean - self.mean;
        let delta_mean_weight = delta_mean / new_weight;
        let new_mean = self.mean + other.weight * delta_mean_weight;

        let weight_diff = self.weight - other.weight;
        let self_weight_other_m2 = self.weight * other.m2;
        let other_weight_self_m2 = other.weight * self.m2;
        let m2_update = other.weight * (other.mean - new_mean) * delta_mean;
        let new_m2 = self.m2 + other.m2 + m2_update;
        let new_m3 = self.m3
            + other.m3
            + delta_mean_weight
                * (m2_update * weight_diff + 3.0 * (self_weight_other_m2 - other_weight_self_m2));

        self.weight = new_weight;
        self.mean = new_mean;
        self.m2 = new_m2;
        self.m3 = new_m3;
        self.clear_zero_weight_nan();
    }

    pub fn finalize(&self, bias: bool) -> Option<f64> {
        let m2 = self.m2 / self.weight;
        let m3 = self.m3 / self.weight;
        let is_zero = m2 <= (f64::EPSILON * self.mean).powi(2);
        let biased_est = if is_zero { f64::NAN } else { m3 / m2.powf(1.5) };
        if bias {
            if self.weight == 0.0 {
                None
            } else {
                Some(biased_est)
            }
        } else {
            if self.weight <= 2.0 {
                None
            } else {
                let correction = (self.weight * (self.weight - 1.0)).sqrt() / (self.weight - 2.0);
                Some(correction * biased_est)
            }
        }
    }
}

#[derive(Default, Clone)]
#[repr(C)] // For serialization, don't change struct member order.
pub struct KurtosisState {
    weight: f64,
    mean: f64,
    m2: f64,
    m3: f64,
    m4: f64,
}

impl KurtosisState {
    /// The state of `weight` copies of `mean`; see [`VarState::repeated`].
    fn repeated(mean: f64, weight: f64) -> Self {
        let d = deviation_of(mean);
        let d2 = d * d;
        let mut state = Self {
            weight,
            mean,
            m2: d2 * weight,
            m3: d * d2 * weight,
            m4: d2 * d2 * weight,
        };
        state.clear_zero_weight_nan();
        state
    }

    pub fn new(x: &[f64]) -> Self {
        Self::from_iter(x.iter().copied(), x.len())
    }

    pub fn from_iter(iter: impl Iterator<Item = f64> + Clone, length: usize) -> Self {
        if length == 0 {
            return Self::default();
        }

        let weight = length as f64;
        let mean = alg_sum_f64(iter.clone()) / weight;
        let mut m2 = 0.0;
        let mut m3 = 0.0;
        let mut m4 = 0.0;
        for xi in iter {
            let d = xi - mean;
            let d2 = d * d;
            let d3 = d * d2;
            let d4 = d2 * d2;
            m2 = alg_add_f64(m2, d2);
            m3 = alg_add_f64(m3, d3);
            m4 = alg_add_f64(m4, d4);
        }
        Self {
            weight,
            mean,
            m2,
            m3,
            m4,
        }
    }

    /// The state of the `length` elements of `arr` starting at `start`, folded in one pass; see
    /// [`SkewState::from_array`].
    ///
    /// # Panics
    /// Panics if `start + length` exceeds the length of `arr`.
    pub fn from_array(arr: &PlPrimitiveArray<f64>, start: usize, length: usize) -> Self {
        let arr = arr.clone().sliced(start, length);

        if let Some(value) = arr.scalar_values() {
            return Self::repeated(value, weight_of(&arr));
        }

        if arr.has_nulls() {
            Self::from_iter(arr.iter().flatten(), arr.len() - arr.null_count())
        } else {
            Self::from_iter(arr.values_iter(), arr.len())
        }
    }

    fn clear_zero_weight_nan(&mut self) {
        // Clear NaNs due to division by zero.
        if self.weight == 0.0 {
            self.mean = 0.0;
            self.m2 = 0.0;
            self.m3 = 0.0;
            self.m4 = 0.0;
        }
    }

    pub fn insert_one(&mut self, x: f64) {
        // Specialization of self.combine(&KurtosisState { weight: 1.0, mean: x, m2: 0.0, m3: 0.0, m4: 0.0 });
        let new_weight = self.weight + 1.0;
        let delta_mean = x - self.mean;
        let delta_mean_weight = delta_mean / new_weight;
        let new_mean = self.mean + delta_mean_weight;

        let weight_diff = self.weight - 1.0;
        let m2_update = (x - new_mean) * delta_mean;
        let new_m2 = self.m2 + m2_update;
        let new_m3 = self.m3 + delta_mean_weight * (m2_update * weight_diff - 3.0 * self.m2);
        let new_m4 = self.m4
            + delta_mean_weight
                * (delta_mean_weight
                    * (m2_update * (self.weight * weight_diff + 1.0) + 6.0 * self.m2)
                    - 4.0 * self.m3);

        self.weight = new_weight;
        self.mean = new_mean;
        self.m2 = new_m2;
        self.m3 = new_m3;
        self.m4 = new_m4;
        self.clear_zero_weight_nan();
    }

    pub fn combine(&mut self, other: &Self) {
        if other.weight == 0.0 {
            return;
        } else if self.weight == 0.0 {
            *self = other.clone();
            return;
        }

        let new_weight = self.weight + other.weight;
        let delta_mean = other.mean - self.mean;
        let delta_mean_weight = delta_mean / new_weight;
        let new_mean = self.mean + other.weight * delta_mean_weight;

        let weight_diff = self.weight - other.weight;
        let self_weight_other_m2 = self.weight * other.m2;
        let other_weight_self_m2 = other.weight * self.m2;
        let m2_update = other.weight * (other.mean - new_mean) * delta_mean;
        let new_m2 = self.m2 + other.m2 + m2_update;
        let new_m3 = self.m3
            + other.m3
            + delta_mean_weight
                * (m2_update * weight_diff + 3.0 * (self_weight_other_m2 - other_weight_self_m2));
        let new_m4 = self.m4
            + other.m4
            + delta_mean_weight
                * (delta_mean_weight
                    * (m2_update * (self.weight * weight_diff + other.weight * other.weight)
                        + 6.0
                            * (self.weight * self_weight_other_m2
                                + other.weight * other_weight_self_m2))
                    + 4.0 * (self.weight * other.m3 - other.weight * self.m3));

        self.weight = new_weight;
        self.mean = new_mean;
        self.m2 = new_m2;
        self.m3 = new_m3;
        self.m4 = new_m4;
        self.clear_zero_weight_nan();
    }

    pub fn finalize(&self, fisher: bool, bias: bool) -> Option<f64> {
        let m4 = self.m4 / self.weight;
        let m2 = self.m2 / self.weight;
        let is_zero = m2 <= (f64::EPSILON * self.mean).powi(2);
        let biased_est = if is_zero { f64::NAN } else { m4 / (m2 * m2) };
        let out = if bias {
            if self.weight == 0.0 {
                return None;
            }

            biased_est
        } else {
            if self.weight <= 3.0 {
                return None;
            }

            let n = self.weight;
            let nm1_nm2 = (n - 1.0) / (n - 2.0);
            let np1_nm3 = (n + 1.0) / (n - 3.0);
            let nm1_nm3 = (n - 1.0) / (n - 3.0);
            nm1_nm2 * (np1_nm3 * biased_est - 3.0 * nm1_nm3) + 3.0
        };

        if fisher { Some(out - 3.0) } else { Some(out) }
    }
}

fn chunk_as_float<T, I, F>(it: I, mut f: F)
where
    T: NativeType + AsPrimitive<f64>,
    I: IntoIterator<Item = T>,
    F: FnMut(&[f64]),
{
    let mut chunk = [0.0; CHUNK_SIZE];
    let mut i = 0;
    for val in it {
        if i >= CHUNK_SIZE {
            f(&chunk);
            i = 0;
        }
        chunk[i] = val.as_();
        i += 1;
    }
    if i > 0 {
        f(&chunk[..i]);
    }
}

fn chunk_as_float_binary<T, U, I, F>(it: I, mut f: F)
where
    T: NativeType + AsPrimitive<f64>,
    U: NativeType + AsPrimitive<f64>,
    I: IntoIterator<Item = (T, U)>,
    F: FnMut(&[f64], &[f64]),
{
    let mut left_chunk = [0.0; CHUNK_SIZE];
    let mut right_chunk = [0.0; CHUNK_SIZE];
    let mut i = 0;
    for (l, r) in it {
        if i >= CHUNK_SIZE {
            f(&left_chunk, &right_chunk);
            i = 0;
        }
        left_chunk[i] = l.as_();
        right_chunk[i] = r.as_();
        i += 1;
    }
    if i > 0 {
        f(&left_chunk[..i], &right_chunk[..i]);
    }
}

pub fn var<T>(arr: &PlPrimitiveArray<T>) -> VarState
where
    T: NativeType + AsPrimitive<f64>,
{
    // Every element of a chunk that repeats one value is that value, which is therefore the
    // chunk's mean: the whole chunk weighs in at once, without an element of it being walked.
    if let Some(value) = arr.scalar_values() {
        return VarState::repeated(value.as_(), weight_of(arr));
    }

    let mut out = VarState::default();
    if arr.has_nulls() {
        chunk_as_float(arr.iter().flatten(), |chunk| {
            out.combine(&VarState::new(chunk))
        });
    } else {
        chunk_as_float(arr.values_iter(), |chunk| {
            out.combine(&VarState::new(chunk))
        });
    }
    out
}

pub fn cov<T, U>(x: &PlPrimitiveArray<T>, y: &PlPrimitiveArray<U>) -> CovState
where
    T: NativeType + AsPrimitive<f64>,
    U: NativeType + AsPrimitive<f64>,
{
    assert!(x.len() == y.len());

    // Two chunks that each repeat one value are each their own mean, and the pair weighs in at
    // the elements where both of them are non-null.
    if let (Some(x_value), Some(y_value)) = (x.scalar_values(), y.scalar_values()) {
        return CovState::repeated(x_value.as_(), y_value.as_(), joint_weight_of(x, y));
    }

    let mut out = CovState::default();
    if x.has_nulls() || y.has_nulls() {
        chunk_as_float_binary(
            x.iter().zip(y.iter()).filter_map(|(l, r)| l.zip(r)),
            |l, r| out.combine(&CovState::new(l, r)),
        );
    } else {
        chunk_as_float_binary(x.values_iter().zip(y.values_iter()), |l, r| {
            out.combine(&CovState::new(l, r))
        });
    }
    out
}

pub fn pearson_corr<T, U>(x: &PlPrimitiveArray<T>, y: &PlPrimitiveArray<U>) -> PearsonState
where
    T: NativeType + AsPrimitive<f64>,
    U: NativeType + AsPrimitive<f64>,
{
    assert!(x.len() == y.len());

    if let (Some(x_value), Some(y_value)) = (x.scalar_values(), y.scalar_values()) {
        return PearsonState::repeated(x_value.as_(), y_value.as_(), joint_weight_of(x, y));
    }

    let mut out = PearsonState::default();
    if x.has_nulls() || y.has_nulls() {
        chunk_as_float_binary(
            x.iter().zip(y.iter()).filter_map(|(l, r)| l.zip(r)),
            |l, r| out.combine(&PearsonState::new(l, r)),
        );
    } else {
        chunk_as_float_binary(x.values_iter().zip(y.values_iter()), |l, r| {
            out.combine(&PearsonState::new(l, r))
        });
    }
    out
}

pub fn skew<T>(arr: &PlPrimitiveArray<T>) -> SkewState
where
    T: NativeType + AsPrimitive<f64>,
{
    if let Some(value) = arr.scalar_values() {
        return SkewState::repeated(value.as_(), weight_of(arr));
    }

    let mut out = SkewState::default();
    if arr.has_nulls() {
        chunk_as_float(arr.iter().flatten(), |chunk| {
            out.combine(&SkewState::new(chunk))
        });
    } else {
        chunk_as_float(arr.values_iter(), |chunk| {
            out.combine(&SkewState::new(chunk))
        });
    }
    out
}

pub fn kurtosis<T>(arr: &PlPrimitiveArray<T>) -> KurtosisState
where
    T: NativeType + AsPrimitive<f64>,
{
    if let Some(value) = arr.scalar_values() {
        return KurtosisState::repeated(value.as_(), weight_of(arr));
    }

    let mut out = KurtosisState::default();
    if arr.has_nulls() {
        chunk_as_float(arr.iter().flatten(), |chunk| {
            out.combine(&KurtosisState::new(chunk))
        });
    } else {
        chunk_as_float(arr.values_iter(), |chunk| {
            out.combine(&KurtosisState::new(chunk))
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use arrow::bitmap::Bitmap;
    use polars_array::PlBitmap;

    use super::*;

    /// `length` copies of `value`, marked by `validity`, in both representations.
    fn repeated(
        value: f64,
        validity: Option<&Bitmap>,
        length: usize,
    ) -> [PlPrimitiveArray<f64>; 2] {
        let scalar = PlPrimitiveArray::new_scalar(value, length)
            .with_validity(validity.cloned().map(PlBitmap::from_bitmap));
        let flat = PlPrimitiveArray::from_vec(vec![value; length])
            .with_validity(validity.cloned().map(PlBitmap::from_bitmap));
        [scalar, flat]
    }

    /// A chunk that repeats one value is its own mean and has no spread about it, whichever
    /// representation it is stored in.
    #[test]
    fn a_repeated_value_has_no_spread() {
        for length in [0, 1, 2, 3, 65, 300] {
            for valid in 0..=length {
                let mask: Bitmap = (0..length).map(|i| i < valid).collect();
                for validity in [None, Some(&mask)] {
                    let [scalar, flat] = repeated(7.5, validity, length);
                    let weight = validity.map_or(length, |_| valid);

                    for ddof in [0, 1] {
                        let expected = (weight > ddof as usize).then_some(0.0);
                        assert_eq!(var(&scalar).finalize(ddof), expected, "var of {scalar:?}");
                        assert_eq!(var(&flat).finalize(ddof), expected);
                    }

                    // The spread is zero either way, so skew and kurtosis are undefined; what
                    // matters is that the two representations agree on that.
                    assert_eq!(
                        skew(&scalar).finalize(true).map(f64::is_nan),
                        skew(&flat).finalize(true).map(f64::is_nan),
                    );
                    assert_eq!(
                        kurtosis(&scalar).finalize(true, true).map(f64::is_nan),
                        kurtosis(&flat).finalize(true, true).map(f64::is_nan),
                    );
                }
            }
        }
    }

    /// A repeated value is read as the mean exactly, which is what the chunk holds however many
    /// times it holds it.
    #[test]
    fn a_repeated_value_is_read_as_the_mean() {
        // A tenth is not exactly representable, so summing it 300 times and dividing drifts off
        // the value itself; reading the repeated value takes the mean it should have.
        let scalar = PlPrimitiveArray::new_scalar(0.1, 300);
        let state = var(&scalar);
        assert_eq!(state.mean, 0.1);
        assert_eq!(state.dp, 0.0);
        assert_eq!(state.weight, 300.0);
    }

    /// A chunk of nothing but NaNs has a NaN mean, and the NaN reaches the moments the way it
    /// would have had the chunk been walked.
    #[test]
    fn a_repeated_nan_carries_through() {
        for length in [1, 3, 300] {
            let scalar = PlPrimitiveArray::new_scalar(f64::NAN, length);
            let flat = PlPrimitiveArray::from_vec(vec![f64::NAN; length]);

            for ddof in [0, 1] {
                assert_eq!(
                    var(&scalar).finalize(ddof).map(f64::is_nan),
                    var(&flat).finalize(ddof).map(f64::is_nan),
                    "a NaN chunk of {length} must read alike either way",
                );
                assert_eq!(var(&scalar).finalize(ddof).map(f64::is_nan), {
                    let expected = length > ddof as usize;
                    expected.then_some(true)
                });
            }
        }
    }

    /// An all-null chunk weighs nothing, whatever value sits under the mask.
    #[test]
    fn an_all_null_chunk_weighs_nothing() {
        let scalar = PlPrimitiveArray::<f64>::new_full_null(300);
        assert_eq!(var(&scalar).finalize(0), None);
        assert_eq!(skew(&scalar).finalize(true), None);
        assert_eq!(kurtosis(&scalar).finalize(true, true), None);

        // Nor does an empty one, which holds no value to repeat at all.
        let empty = PlPrimitiveArray::<f64>::new_empty();
        assert_eq!(var(&empty).finalize(0), None);
    }

    /// A chunk laid out one value per element folds its non-null elements as it always has.
    #[test]
    fn null_elements_are_passed_over() {
        let arr = PlPrimitiveArray::from_iter([Some(1.0), None, Some(2.0), Some(3.0)]);
        // The variance of 1, 2 and 3 with one degree of freedom is exactly 1.
        assert_eq!(var(&arr).finalize(1), Some(1.0));
        assert_eq!(var(&arr).finalize(0), Some(2.0 / 3.0));
    }

    /// Two chunks that each repeat one value never vary together, so they have no covariance and
    /// no correlation to speak of.
    #[test]
    fn two_repeated_values_do_not_vary_together() {
        for length in [1, 2, 65] {
            let x = PlPrimitiveArray::new_scalar(3.0, length);
            let y = PlPrimitiveArray::new_scalar(-1.0, length);
            let flat_x = PlPrimitiveArray::from_vec(vec![3.0; length]);
            let flat_y = PlPrimitiveArray::from_vec(vec![-1.0; length]);

            assert_eq!(cov(&x, &y).finalize(0), Some(0.0));
            assert_eq!(cov(&x, &y).finalize(0), cov(&flat_x, &flat_y).finalize(0));
            assert_eq!(cov(&x, &y).weight(), length as f64);

            assert!(pearson_corr(&x, &y).finalize().is_nan());
            assert!(pearson_corr(&flat_x, &flat_y).finalize().is_nan());
        }
    }

    /// A pair weighs in only at the elements where neither side is null, whichever side the mask
    /// is on and whichever representation it is in.
    #[test]
    fn a_pair_weighs_where_both_sides_are_non_null() {
        let length = 8;
        let x = PlPrimitiveArray::new_scalar(3.0, length)
            .with_validity(Some((0..length).map(|i| i < 6).collect()));
        let y = PlPrimitiveArray::new_scalar(5.0, length)
            .with_validity(Some((0..length).map(|i| i >= 2).collect()));

        // Elements 2 through 5 are non-null on both sides.
        assert_eq!(cov(&x, &y).weight(), 4.0);
        assert_eq!(cov(&x, &y).finalize(0), Some(0.0));

        // A scalar mask that leaves every element null leaves the pair weighing nothing.
        let none = PlPrimitiveArray::<f64>::new_full_null(length);
        assert_eq!(cov(&x, &none).weight(), 0.0);
        assert_eq!(cov(&x, &none).finalize(0), None);
    }

    /// The covariance of a chunk that repeats a value against one that does not is folded the
    /// long way, and is the same as if neither of them repeated.
    #[test]
    fn one_repeated_side_still_folds_the_long_way() {
        let values: Vec<f64> = (0..70).map(|i| i as f64).collect();
        let y = PlPrimitiveArray::from_vec(values.clone());
        let x = PlPrimitiveArray::new_scalar(2.0, y.len());
        let flat_x = PlPrimitiveArray::from_vec(vec![2.0; y.len()]);

        // A constant does not vary, so it covaries with nothing.
        assert_eq!(cov(&x, &y).finalize(1), Some(0.0));
        assert_eq!(cov(&x, &y).finalize(1), cov(&flat_x, &y).finalize(1));
    }

    /// The states a range of a chunk folds to are the ones the whole chunk folds to when the
    /// range is all of it, in either representation.
    #[test]
    fn a_range_reads_the_same_either_way() {
        let length = 70;
        let scalar = PlPrimitiveArray::new_scalar(4.0, length);
        let flat = PlPrimitiveArray::from_vec(vec![4.0; length]);

        for (start, len) in [(0, length), (0, 1), (3, 17), (length - 1, 1), (10, 0)] {
            let from_scalar = SkewState::from_array(&scalar, start, len);
            let from_flat = SkewState::from_array(&flat, start, len);
            assert_eq!(
                from_scalar.weight, from_flat.weight,
                "range {start}..+{len}"
            );
            assert_eq!(from_scalar.weight, len as f64);
            assert_eq!(
                from_scalar.finalize(true).map(f64::is_nan),
                from_flat.finalize(true).map(f64::is_nan),
            );

            let from_scalar = KurtosisState::from_array(&scalar, start, len);
            let from_flat = KurtosisState::from_array(&flat, start, len);
            assert_eq!(from_scalar.weight, from_flat.weight);
            assert_eq!(
                from_scalar.finalize(true, true).map(f64::is_nan),
                from_flat.finalize(true, true).map(f64::is_nan),
            );
        }
    }

    /// A range that a mask leaves partly null weighs only its non-null elements, and reads the
    /// same whichever representation the values are in.
    #[test]
    fn a_masked_range_weighs_its_non_null_elements() {
        let length = 40;
        let mask: Bitmap = (0..length).map(|i| i % 3 == 0).collect();
        let [scalar, flat] = repeated(2.0, Some(&mask), length);

        for (start, len) in [(0, length), (1, 9), (7, 12), (length - 2, 2)] {
            let from_scalar = SkewState::from_array(&scalar, start, len);
            let from_flat = SkewState::from_array(&flat, start, len);
            let expected = (start..start + len).filter(|i| i % 3 == 0).count() as f64;

            assert_eq!(from_scalar.weight, expected, "range {start}..+{len}");
            assert_eq!(from_flat.weight, expected);
            assert_eq!(from_scalar.mean, from_flat.mean);
        }
    }

    /// A range of a chunk laid out one value per element folds the elements it covers, and only
    /// those.
    #[test]
    fn a_range_covers_only_its_own_elements() {
        let arr = PlPrimitiveArray::from_vec(vec![1.0, 2.0, 3.0, 100.0]);

        // The variance of 1, 2 and 3 with one degree of freedom is exactly 1, so a range that
        // stops short of the outlier is unmoved by it.
        assert_eq!(var(&arr.clone().sliced(0, 3)).finalize(1), Some(1.0));
        assert_eq!(SkewState::from_array(&arr, 0, 3).weight, 3.0);
        assert_eq!(SkewState::from_array(&arr, 0, 3).mean, 2.0);
    }
}
