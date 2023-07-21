use nulls;
use nulls::{rolling_apply_agg_window, RollingAggWindowNulls};

use super::*;

pub struct MinMaxWindow<'a, T: NativeType + PartialOrd + IsFloat, const R: i8> {
    slice: &'a [T],
    validity: &'a Bitmap,
    m: Option<T>,
    m_idx: usize,
    sorted_to: usize,
    last_start: usize,
    last_end: usize,
    null_count: usize,
}

// R should be an Ordering but isn't because non-primitive const generics aren't considered stable
// as of this writing. Passing comparison information this way is faster than having a field inside
// the struct so we live with it.
impl<'a, T: NativeType + PartialOrd + IsFloat, const R: i8> MinMaxWindow<'a, T, R> {
    #[inline]
    unsafe fn cmp_f(&self, a: &T, b: &T) -> i8 {
        // Seems like it's faster to directly put this here instead of having a field in the window
        // that gets called dynamically
        if T::is_float() {
            match (a.is_nan(), b.is_nan()) {
                // safety: we checked nans
                (false, false) => a.partial_cmp(b).unwrap_unchecked() as i8,
                (true, true) => 0,
                (true, false) => -R,
                (false, true) => R,
            }
        } else {
            a.partial_cmp(b).unwrap_unchecked() as i8
        }
    }
    #[inline]
    unsafe fn new_is_m(&self, old: &T, new: &T) -> bool {
        self.cmp_f(new, old) != R
    }

    #[inline]
    unsafe fn update_sorted_to(&mut self) {
        let mut last_val = self.m.unwrap();
        self.sorted_to = self.m_idx;
        for i in (self.m_idx + 1)..self.slice.len() {
            self.sorted_to += 1;
            if self.validity.get_bit_unchecked(i) {
                let val = self.slice.get_unchecked(i);
                if self.cmp_f(&last_val, val) == R {
                    break;
                }
                last_val = *val;
            }
        }
    }

    #[inline]
    unsafe fn get_m_m_idx_and_null_count(
        &self,
        start: usize,
        end: usize,
    ) -> (Option<(usize, &'a T)>, usize) {
        let ilen = end - start;
        let leading_nc = self
            .validity
            .iter()
            .skip(start)
            .take(ilen)
            .position(|x| x)
            .unwrap_or(ilen);
        if leading_nc == ilen {
            return (None, ilen);
        }
        let vstart = start + leading_nc; // First _V_alid entry start
        let (mut m, mut m_idx) = (self.slice.get_unchecked(vstart), vstart);
        if self.sorted_to >= end {
            let remaining_nc = self.validity.null_count_range(vstart, end - vstart);
            (Some((m_idx, m)), leading_nc + remaining_nc)
        } else {
            let tstart = (vstart + 1).max(self.sorted_to); // _T_ail elements to check
            let mut remaining_nc = self.validity.null_count_range(vstart, tstart - vstart);
            for i in tstart..end {
                if !self.validity.get_bit_unchecked(i) {
                    remaining_nc += 1;
                    continue;
                }
                let newval = self.slice.get_unchecked(i);
                if self.new_is_m(m, newval) {
                    (m, m_idx) = (newval, i);
                }
            }
            (Some((m_idx, m)), remaining_nc + leading_nc)
        }
    }

    #[inline]
    unsafe fn update_m_and_m_idx(&mut self, m_and_idx: (usize, &T)) {
        self.m = Some(*m_and_idx.1);
        self.m_idx = m_and_idx.0;
        if self.sorted_to <= self.m_idx {
            // Track how far past the current extremum values are sorted. Direction depends on min/max
            // Tracking sorted ranges lets us only do comparisons when we have to.
            self.update_sorted_to();
        }
    }

    unsafe fn new(
        slice: &'a [T],
        validity: &'a Bitmap,
        start: usize,
        end: usize,
        //        sort_end_order: Ordering,
    ) -> Self {
        let mut out = Self {
            slice,
            validity,
            m: None,
            m_idx: 0,
            sorted_to: 1,
            last_start: start,
            last_end: end,
            null_count: 0,
        };
        let (m_idx, null_count) = out.get_m_m_idx_and_null_count(start, end);
        let (m, idx) = (m_idx.map(|x| x.1), m_idx.map_or(0, |x| x.0));
        (out.m, out.m_idx, out.null_count) = (m.cloned(), idx, null_count);
        if out.m.is_some() {
            out.update_sorted_to();
        }
        out
    }

    unsafe fn update(&mut self, start: usize, end: usize) -> Option<T> {
        //For details see: https://github.com/pola-rs/polars/pull/9277#issuecomment-1581401692
        let leaving_nc = self
            .validity
            .null_count_range(self.last_start, start - self.last_start);
        self.last_start = start; // Don't care where the last one started now
        let old_last_end = self.last_end; // But we need this
        self.last_end = end;
        let entering_start = std::cmp::max(old_last_end, start);
        let (entering, entering_nc) = if end - entering_start == 1 {
            // Faster in the special, but common, case of a fixed window rolling by one
            if self.validity.get_bit_unchecked(entering_start) {
                (
                    Some((entering_start, self.slice.get_unchecked(entering_start))),
                    0,
                )
            } else {
                (None, 1)
            }
        } else {
            self.get_m_m_idx_and_null_count(entering_start, end)
        };
        let empty_overlap = old_last_end <= start;
        self.null_count = (self.null_count + entering_nc) - leaving_nc;

        if entering.is_some_and(|em| {
            self.m.is_none() || empty_overlap || self.new_is_m(&self.m.unwrap(), em.1)
        }) {
            // The entering extremum "beats" the previous extremum so we can ignore the overlap
            self.update_m_and_m_idx(entering.unwrap());
            return self.m;
        } else if self.m_idx >= start || empty_overlap {
            // The previous extremum didn't drop off. Keep it
            return self.m;
        }
        // Otherwise get the min of the overlapping window and the entering extremum
        // If the last value was None, the value in the overlapping window will be too
        let (previous, _) = match self.m {
            None => (None, 0),
            Some(_) => self.get_m_m_idx_and_null_count(start, old_last_end),
        };
        match (previous, entering) {
            (Some(pm), Some(em)) => {
                if self.new_is_m(pm.1, em.1) {
                    self.update_m_and_m_idx(em);
                } else {
                    self.update_m_and_m_idx(pm);
                }
            }
            (Some(pm), None) => self.update_m_and_m_idx(pm),
            (None, Some(em)) => self.update_m_and_m_idx(em),
            // Both the entering and previous windows are empty or all null
            (None, None) => self.m = None,
        }

        self.m
    }
    fn is_valid(&self, min_periods: usize) -> bool {
        ((self.last_end - self.last_start) - self.null_count) >= min_periods
    }
}

// This counts as dispatch
macro_rules! minmax_window {
    ($m_window:tt, $rolling_fn:ident, $sort_end_order:literal) => {
        pub struct $m_window<'a, T: NativeType + PartialOrd + IsFloat> {
            the_window: MinMaxWindow<'a, T, $sort_end_order>,
        }

        impl<'a, T: NativeType + IsFloat + PartialOrd> RollingAggWindowNulls<'a, T>
            for $m_window<'a, T>
        {
            unsafe fn new(
                slice: &'a [T],
                validity: &'a Bitmap,
                start: usize,
                end: usize,
                _params: DynArgs,
            ) -> Self {
                Self {
                    the_window: MinMaxWindow::new(slice, validity, start, end),
                }
            }
            unsafe fn update(&mut self, start: usize, end: usize) -> Option<T> {
                self.the_window.update(start, end)
            }
            fn is_valid(&self, min_periods: usize) -> bool {
                self.the_window.is_valid(min_periods)
            }
        }

        pub fn $rolling_fn<T>(
            arr: &PrimitiveArray<T>,
            window_size: usize,
            min_periods: usize,
            center: bool,
            weights: Option<&[f64]>,
            _params: DynArgs,
        ) -> ArrayRef
        where
            T: NativeType + Zero + Copy + PartialOrd + Bounded + IsFloat,
        {
            if weights.is_some() {
                panic!("weights not yet supported on array with null values")
            }
            let offset_fn = match center {
                true => det_offsets_center,
                false => det_offsets,
            };
            rolling_apply_agg_window::<$m_window<_>, _, _>(
                arr.values().as_slice(),
                arr.validity().as_ref().unwrap(),
                window_size,
                min_periods,
                offset_fn,
                None,
            )
        }
    };
}

minmax_window!(MinWindow, rolling_min, 1);
minmax_window!(MaxWindow, rolling_max, -1);
