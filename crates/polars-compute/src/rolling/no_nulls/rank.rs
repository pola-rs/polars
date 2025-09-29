use polars_utils::total_ord::{self, TotalEq, TotalOrd};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use skiplist::OrderedSkipList;

use super::super::rank::*;
use super::*;

#[derive(Debug)]
struct SkipListContainer<T: NativeType + TotalOrd + TotalEq>(T);

impl<T: NativeType> PartialEq for SkipListContainer<T> {
    fn eq(&self, other: &Self) -> bool {
        TotalEq::tot_eq(&self.0, &other.0)
    }
}

impl<T: NativeType> PartialOrd for SkipListContainer<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(TotalOrd::tot_cmp(&self.0, &other.0))
    }
}

#[derive(Debug)]
pub struct RankWindow<'a, T: NativeType> {
    start: usize,
    end: usize,
    slice: &'a [T],
    sl: skiplist::OrderedSkipList<SkipListContainer<T>>,
    method: RankMethod,
    rng: Option<SmallRng>,
}

impl<'a, T> RollingAggWindowNoNulls<'a, T> for RankWindow<'a, T>
where
    T: Debug + NativeType + total_ord::TotalOrd,
{
    fn new(
        slice: &'a [T],
        start: usize,
        end: usize,
        params: Option<RollingFnParams>,
        window_size: Option<usize>,
    ) -> Self {
        // TODO: [amber] LEFT HERE: For some reason, `params` is None here.  Should we insert some default or should it be Some(..)?
        let RollingFnParams::Rank { method, seed } = params.unwrap() else {
            unreachable!("expected RollingFnParams::Rank");
        };
        let sl = OrderedSkipList::with_capacity(window_size.unwrap());
        let mut rng = None;
        if method == RankMethod::Random {
            rng = Some(match seed {
                Some(x) => SmallRng::seed_from_u64(x),
                None => SmallRng::from_os_rng(),
            });
        }

        Self {
            start,
            end,
            slice,
            sl,
            method,
            rng,
        }
    }

    unsafe fn update(&mut self, new_start: usize, new_end: usize) -> Option<T> {
        use std::ops::Bound::*;

        use SkipListContainer as SLC;

        debug_assert!(self.sl.len() == self.end - self.start);
        debug_assert!(self.start <= self.end);
        debug_assert!(self.end <= self.slice.len());
        debug_assert!(new_start <= new_end);
        debug_assert!(new_end <= self.slice.len());
        debug_assert!(self.start <= new_start);
        debug_assert!(self.end <= new_end);

        while self.start < new_start {
            let v = unsafe { *self.slice.get_unchecked(self.start) };
            self.sl.remove(&SLC(v));
            self.start += 1;
        }
        while self.end < new_end {
            let v = unsafe { *self.slice.get_unchecked(self.end) };
            self.sl.insert(SLC(v));
            self.end += 1;
        }

        let cur = unsafe { *self.slice.get_unchecked(new_end) };
        let loe = self.sl.lower_bound(Excluded(&SLC(cur)));
        let loi = self.sl.lower_bound(Included(&SLC(cur)));
        let uoe = self.sl.upper_bound(Excluded(&SLC(cur)));
        let uoi = self.sl.upper_bound(Included(&SLC(cur)));

        dbg!(&self.sl);
        dbg!(&cur);
        dbg!(loe);
        dbg!(loi);
        dbg!(uoe);
        dbg!(uoi);

        return None;

        match self.method {
            RankMethod::Average => todo!(),
            RankMethod::Min => todo!(),
            RankMethod::Max => todo!(),
            RankMethod::Dense => todo!(),
            RankMethod::Ordinal => todo!(),
            RankMethod::Random => todo!(),
        }
    }
}

pub fn rolling_rank<T>(
    values: &[T],
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<&[f64]>,
    params: Option<RollingFnParams>,
) -> PolarsResult<ArrayRef>
where
    T: NativeType + num_traits::Num,
{
    assert!(weights.is_none(), "weights are not supported for rank");

    let offset_fn = match center {
        true => det_offsets_center,
        false => det_offsets,
    };

    rolling_apply_agg_window::<RankWindow<_>, _, _>(
        values,
        window_size,
        min_periods,
        offset_fn,
        params,
    )
}
