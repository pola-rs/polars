use std::cmp::Ordering;

use arrow::array::Utf8Array;

use crate::prelude::compare_inner::PartialOrdInner;
use crate::prelude::{
    CategoricalChunked, IntoTakeRandom, NumTakeRandomChunked, NumTakeRandomCont,
    NumTakeRandomSingleChunk, PlHashMap, RevMapping, TakeRandBranch3, TakeRandom,
};

type TakeCats<'a> = TakeRandBranch3<
    NumTakeRandomCont<'a, u32>,
    NumTakeRandomSingleChunk<'a, u32>,
    NumTakeRandomChunked<'a, u32>,
>;

pub(crate) struct CategoricalTakeRandomLocal<'a> {
    rev_map: &'a Utf8Array<i64>,
    cats: TakeCats<'a>,
}

impl<'a> CategoricalTakeRandomLocal<'a> {
    pub(crate) fn new(ca: &'a CategoricalChunked) -> Self {
        // should be rechunked upstream
        assert_eq!(ca.logical.chunks.len(), 1, "implementation error");
        if let RevMapping::Local(rev_map) = &**ca.get_rev_map() {
            let cats = ca.logical().take_rand();
            Self { rev_map, cats }
        } else {
            unreachable!()
        }
    }
}

impl PartialOrdInner for CategoricalTakeRandomLocal<'_> {
    unsafe fn cmp_element_unchecked(&self, idx_a: usize, idx_b: usize) -> Ordering {
        let a = self
            .cats
            .get_unchecked(idx_a)
            .map(|cat| self.rev_map.value_unchecked(cat as usize));
        let b = self
            .cats
            .get_unchecked(idx_b)
            .map(|cat| self.rev_map.value_unchecked(cat as usize));
        a.partial_cmp(&b).unwrap()
    }
}

pub(crate) struct CategoricalTakeRandomGlobal<'a> {
    rev_map_part_1: &'a PlHashMap<u32, u32>,
    rev_map_part_2: &'a Utf8Array<i64>,
    cats: TakeCats<'a>,
}
impl<'a> CategoricalTakeRandomGlobal<'a> {
    pub(crate) fn new(ca: &'a CategoricalChunked) -> Self {
        // should be rechunked upstream
        assert_eq!(ca.logical.chunks.len(), 1, "implementation error");
        if let RevMapping::Global(rev_map_part_1, rev_map_part_2, _) = &**ca.get_rev_map() {
            let cats = ca.logical().take_rand();
            Self {
                rev_map_part_1,
                rev_map_part_2,
                cats,
            }
        } else {
            unreachable!()
        }
    }
}

impl PartialOrdInner for CategoricalTakeRandomGlobal<'_> {
    unsafe fn cmp_element_unchecked(&self, idx_a: usize, idx_b: usize) -> Ordering {
        let a = self.cats.get_unchecked(idx_a).map(|cat| {
            let idx = self.rev_map_part_1.get(&cat).unwrap();
            self.rev_map_part_2.value_unchecked(*idx as usize)
        });
        let b = self.cats.get_unchecked(idx_b).map(|cat| {
            let idx = self.rev_map_part_1.get(&cat).unwrap();
            self.rev_map_part_2.value_unchecked(*idx as usize)
        });
        a.partial_cmp(&b).unwrap()
    }
}
