mod expressions;
pub mod groups;
pub mod hash_keys;
pub mod hot_groups;
pub mod idx_table;
pub mod planner;
pub mod prelude;
pub mod reduce;
pub mod state;

use polars_utils::IdxSize;

pub use crate::planner::{ExpressionConversionState, create_physical_expr};

/// An index where the top bit indicates whether a value should be evicted.
pub struct EvictIdx(IdxSize);

impl EvictIdx {
    #[inline(always)]
    pub fn new(idx: IdxSize, should_evict: bool) -> Self {
        debug_assert!(idx >> (IdxSize::BITS - 1) == 0);
        Self(idx | ((should_evict as IdxSize) << (IdxSize::BITS - 1)))
    }

    #[inline(always)]
    pub fn idx(&self) -> usize {
        (self.0 & ((1 << (IdxSize::BITS - 1)) - 1)) as usize
    }

    #[inline(always)]
    pub fn should_evict(&self) -> bool {
        (self.0 >> (IdxSize::BITS - 1)) != 0
    }
}
