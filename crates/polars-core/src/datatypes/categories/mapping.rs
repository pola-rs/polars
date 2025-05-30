use std::fmt;
use std::hash::BuildHasher;
use std::sync::atomic::{AtomicUsize, Ordering};

use polars_error::{PolarsResult, polars_bail};
use polars_utils::aliases::PlSeedableRandomStateQuality;
use polars_utils::parma::raw::RawTable;

use crate::datatypes::CatSize;

pub struct CategoricalMapping {
    str_to_cat: RawTable<str, CatSize>,
    cat_to_str: boxcar::Vec<&'static str>,
    max_categories: usize,
    upper_bound: AtomicUsize,
    hasher: PlSeedableRandomStateQuality,
}

impl CategoricalMapping {
    pub fn new(max_categories: usize) -> Self {
        Self::with_hasher(max_categories, PlSeedableRandomStateQuality::default())
    }

    pub fn with_hasher(max_categories: usize, hasher: PlSeedableRandomStateQuality) -> Self {
        Self {
            str_to_cat: RawTable::default(),
            cat_to_str: boxcar::Vec::default(),
            max_categories,
            upper_bound: AtomicUsize::new(0),
            hasher,
        }
    }

    #[inline(always)]
    pub fn hasher(&self) -> &PlSeedableRandomStateQuality {
        &self.hasher
    }

    /// Try to convert a string to a categorical id, but don't insert it if it is missing.
    #[inline(always)]
    pub fn get_cat(&self, s: &str) -> Option<CatSize> {
        let hash = self.hasher.hash_one(s);
        self.get_cat_with_hash(s, hash)
    }

    /// Same as get_cat, but with the hash pre-computed.
    #[inline(always)]
    pub fn get_cat_with_hash(&self, s: &str, hash: u64) -> Option<CatSize> {
        self.str_to_cat.get(hash, |k| k == s).copied()
    }

    /// Convert a string to a categorical id.
    #[inline(always)]
    pub fn insert_cat(&self, s: &str) -> PolarsResult<CatSize> {
        let hash = self.hasher.hash_one(s);
        self.insert_cat_with_hash(s, hash)
    }

    /// Same as to_cat, but with the hash pre-computed.
    #[inline(always)]
    pub fn insert_cat_with_hash(&self, s: &str, hash: u64) -> PolarsResult<CatSize> {
        self.str_to_cat
            .try_get_or_insert_with(
                hash,
                s,
                |k| k == s,
                |k| {
                    let old_upper_bound = self.upper_bound.fetch_add(1, Ordering::Relaxed);
                    if old_upper_bound + 1 > self.max_categories {
                        self.upper_bound.fetch_sub(1, Ordering::Relaxed);
                        polars_bail!(ComputeError: "attempted to insert more categories than the maximum allowed");
                    }
                    let idx = self
                        .cat_to_str
                        .push(unsafe { core::mem::transmute::<&str, &'static str>(k) });
                    Ok(idx as CatSize)
                },
            )
            .copied()
    }

    /// Try to convert a categorical id to its corresponding string, returning
    /// None if the string is not in the data structure.
    #[inline(always)]
    pub fn cat_to_str(&self, cat: CatSize) -> Option<&str> {
        self.cat_to_str.get(cat as usize).copied()
    }

    /// Get the string corresponding to a categorical id.
    ///
    /// # Safety
    /// The categorical id must have been returned from `to_cat`, and you must
    /// have synchronized with the call which inserted it.
    #[inline(always)]
    pub unsafe fn cat_to_str_unchecked(&self, cat: CatSize) -> &str {
        unsafe { self.cat_to_str.get_unchecked(cat as usize) }
    }

    /// Returns an upper bound such that all strings inserted into the CategoricalMapping
    /// have a categorical id less than it. Note that due to parallel inserts which
    /// you have not synchronized with, there may be gaps when using `from_cat`.
    #[inline(always)]
    pub fn num_cats_upper_bound(&self) -> usize {
        // We need to clamp to self.max_categories because a `fetch_add` may
        // have (temporarily) pushed it beyond the max allowed.
        self.upper_bound
            .load(Ordering::Relaxed)
            .min(self.max_categories)
    }

    /// Returns the number of categories in this mapping.
    #[inline(always)]
    pub fn len(&mut self) -> usize {
        *self.upper_bound.get_mut()
    }

    #[inline(always)]
    pub fn is_empty(&mut self) -> bool {
        self.len() == 0
    }
}

impl fmt::Debug for CategoricalMapping {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CategoricalMapping")
            .field("max_categories", &self.max_categories)
            .field("upper_bound", &self.upper_bound.load(Ordering::Relaxed))
            .finish()
    }
}
