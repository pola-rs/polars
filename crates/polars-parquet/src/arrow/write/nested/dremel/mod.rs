//! Implements the Dremel encoding part of Parquet with *repetition-levels* and *definition-levels*

use polars_arrow::bitmap::Bitmap;
use polars_arrow::bitmap::utils::BitmapIter;
use polars_arrow::offset::Offset;

use super::super::pages::{ListNested, Nested};

#[cfg(test)]
mod tests;

/// Integer type used to store levels. `u8` is enough for almost all schemas and halves the
/// memory of the level buffers.
pub trait Level:
    Copy + Default + PartialEq + std::ops::Add<Output = Self> + From<bool> + Into<u64>
{
}

impl Level for u8 {}
impl Level for u16 {}

/// The *repetition-levels* and *definition-levels* of a leaf column.
pub struct Levels<T: Level> {
    /// Empty if the repetition-levels are not tracked.
    pub rep: Vec<T>,
    pub def: Vec<T>,
    track_rep: bool,
}

impl<T: Level> Levels<T> {
    /// Computes the levels of the leaf described by `nested` in a single pass.
    ///
    /// The repetition-levels are only computed if `track_rep` is set.
    pub fn new(nested: &[Nested], track_rep: bool) -> Self {
        let mut levels = Self {
            rep: Vec::new(),
            def: Vec::new(),
            track_rep,
        };
        if let Some(first) = nested.first() {
            let zero = T::default();
            levels.extend(nested, 0, first.len(), zero, zero, zero);
        }
        levels
    }

    pub fn num_values(&self) -> usize {
        self.def.len()
    }

    /// Pushes `n` values that all have definition-level `def`. The first one gets repetition-level
    /// `first_rep`, the others `rep`.
    fn push_constant(&mut self, n: usize, first_rep: T, rep: T, def: T) {
        if n == 0 {
            return;
        }
        self.push_rep(n, first_rep, rep);
        self.def.resize(self.def.len() + n, def);
    }

    /// Pushes `n > 0` repetition-levels. The first one is `first_rep`, the others `rep`.
    fn push_rep(&mut self, n: usize, first_rep: T, rep: T) {
        if self.track_rep {
            self.rep.push(first_rep);
            self.rep.resize(self.rep.len() + n - 1, rep);
        }
    }

    /// Adds the levels of the items `start..end` of `nested[0]`.
    ///
    /// - `def` is the definition-level when all ancestors of `nested[0]` are defined.
    /// - `first_rep` is the repetition-level of the first value.
    /// - `rep` is the repetition-level of all other values, which is the number of lists above
    ///   `nested[0]`.
    fn extend(
        &mut self,
        nested: &[Nested],
        start: usize,
        end: usize,
        def: T,
        first_rep: T,
        rep: T,
    ) {
        let Some((head, rest)) = nested.split_first() else {
            return;
        };
        if start == end {
            return;
        }

        match head {
            Nested::Primitive(n) => {
                let len = end - start;
                match optional_validity(n.is_optional, n.validity.as_ref()) {
                    Some(validity) => {
                        self.push_rep(len, first_rep, rep);
                        let (bytes, offset, _) = validity.as_slice();
                        self.def.extend(
                            BitmapIter::new(bytes, offset + start, len)
                                .map(|is_valid| def + T::from(is_valid)),
                        );
                    },
                    None => self.push_constant(len, first_rep, rep, def + T::from(n.is_optional)),
                }
            },
            Nested::Struct(n) => {
                let child_def = def + T::from(n.is_optional);
                let Some(validity) = optional_validity(n.is_optional, n.validity.as_ref()) else {
                    self.extend(rest, start, end, child_def, first_rep, rep);
                    return;
                };

                let (bytes, offset, _) = validity.as_slice();
                let mut iter = BitmapIter::new(bytes, offset + start, end - start);
                let mut i = start;
                let mut run_rep = first_rep;
                while iter.num_remaining() > 0 {
                    let num_valid = iter.take_leading_ones();
                    if num_valid > 0 {
                        self.extend(rest, i, i + num_valid, child_def, run_rep, rep);
                        i += num_valid;
                        run_rep = rep;
                    }

                    let num_invalid = iter.take_leading_zeros();
                    if num_invalid > 0 {
                        self.push_constant(num_invalid, run_rep, rep, def);
                        i += num_invalid;
                        run_rep = rep;
                    }
                }
            },
            Nested::List(n) => self.extend_list(rest, n, start, end, def, first_rep, rep),
            Nested::LargeList(n) => self.extend_list(rest, n, start, end, def, first_rep, rep),
            Nested::FixedSizeList(n) => {
                let validity = optional_validity(n.is_optional, n.validity.as_ref());
                let list_def = def + T::from(n.is_optional);
                let mut item_rep = first_rep;
                for i in start..end {
                    // SAFETY: `i` is smaller than the number of items on this level.
                    let is_valid = validity.is_none_or(|v| unsafe { v.get_bit_unchecked(i) });
                    if !is_valid {
                        self.push_constant(1, item_rep, rep, def);
                    } else if n.width == 0 {
                        self.push_constant(1, item_rep, rep, list_def);
                    } else {
                        let child_start = i * n.width;
                        self.extend(
                            rest,
                            child_start,
                            child_start + n.width,
                            list_def + T::from(true),
                            item_rep,
                            rep + T::from(true),
                        );
                    }
                    item_rep = rep;
                }
            },
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn extend_list<O: Offset>(
        &mut self,
        rest: &[Nested],
        n: &ListNested<O>,
        start: usize,
        end: usize,
        def: T,
        first_rep: T,
        rep: T,
    ) {
        let validity = optional_validity(n.is_optional, n.validity.as_ref());
        let list_def = def + T::from(n.is_optional);
        let offsets = n.offsets.as_slice();
        // The children are sliced to start at the first offset.
        let base = offsets[0].to_usize();
        let mut item_rep = first_rep;
        for i in start..end {
            // SAFETY: `i` is smaller than the number of items on this level.
            let is_valid = validity.is_none_or(|v| unsafe { v.get_bit_unchecked(i) });
            let child_start = offsets[i].to_usize() - base;
            let child_end = offsets[i + 1].to_usize() - base;
            if !is_valid {
                self.push_constant(1, item_rep, rep, def);
            } else if child_start == child_end {
                self.push_constant(1, item_rep, rep, list_def);
            } else {
                self.extend(
                    rest,
                    child_start,
                    child_end,
                    list_def + T::from(true),
                    item_rep,
                    rep + T::from(true),
                );
            }
            item_rep = rep;
        }
    }
}

/// Returns the validity if it has to be checked to compute the definition-levels.
fn optional_validity(is_optional: bool, validity: Option<&Bitmap>) -> Option<&Bitmap> {
    validity.filter(|v| is_optional && v.unset_bits() > 0)
}
