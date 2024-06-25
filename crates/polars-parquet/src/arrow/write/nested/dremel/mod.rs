//! Implements the Dremel encoding part of Parquet with *repetition-levels* and *definition-levels*

use arrow::bitmap::Bitmap;
use arrow::offset::OffsetsBuffer;
use polars_utils::fixedringbuffer::FixedRingBuffer;

use super::super::pages::Nested;

#[cfg(test)]
mod tests;

/// A Dremel encoding value
#[derive(Clone, Copy)]
pub struct DremelValue {
    /// A *repetition-level* value
    pub rep: u16,
    /// A *definition-level* value
    pub def: u16,
}

/// This tries to mirror the Parquet Schema structures, so that is simple to reason about the
/// Dremel structures.
enum LevelContent<'a> {
    /// Always 1 instance
    Required,
    /// Zero or more instances
    Repeated,
    /// Zero or one instance
    Optional(Option<&'a Bitmap>),
}

struct Level<'a> {
    content: LevelContent<'a>,
    /// "Iterator" with number of elements for the next level
    lengths: LevelLength<'a>,
    /// Remaining number of elements to process. NOTE: This is **not** equal to `length - offset`.
    remaining: usize,
    /// Offset into level elements
    offset: usize,
    /// The definition-level associated with this level
    definition_depth: u16,
    /// The repetition-level associated with this level
    repetition_depth: u16,
}

/// This contains the number of elements on the next level for each
enum LevelLength<'a> {
    /// Fixed number of elements based on the validity of this element
    Optional(usize),
    /// Fixed number of elements irregardless of the validity of this element
    Constant(usize),
    /// Variable number of elements and calculated from the difference between two `i32` offsets
    OffsetsI32(&'a OffsetsBuffer<i32>),
    /// Variable number of elements and calculated from the difference between two `i64` offsets
    OffsetsI64(&'a OffsetsBuffer<i64>),
}

/// A iterator for Dremel *repetition* and *definition-levels* in Parquet
///
/// This buffers many consequentive repetition and definition-levels as to not have to branch in
/// and out of this code constantly.
pub struct BufferedDremelIter<'a> {
    buffer: FixedRingBuffer<DremelValue>,

    levels: Box<[Level<'a>]>,
    /// Current offset into `levels` that is being explored
    current_level: usize,

    last_repetition: u16,
}

/// return number values of the nested
pub fn num_values(nested: &[Nested]) -> usize {
    // @TODO: Make this smarter
    //
    // This is not that smart because it is really slow, but not doing this would be:
    // 1. Error prone
    // 2. Repeat much of the logic that you find below
    BufferedDremelIter::new(nested).count()
}

impl<'a> Level<'a> {
    /// Fetch the number of elements given on the next level at `offset` on this level
    fn next_level_length(&self, offset: usize, is_valid: bool) -> usize {
        match self.lengths {
            LevelLength::Optional(n) if is_valid => n,
            LevelLength::Optional(_) => 0,
            LevelLength::Constant(n) => n,
            LevelLength::OffsetsI32(n) => n.length_at(offset),
            LevelLength::OffsetsI64(n) => n.length_at(offset),
        }
    }
}

impl<'a> BufferedDremelIter<'a> {
    // @NOTE: This can maybe just directly be gotten from the Field and array, this double
    // conversion seems rather wasteful.
    /// Create a new [`BufferedDremelIter`] from a set of nested structures
    ///
    /// This creates a structure that resembles (but is not exactly the same) the Parquet schema,
    /// we can then iterate this quite well.
    pub fn new(nested: &'a [Nested]) -> Self {
        let mut levels = Vec::with_capacity(nested.len() * 2 - 1);

        let mut definition_depth = 0u16;
        let mut repetition_depth = 0u16;
        for n in nested {
            match n {
                Nested::Primitive(n) => {
                    let (content, lengths) = if n.is_optional {
                        definition_depth += 1;
                        (
                            LevelContent::Optional(n.validity.as_ref()),
                            LevelLength::Optional(1),
                        )
                    } else {
                        (LevelContent::Required, LevelLength::Constant(1))
                    };

                    levels.push(Level {
                        content,
                        lengths,
                        remaining: n.length,
                        offset: 0,
                        definition_depth,
                        repetition_depth,
                    });
                },
                Nested::List(n) => {
                    if n.is_optional {
                        definition_depth += 1;
                        levels.push(Level {
                            content: LevelContent::Optional(n.validity.as_ref()),
                            lengths: LevelLength::Constant(1),
                            remaining: n.offsets.len_proxy(),
                            offset: 0,
                            definition_depth,
                            repetition_depth,
                        });
                    }

                    definition_depth += 1;
                    levels.push(Level {
                        content: LevelContent::Repeated,
                        lengths: LevelLength::OffsetsI32(&n.offsets),
                        remaining: n.offsets.len_proxy(),
                        offset: 0,
                        definition_depth,
                        repetition_depth,
                    });
                    repetition_depth += 1;
                },
                Nested::LargeList(n) => {
                    if n.is_optional {
                        definition_depth += 1;
                        levels.push(Level {
                            content: LevelContent::Optional(n.validity.as_ref()),
                            lengths: LevelLength::Constant(1),
                            remaining: n.offsets.len_proxy(),
                            offset: 0,
                            definition_depth,
                            repetition_depth,
                        });
                    }

                    definition_depth += 1;
                    levels.push(Level {
                        content: LevelContent::Repeated,
                        lengths: LevelLength::OffsetsI64(&n.offsets),
                        remaining: n.offsets.len_proxy(),
                        offset: 0,
                        definition_depth,
                        repetition_depth,
                    });
                    repetition_depth += 1;
                },
                Nested::FixedSizeList(n) => {
                    if n.is_optional {
                        definition_depth += 1;
                        levels.push(Level {
                            content: LevelContent::Optional(n.validity.as_ref()),
                            lengths: LevelLength::Constant(1),
                            remaining: n.length,
                            offset: 0,
                            definition_depth,
                            repetition_depth,
                        });
                    }

                    definition_depth += 1;
                    levels.push(Level {
                        content: LevelContent::Repeated,
                        lengths: LevelLength::Constant(n.width),
                        remaining: n.length,
                        offset: 0,
                        definition_depth,
                        repetition_depth,
                    });
                    repetition_depth += 1;
                },
                Nested::Struct(n) => {
                    let content = if n.is_optional {
                        definition_depth += 1;
                        LevelContent::Optional(n.validity.as_ref())
                    } else {
                        LevelContent::Required
                    };

                    levels.push(Level {
                        content,
                        lengths: LevelLength::Constant(1),
                        remaining: n.length,
                        offset: 0,
                        definition_depth,
                        repetition_depth,
                    });
                },
            };
        }

        let levels = levels.into_boxed_slice();

        Self {
            // This size is rather arbitrary, but it seems good to make it not too, too high as to
            // reduce memory consumption.
            buffer: FixedRingBuffer::new(256),

            levels,
            current_level: 0,
            last_repetition: 0,
        }
    }

    /// Attempt to fill the rest to the buffer with as many values as possible
    fn fill(&mut self) {
        // First exit condition:
        // If the buffer is full stop trying to fetch more values and just pop the first
        // element in the buffer.
        //
        // Second exit condition:
        // We have exhausted all elements at the final level, there are no elements left.
        while !(self.buffer.is_full() || (self.current_level == 0 && self.levels[0].remaining == 0))
        {
            if self.levels[self.current_level].remaining == 0 {
                self.last_repetition = u16::min(
                    self.last_repetition,
                    self.levels[self.current_level - 1].repetition_depth,
                );
                self.current_level -= 1;
                continue;
            }

            let ns = &mut self.levels;
            let lvl = self.current_level;

            let is_last_nesting = ns.len() == self.current_level + 1;

            macro_rules! push_value {
                ($def:expr) => {
                    self.buffer
                        .push(DremelValue {
                            rep: self.last_repetition,
                            def: $def,
                        })
                        .unwrap();
                    self.last_repetition = ns[lvl].repetition_depth;
                };
            }

            let num_done = match (&ns[lvl].content, is_last_nesting) {
                (LevelContent::Required | LevelContent::Optional(None), true) => {
                    push_value!(ns[lvl].definition_depth);

                    1 + self.buffer.fill_repeat(
                        DremelValue {
                            rep: self.last_repetition,
                            def: ns[lvl].definition_depth,
                        },
                        ns[lvl].remaining - 1,
                    )
                },
                (LevelContent::Required, false) => {
                    self.current_level += 1;
                    ns[lvl + 1].remaining = ns[lvl].next_level_length(ns[lvl].offset, true);
                    1
                },

                (LevelContent::Optional(Some(validity)), true) => {
                    let num_possible =
                        usize::min(self.buffer.remaining_capacity(), ns[lvl].remaining);

                    let validity = (*validity).clone().sliced(ns[lvl].offset, num_possible);

                    // @NOTE: maybe, we can do something here with leading zeros
                    for is_valid in validity.iter() {
                        push_value!(ns[lvl].definition_depth - u16::from(!is_valid));
                    }

                    num_possible
                },
                (LevelContent::Optional(None), false) => {
                    let num_possible =
                        usize::min(self.buffer.remaining_capacity(), ns[lvl].remaining);
                    let mut num_done = num_possible;
                    let def = ns[lvl].definition_depth;

                    // @NOTE: maybe, we can do something here with leading zeros
                    for i in 0..num_possible {
                        let next_level_length = ns[lvl].next_level_length(ns[lvl].offset + i, true);

                        if next_level_length == 0 {
                            // Zero-sized (fixed) lists
                            push_value!(def);
                        } else {
                            self.current_level += 1;
                            ns[lvl + 1].remaining = next_level_length;
                            num_done = i + 1;
                            break;
                        }
                    }

                    num_done
                },
                (LevelContent::Optional(Some(validity)), false) => {
                    let mut num_done = 0;
                    let num_possible =
                        usize::min(self.buffer.remaining_capacity(), ns[lvl].remaining);
                    let def = ns[lvl].definition_depth;

                    let validity = (*validity).clone().sliced(ns[lvl].offset, num_possible);

                    // @NOTE: we can do something here with trailing ones and trailing zeros
                    for is_valid in validity.iter() {
                        num_done += 1;
                        let next_level_length =
                            ns[lvl].next_level_length(ns[lvl].offset + num_done - 1, is_valid);

                        match (is_valid, next_level_length) {
                            (true, 0) => {
                                // Zero-sized (fixed) lists
                                push_value!(def);
                            },
                            (true, _) => {
                                self.current_level += 1;
                                ns[lvl + 1].remaining = next_level_length;
                                break;
                            },
                            (false, 0) => {
                                push_value!(def - 1);
                            },
                            (false, _) => {
                                ns[lvl + 1].remaining = next_level_length;

                                // @NOTE:
                                // This is needed for structs and fixed-size lists. These will have
                                // a non-zero length even if they are invalid. In that case, we
                                // need to skip over all the elements that would have been read if
                                // it was valid.
                                let mut embed_lvl = lvl + 1;
                                'embed: while embed_lvl > lvl {
                                    if embed_lvl == ns.len() - 1 {
                                        ns[embed_lvl].offset += ns[embed_lvl].remaining;
                                    } else {
                                        while ns[embed_lvl].remaining > 0 {
                                            let length = ns[embed_lvl]
                                                .next_level_length(ns[embed_lvl].offset, false);

                                            ns[embed_lvl].offset += 1;
                                            ns[embed_lvl].remaining -= 1;

                                            if length > 0 {
                                                ns[embed_lvl + 1].remaining = length;
                                                embed_lvl += 1;
                                                continue 'embed;
                                            }
                                        }
                                    }

                                    embed_lvl -= 1;
                                }

                                push_value!(def - 1);
                            },
                        }
                    }

                    num_done
                },
                (LevelContent::Repeated, _) => {
                    debug_assert!(!is_last_nesting);
                    let length = ns[lvl].next_level_length(ns[lvl].offset, true);

                    if length == 0 {
                        push_value!(ns[lvl].definition_depth - 1);
                    } else {
                        self.current_level += 1;
                        ns[lvl + 1].remaining = length;
                    }

                    1
                },
            };

            ns[lvl].offset += num_done;
            ns[lvl].remaining -= num_done;
        }
    }
}

impl<'a> Iterator for BufferedDremelIter<'a> {
    type Item = DremelValue;

    fn next(&mut self) -> Option<Self::Item> {
        // Use an item from the buffer if it is available
        if let Some(item) = self.buffer.pop_front() {
            return Some(item);
        }

        self.fill();
        self.buffer.pop_front()
    }
}
