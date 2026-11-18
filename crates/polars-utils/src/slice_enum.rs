use std::num::NonZeroUsize;
use std::ops::{Index, IndexMut, Range};

use crate::IdxSize;
use crate::pl_str::PlSmallStr;

/// A selection of rows, resolved against the height of its input once that is known.
///
/// A slice from 0 that runs to the end has no variant; `Option<Slice>::None` stands for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum Slice {
    /// Rows `offset..offset + len`.
    Positive { offset: usize, len: usize },
    /// At most `len` rows, starting `offset_from_end` rows before the end.
    ///
    /// A `len` above `offset_from_end` selects the same rows as `len == offset_from_end`.
    Negative { offset_from_end: usize, len: usize },
    /// Rows `offset..`.
    PositiveUnbounded { offset: NonZeroUsize },
}

impl Slice {
    pub const EMPTY: Slice = Slice::Positive { offset: 0, len: 0 };

    /// Create the slice of `len` rows from `offset`, counting from the end if `offset` is
    /// negative. Run to the end if `len` is `None`. Return `None` if that selects every row.
    #[inline]
    pub fn new(offset: i64, len: Option<usize>) -> Option<Self> {
        let abs_offset = usize::try_from(offset.unsigned_abs()).unwrap_or(usize::MAX);

        match (offset >= 0, len) {
            (true, Some(len)) => Some(Slice::Positive {
                offset: abs_offset,
                len,
            }),
            (true, None) => Slice::skip(abs_offset),
            (false, len) => Some(Slice::Negative {
                offset_from_end: abs_offset,
                len: len.map_or(abs_offset, |len| len.min(abs_offset)),
            }),
        }
        .map(Self::canonical)
    }

    /// Return the position `offset` refers to in `n_rows` rows, counting from the end if `offset`
    /// is negative, clamped to `0..=n_rows`.
    #[inline]
    pub fn resolve_offset(offset: i64, n_rows: usize) -> usize {
        Slice::new(offset, None).map_or(0, |s| s.resolve(n_rows).start)
    }

    /// Create the slice of `len` rows from `offset`. Run to the end if `len` is `None`. Return
    /// `None` if that selects every row.
    #[inline]
    pub fn new_positive(offset: usize, len: Option<usize>) -> Option<Self> {
        match len {
            Some(len) => Some(Slice::Positive { offset, len }),
            None => Slice::skip(offset),
        }
    }

    /// Create the slice of the first `n` rows.
    #[inline]
    pub fn head(n: usize) -> Self {
        Slice::Positive { offset: 0, len: n }
    }

    /// Create the slice of the last `n` rows.
    #[inline]
    pub fn tail(n: usize) -> Self {
        Slice::Negative {
            offset_from_end: n,
            len: n,
        }
    }

    /// Create the slice of all rows after the first `n`.
    ///
    /// This returns `None` if `n` is 0.
    #[inline]
    pub fn skip(n: usize) -> Option<Self> {
        NonZeroUsize::new(n).map(|offset| Slice::PositiveUnbounded { offset })
    }

    /// Return the maximum number of rows selected, or `None` if the slice runs to the end.
    #[inline]
    pub fn len(&self) -> Option<usize> {
        match self.canonical() {
            Slice::Positive { len, .. } | Slice::Negative { len, .. } => Some(len),
            Slice::PositiveUnbounded { .. } => None,
        }
    }

    /// Return whether no rows are selected at any height.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == Some(0)
    }

    /// Return whether the slice counts from the start.
    #[inline]
    pub fn is_positive(&self) -> bool {
        !self.is_negative()
    }

    /// Return whether the slice counts from the end.
    #[inline]
    pub fn is_negative(&self) -> bool {
        matches!(self, Slice::Negative { .. })
    }

    /// Return the offset of a positive slice.
    ///
    /// # Panics
    /// Panics if `self` is [`Slice::Negative`].
    pub fn positive_offset(&self) -> usize {
        match self {
            Slice::Positive { offset, .. } => *offset,
            Slice::PositiveUnbounded { offset } => offset.get(),
            Slice::Negative { .. } => panic!("cannot use positive_offset() on a negative slice"),
        }
    }

    /// Return the number of leading rows needed to apply the slice, or `None` if that depends
    /// on the height.
    pub fn end_position(&self) -> Option<usize> {
        match self {
            Slice::Positive { offset, len } => offset.checked_add(*len),
            Slice::PositiveUnbounded { .. } | Slice::Negative { .. } => None,
        }
    }

    /// Return whether the signed offset and the end of the slice fit in an `i64`.
    pub fn is_within_i64(&self) -> bool {
        match *self {
            Slice::Positive { .. } => self
                .end_position()
                .is_some_and(|end| i64::try_from(end).is_ok()),
            Slice::PositiveUnbounded { offset } => i64::try_from(offset.get()).is_ok(),
            Slice::Negative {
                offset_from_end, ..
            } => 0i64.checked_sub_unsigned(offset_from_end as u64).is_some(),
        }
    }

    /// Return the smallest `head` or `tail` that contains the slice, or `None` if the slice
    /// may need every row.
    pub fn bounding_limit(&self) -> Option<Slice> {
        match self {
            Slice::Positive { .. } => self.end_position().map(Slice::head),
            Slice::Negative {
                offset_from_end, ..
            } => Some(Slice::tail(*offset_from_end)),
            Slice::PositiveUnbounded { .. } => None,
        }
    }

    /// Return the equivalent slice to apply to the rows from `position` on, or `None` if that
    /// would select every row.
    ///
    /// # Panics
    /// Panics if `self` is [`Slice::Negative`].
    pub fn offsetted(self, position: usize) -> Option<Self> {
        match self {
            Slice::Positive { offset, len } => {
                let (offset, len) = if position <= offset {
                    (offset - position, len)
                } else {
                    (0, len.saturating_sub(position - offset))
                };
                Some(Slice::Positive { offset, len })
            },
            Slice::PositiveUnbounded { offset } => {
                Slice::skip(offset.get().saturating_sub(position))
            },
            Slice::Negative { .. } => panic!("cannot use offsetted() on a negative slice"),
        }
    }

    /// Return which rows the slice selects from `n_rows` rows.
    #[inline]
    pub fn resolve(self, n_rows: usize) -> Range<usize> {
        match self.canonical() {
            Slice::Positive { offset, len } => {
                offset.min(n_rows)..offset.saturating_add(len).min(n_rows)
            },
            Slice::PositiveUnbounded { offset } => offset.get().min(n_rows)..n_rows,
            Slice::Negative {
                offset_from_end,
                len,
            } => {
                n_rows.saturating_sub(offset_from_end)..n_rows.saturating_sub(offset_from_end - len)
            },
        }
    }

    /// Return the positive slice that selects the same rows from `n_rows` rows.
    #[inline]
    pub fn restrict_to_bounds(self, n_rows: usize) -> Self {
        Slice::from(self.resolve(n_rows))
    }

    /// Return the signed offset and the length, `None` if the slice runs to the end.
    ///
    /// # Panics
    /// Panics if the signed offset does not fit in an `i64`.
    pub fn to_signed_offset_len(&self) -> (i64, Option<usize>) {
        let (offset, len) = match self.canonical() {
            Slice::Positive { offset, len } => (i64::try_from(offset).ok(), Some(len)),
            Slice::PositiveUnbounded { offset } => (i64::try_from(offset.get()).ok(), None),
            Slice::Negative {
                offset_from_end,
                len,
            } => (0i64.checked_sub_unsigned(offset_from_end as u64), Some(len)),
        };
        (offset.expect("slice offset does not fit in an i64"), len)
    }

    /// Return the offset and the length of a positive slice, `None` if the slice runs to the end.
    ///
    /// # Panics
    /// Panics if `self` is [`Slice::Negative`].
    pub fn to_unsigned_offset_len(&self) -> (usize, Option<usize>) {
        (self.positive_offset(), self.len())
    }

    /// Format the slice as `(offset, len)`, writing `∞` for the length of a slice that runs to the
    /// end.
    pub fn fmt_tuple(&self) -> PlSmallStr {
        let (offset, len) = self.to_signed_offset_len();
        match len {
            Some(len) => crate::format_pl_smallstr!("({offset}, {len})"),
            None => crate::format_pl_smallstr!("({offset}, ∞)"),
        }
    }

    /// Return the signed offset and the length, writing the length of a slice that runs to the
    /// end as `IdxSize::MAX`.
    // TODO: @3.0: Remove once the Python IR visitor takes an optional length.
    pub fn to_legacy_signed_offset_len(&self) -> (i64, IdxSize) {
        let (offset, len) = self.to_signed_offset_len();
        let len = len.map_or(IdxSize::MAX, |len| {
            IdxSize::try_from(len).unwrap_or(IdxSize::MAX)
        });
        (offset, len)
    }

    /// Return the single slice that selects the same rows as applying `self` and then `next`,
    /// or `None` if that cannot be known statically or if it does not fit in an `i64`.
    pub fn reslice(self, next: Slice) -> Option<Slice> {
        use Slice::*;

        if self.is_empty() || next.is_empty() {
            return Some(Self::EMPTY);
        }

        let combined = match (self.canonical(), next.canonical()) {
            (Positive { offset: a, len: l }, Positive { offset: b, len: m }) => Positive {
                offset: a.checked_add(b)?,
                len: m.min(l.saturating_sub(b)),
            },
            (Positive { offset: a, len: l }, PositiveUnbounded { offset: b }) => Positive {
                offset: a.checked_add(b.get())?,
                len: l.saturating_sub(b.get()),
            },
            (PositiveUnbounded { offset: a }, Positive { offset: b, len: m }) => Positive {
                offset: a.get().checked_add(b)?,
                len: m,
            },
            (PositiveUnbounded { offset: a }, PositiveUnbounded { offset: b }) => {
                PositiveUnbounded {
                    offset: a.checked_add(b.get())?,
                }
            },
            (first @ Negative { len: l, .. }, Positive { offset: 0, len: m }) if m >= l => first,
            (
                Negative {
                    offset_from_end: k,
                    len: l,
                },
                Negative {
                    offset_from_end: k2,
                    len: m2,
                },
            ) => {
                if k2 >= l {
                    Negative {
                        offset_from_end: k,
                        len: l.saturating_sub(k2 - m2),
                    }
                } else {
                    Negative {
                        offset_from_end: k - l + k2,
                        len: m2,
                    }
                }
            },
            _ => return None,
        };

        if combined.is_empty() {
            return Some(Self::EMPTY);
        }
        combined.is_within_i64().then_some(combined)
    }

    /// Return the smallest slice that contains both `a` and `b` at any height, or `None` if
    /// one counts from the start and the other from the end, or if only every row contains both.
    pub fn try_new_enclosing(a: Slice, b: Slice) -> Option<Slice> {
        use Slice::*;

        if a.is_empty() {
            return Some(b);
        }
        if b.is_empty() {
            return Some(a);
        }

        match (a.canonical(), b.canonical()) {
            (
                Negative {
                    offset_from_end: k,
                    len: l,
                },
                Negative {
                    offset_from_end: k2,
                    len: l2,
                },
            ) => {
                let offset_from_end = usize::max(k, k2);
                let stop_from_end = usize::min(k - l, k2 - l2);
                Some(Negative {
                    offset_from_end,
                    len: offset_from_end - stop_from_end,
                })
            },
            (Negative { .. }, _) | (_, Negative { .. }) => None,
            (a, b) => {
                let offset = usize::min(a.positive_offset(), b.positive_offset());
                match Option::zip(a.end_position(), b.end_position()) {
                    Some((end_a, end_b)) => Some(Positive {
                        offset,
                        len: usize::max(end_a, end_b) - offset,
                    }),
                    None => Slice::skip(offset),
                }
            },
        }
    }

    /// Return the slice that selects the rows of `self` from the output of `outer`, or `None`
    /// if `outer` does not contain `self` or if both are the same open-ended slice.
    pub fn try_within(self, outer: Slice) -> Option<Slice> {
        use Slice::*;

        if self.is_empty() {
            return Some(Self::EMPTY);
        }

        match (self.canonical(), outer.canonical()) {
            (
                Negative {
                    offset_from_end: k,
                    len: l,
                },
                Negative {
                    offset_from_end: outer_k,
                    len: outer_l,
                },
            ) => {
                let outer_stop_from_end = outer_k - outer_l;
                (k <= outer_k && k - l >= outer_stop_from_end).then(|| Negative {
                    offset_from_end: k - outer_stop_from_end,
                    len: l,
                })
            },
            (Negative { .. }, _) | (_, Negative { .. }) => None,
            (inner, outer) => {
                let offset = inner
                    .positive_offset()
                    .checked_sub(outer.positive_offset())?;
                let is_within_end = match (inner.end_position(), outer.end_position()) {
                    (_, None) => true,
                    (Some(end), Some(outer_end)) => end <= outer_end,
                    (None, Some(_)) => false,
                };
                if !is_within_end {
                    return None;
                }
                match inner {
                    Positive { len, .. } => Some(Positive { offset, len }),
                    _ => Slice::skip(offset),
                }
            },
        }
    }

    #[inline]
    fn canonical(self) -> Self {
        match self {
            Slice::Negative {
                offset_from_end,
                len,
            } => Slice::Negative {
                offset_from_end,
                len: len.min(offset_from_end),
            },
            slice => slice,
        }
    }
}

impl std::fmt::Display for Slice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (offset, len) = self.to_signed_offset_len();
        match len {
            Some(len) => write!(f, "offset: {offset}, len: {len}"),
            None => write!(f, "offset: {offset}, len: ∞"),
        }
    }
}

impl From<(i64, usize)> for Slice {
    #[inline]
    fn from((offset, len): (i64, usize)) -> Self {
        Slice::new(offset, Some(len)).unwrap()
    }
}

impl From<Range<usize>> for Slice {
    #[inline]
    fn from(range: Range<usize>) -> Self {
        Slice::Positive {
            offset: range.start,
            len: range.len(),
        }
    }
}

impl From<(usize, usize)> for Slice {
    #[inline]
    fn from((offset, len): (usize, usize)) -> Self {
        Slice::Positive { offset, len }
    }
}

impl<T> Index<Slice> for [T] {
    type Output = [T];

    #[inline]
    fn index(&self, slice: Slice) -> &[T] {
        &self[slice.resolve(self.len())]
    }
}

impl<T> IndexMut<Slice> for [T] {
    #[inline]
    fn index_mut(&mut self, slice: Slice) -> &mut [T] {
        let range = slice.resolve(self.len());
        &mut self[range]
    }
}

impl<T> Index<Slice> for Vec<T> {
    type Output = [T];

    #[inline]
    fn index(&self, slice: Slice) -> &[T] {
        &self.as_slice()[slice]
    }
}

impl<T> IndexMut<Slice> for Vec<T> {
    #[inline]
    fn index_mut(&mut self, slice: Slice) -> &mut [T] {
        &mut self.as_mut_slice()[slice]
    }
}

#[cfg(test)]
mod tests {
    use super::Slice;

    #[test]
    fn test_slice_offset() {
        assert_eq!(
            Slice::Positive { offset: 3, len: 10 }.offsetted(1),
            Some(Slice::Positive { offset: 2, len: 10 })
        );
        assert_eq!(
            Slice::Positive { offset: 3, len: 10 }.offsetted(5),
            Some(Slice::Positive { offset: 0, len: 8 })
        );
        assert_eq!(Slice::skip(3).unwrap().offsetted(5), None);
        assert_eq!(Slice::skip(5).unwrap().offsetted(3), Slice::skip(2));
    }

    #[test]
    fn test_slice_resolve() {
        assert_eq!(Slice::Positive { offset: 3, len: 10 }.resolve(7), 3..7,);
        assert_eq!(Slice::Positive { offset: 3, len: 10 }.resolve(0), 0..0,);
        assert_eq!(Slice::Positive { offset: 3, len: 10 }.resolve(1), 1..1,);
        assert_eq!(Slice::Positive { offset: 2, len: 0 }.resolve(10), 2..2,);
        assert_eq!(
            Slice::Negative {
                offset_from_end: 3,
                len: 2
            }
            .resolve(4),
            1..3,
        );
        assert_eq!(
            Slice::Negative {
                offset_from_end: 3,
                len: 2
            }
            .resolve(3),
            0..2,
        );
        assert_eq!(
            Slice::Negative {
                offset_from_end: 3,
                len: 2
            }
            .resolve(2),
            0..1,
        );
        assert_eq!(
            Slice::Negative {
                offset_from_end: 3,
                len: 2
            }
            .resolve(1),
            0..0,
        );
        assert_eq!(Slice::skip(2).unwrap().resolve(5), 2..5,);
    }

    #[test]
    fn test_slice_new() {
        assert_eq!(Slice::new(-3, None), Some(Slice::tail(3)));
        assert_eq!(Slice::new(-3, Some(10)), Some(Slice::tail(3)));
        assert_eq!(Slice::new(5, None), Slice::skip(5));
        assert_eq!(Slice::new(0, None), None);
        assert_eq!(Slice::skip(0), None);
        assert_eq!(
            Slice::from((i64::MIN, 1)).to_signed_offset_len(),
            (i64::MIN, Some(1))
        );
    }

    /// All slices with offsets and lengths up to a small bound, including non-canonical
    /// negative ones.
    fn small_slices() -> Vec<Slice> {
        let mut out = Vec::new();
        for offset in 0..6 {
            out.extend(Slice::skip(offset));
            for len in 0..6 {
                out.push(Slice::Positive { offset, len });
                out.push(Slice::Negative {
                    offset_from_end: offset,
                    len,
                });
            }
        }
        out
    }

    #[test]
    fn test_slice_offsetted() {
        for slice in small_slices().into_iter().filter(|s| s.is_positive()) {
            for position in 0..8 {
                let offsetted = slice.offsetted(position);
                for n_rows in position..12 {
                    let rows: Vec<usize> = (0..n_rows).collect();
                    let rest = &rows[position..];
                    assert_eq!(
                        offsetted.map_or(rest, |s| &rest[s]),
                        rows[slice]
                            .iter()
                            .copied()
                            .filter(|&r| r >= position)
                            .collect::<Vec<_>>(),
                        "{slice:?} from {position} at {n_rows} rows"
                    );
                }
            }
        }
    }

    #[test]
    fn test_slice_try_reslice() {
        for first in small_slices() {
            for next in small_slices() {
                let Some(combined) = first.reslice(next) else {
                    continue;
                };
                for n_rows in 0..12 {
                    let rows: Vec<usize> = (0..n_rows).collect();
                    assert_eq!(
                        &rows[combined], &rows[first][next],
                        "{first:?} then {next:?} at {n_rows} rows"
                    );
                }
            }
        }
    }

    #[test]
    fn test_slice_try_cover_and_within() {
        for a in small_slices() {
            for b in small_slices() {
                let Some(cover) = Slice::try_new_enclosing(a, b) else {
                    let only_all_rows_cover = a.is_positive()
                        && b.is_positive()
                        && usize::min(a.positive_offset(), b.positive_offset()) == 0
                        && (a.len().is_none() || b.len().is_none());
                    assert!(
                        a.is_negative() != b.is_negative() || only_all_rows_cover,
                        "{a:?} {b:?}"
                    );
                    continue;
                };
                for inner in [a, b] {
                    let Some(within) = inner.try_within(cover) else {
                        assert!(
                            inner == cover && inner.len().is_none(),
                            "{inner:?} {cover:?}"
                        );
                        continue;
                    };
                    for n_rows in 0..12 {
                        let rows: Vec<usize> = (0..n_rows).collect();
                        assert_eq!(
                            &rows[cover][within], &rows[inner],
                            "{inner:?} within {cover:?} at {n_rows} rows"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_slice_try_within() {
        for inner in small_slices() {
            for outer in small_slices() {
                let Some(within) = inner.try_within(outer) else {
                    continue;
                };
                for n_rows in 0..12 {
                    let rows: Vec<usize> = (0..n_rows).collect();
                    assert_eq!(
                        &rows[outer][within], &rows[inner],
                        "{inner:?} within {outer:?} at {n_rows} rows"
                    );
                }
            }
        }
    }
}
