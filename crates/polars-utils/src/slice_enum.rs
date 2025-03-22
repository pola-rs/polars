#[derive(Debug, Clone, PartialEq)]
pub enum Slice {
    /// Or zero
    Positive {
        offset: usize,
        len: usize,
    },
    Negative {
        offset_from_end: usize,
        len: usize,
    },
}

impl Slice {
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        match self {
            Slice::Positive { len, .. } => *len,
            Slice::Negative { len, .. } => *len,
        }
    }

    /// Returns the end position of the slice (offset + len).
    ///
    /// # Panics
    /// Panics if self is negative.
    pub fn end_position(&self) -> usize {
        let Slice::Positive { offset, len } = self.clone() else {
            panic!("cannot use end() on a negative slice");
        };

        offset.saturating_add(len)
    }

    /// Returns the equivalent slice to apply from an offsetted position.
    ///
    /// # Panics
    /// Panics if self is negative.
    pub fn offsetted(self, position: usize) -> Self {
        let Slice::Positive { offset, len } = self else {
            panic!("cannot use offsetted() on a negative slice");
        };

        let (offset, len) = if position <= offset {
            (offset - position, len)
        } else {
            let n_past_offset = position - offset;
            (0, len.saturating_sub(n_past_offset))
        };

        Slice::Positive { offset, len }
    }

    /// Restricts the bounds of the slice to within a number of rows. Negative slices will also
    /// be translated to the positive equivalent.
    pub fn restrict_to_bounds(self, n_rows: usize) -> Self {
        match self {
            Slice::Positive { offset, len } => {
                let offset = offset.min(n_rows);
                let len = len.min(n_rows - offset);
                Slice::Positive { offset, len }
            },
            Slice::Negative {
                offset_from_end,
                len,
            } => {
                if n_rows >= offset_from_end {
                    // Trim extra starting rows
                    let offset = n_rows - offset_from_end;
                    let len = len.min(n_rows - offset);
                    Slice::Positive { offset, len }
                } else {
                    // Slice offset goes past start of data.
                    let stop_at_n_from_end = offset_from_end.saturating_sub(len);
                    let len = n_rows.saturating_sub(stop_at_n_from_end);

                    Slice::Positive { offset: 0, len }
                }
            },
        }
    }
}

impl From<(usize, usize)> for Slice {
    fn from((offset, len): (usize, usize)) -> Self {
        Slice::Positive { offset, len }
    }
}

impl From<(i64, usize)> for Slice {
    fn from((offset, len): (i64, usize)) -> Self {
        if offset >= 0 {
            Slice::Positive {
                offset: usize::try_from(offset).unwrap(),
                len,
            }
        } else {
            Slice::Negative {
                offset_from_end: usize::try_from(-offset).unwrap(),
                len,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Slice;

    #[test]
    fn test_slice_offset() {
        assert_eq!(
            Slice::Positive { offset: 3, len: 10 }.offsetted(1),
            Slice::Positive { offset: 2, len: 10 }
        );
        assert_eq!(
            Slice::Positive { offset: 3, len: 10 }.offsetted(5),
            Slice::Positive { offset: 0, len: 8 }
        );
    }

    #[test]
    fn test_slice_restrict_to_bounds() {
        assert_eq!(
            Slice::Positive { offset: 3, len: 10 }.restrict_to_bounds(7),
            Slice::Positive { offset: 3, len: 4 },
        );
        assert_eq!(
            Slice::Positive { offset: 3, len: 10 }.restrict_to_bounds(0),
            Slice::Positive { offset: 0, len: 0 },
        );
        assert_eq!(
            Slice::Positive { offset: 3, len: 10 }.restrict_to_bounds(1),
            Slice::Positive { offset: 1, len: 0 },
        );
        assert_eq!(
            Slice::Positive { offset: 2, len: 0 }.restrict_to_bounds(10),
            Slice::Positive { offset: 2, len: 0 },
        );
        assert_eq!(
            Slice::Negative {
                offset_from_end: 3,
                len: 1
            }
            .restrict_to_bounds(4),
            Slice::Positive { offset: 1, len: 1 },
        );
        assert_eq!(
            Slice::Negative {
                offset_from_end: 3,
                len: 1
            }
            .restrict_to_bounds(1),
            Slice::Positive { offset: 0, len: 0 },
        );
    }
}
