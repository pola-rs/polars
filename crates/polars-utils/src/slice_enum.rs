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

    /// Returns the equivalent slice to apply from an offsetted position.
    ///
    /// # Panics
    /// Panics if self is negative.
    pub fn offsetted(self, position: usize) -> Self {
        let Slice::Positive { offset, len } = self else {
            panic!("expected positive slice");
        };

        let (offset, len) = if position <= offset {
            (offset - position, len)
        } else {
            let n_past_offset = position - offset;
            (0, len.saturating_sub(n_past_offset))
        };

        Slice::Positive { offset, len }
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
                offset: offset as usize,
                len,
            }
        } else {
            Slice::Negative {
                offset_from_end: -offset as usize,
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
}
