use polars_utils::IdxSize;
use polars_utils::relaxed_cell::RelaxedCell;

/// A bit per build row of an [`IdxTable`](super::IdxTable), set once a probe
/// key matched the row. Empty if the unmatched rows are not tracked.
#[derive(Default)]
pub struct MatchedRows {
    words: Vec<RelaxedCell<u64>>,
}

impl MatchedRows {
    /// Makes room for the rows below `num_rows`.
    pub fn grow(&mut self, num_rows: usize) {
        self.words
            .resize_with(num_rows.div_ceil(64), Default::default);
    }

    #[inline(always)]
    pub fn is_marked(&self, row: IdxSize) -> bool {
        self.words
            .get(row as usize / 64)
            .is_some_and(|word| word.load() & (1 << (row % 64)) != 0)
    }

    #[inline(always)]
    pub fn mark(&self, row: IdxSize) {
        if let Some(word) = self.words.get(row as usize / 64) {
            word.fetch_or(1 << (row % 64));
        }
    }

    /// Marks the rows of a key. They are always marked together, so they are
    /// skipped if the first one is marked already.
    #[inline(always)]
    pub fn mark_rows(&self, rows: &[IdxSize]) {
        if rows.first().is_some_and(|first| !self.is_marked(*first)) {
            for row in rows {
                self.mark(*row);
            }
        }
    }

    /// Writes the unmarked rows from row `offset` on into `out`, in row order,
    /// stopping once `out` holds `limit` rows. Returns how many rows were
    /// passed, the offset of the next call moves by that much.
    pub fn unmarked_rows(
        &self,
        num_rows: IdxSize,
        out: &mut Vec<IdxSize>,
        offset: IdxSize,
        limit: IdxSize,
    ) -> IdxSize {
        out.clear();

        // Don't read past the bits if not all rows were inserted while tracking.
        let num_rows = (num_rows as usize).min(self.words.len() * 64);
        let mut row = offset as usize;
        'words: while row < num_rows {
            let base = row & !63;
            let word = unsafe { self.words.get_unchecked(base / 64) }.load();
            let mut unmarked = !word & (u64::MAX << (row - base));
            if num_rows - base < 64 {
                unmarked &= (1 << (num_rows - base)) - 1;
            }
            row = base + 64;
            while unmarked != 0 {
                let bit = unmarked.trailing_zeros() as usize;
                if out.len() >= limit as usize {
                    row = base + bit;
                    break 'words;
                }
                out.push((base + bit) as IdxSize);
                unmarked &= unmarked - 1;
            }
        }
        row.min(num_rows).saturating_sub(offset as usize) as IdxSize
    }
}
