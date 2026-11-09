use polars_utils::clmul::prefix_xorsum_inclusive;

const BLOCK_SIZE: usize = 64;

/// Position of a field relative to the start of a batch of lines.
#[derive(Clone, Copy)]
pub(super) struct FieldPos {
    start: u32,
    len: u32,
}

impl FieldPos {
    /// A field that is not in its line.
    pub(super) const MISSING: Self = Self {
        start: u32::MAX,
        len: 0,
    };

    #[inline]
    pub(super) fn new(start: usize, len: usize) -> Option<Self> {
        Some(Self {
            start: u32::try_from(start).ok().filter(|s| *s != u32::MAX)?,
            len: u32::try_from(len).ok()?,
        })
    }

    /// Returns the field in `batch`, or `None` if it is missing.
    ///
    /// # Safety
    /// The field must have been taken from `batch`.
    #[inline]
    pub(super) unsafe fn get(self, batch: &[u8]) -> Option<&[u8]> {
        if self.start == u32::MAX {
            return None;
        }
        let start = self.start as usize;
        Some(unsafe { batch.get_unchecked(start..start + self.len as usize) })
    }
}

/// Finds the projected fields of whole lines with 64 byte masks.
pub(super) struct LineIndexer<'a> {
    pub separator: u8,
    pub eol_char: u8,
    pub quote_char: u8,
    pub quoting: bool,
    pub projection: &'a [usize],
    pub schema_len: usize,
    pub truncate_ragged_lines: bool,
}

#[cfg(feature = "simd")]
#[inline(always)]
fn eq_mask(block: &[u8; BLOCK_SIZE], c: u8) -> u64 {
    use std::simd::prelude::*;
    u8x64::from_array(*block)
        .simd_eq(u8x64::splat(c))
        .to_bitmask()
}

#[cfg(not(feature = "simd"))]
#[inline(always)]
fn eq_mask(block: &[u8; BLOCK_SIZE], c: u8) -> u64 {
    let mut mask = 0u64;
    for (i, b) in block.iter().enumerate() {
        mask |= ((*b == c) as u64) << i;
    }
    mask
}

impl LineIndexer<'_> {
    /// Pushes the positions of the projected fields of the lines at the start of `bytes`, which
    /// must start a line. Positions are relative to `bytes` plus `batch_offset`.
    ///
    /// Stops after `max_lines` lines, once the batch holds `max_batch_bytes` bytes or
    /// `max_fields` fields, or before a line that the field by field scan must parse: one with a
    /// quote char that is neither at a field start nor after a quote char, one with too many
    /// fields when that is an error, one without an end of line char or one whose positions do
    /// not fit in a `FieldPos`.
    ///
    /// Returns the number of bytes and lines indexed.
    pub(super) fn index_lines(
        &self,
        bytes: &[u8],
        batch_offset: usize,
        fields: &mut Vec<FieldPos>,
        max_lines: u32,
        max_batch_bytes: usize,
        max_fields: usize,
    ) -> (usize, u32) {
        let projection = self.projection;
        let mut lines = 0u32;
        let mut line_start = 0;
        let mut line_fields = fields.len();
        let mut field_start = 0;
        let mut field_idx = 0;
        let mut projected = 0;
        let mut next_projected = projection[0];

        let mut in_quotes_carry = 0u64;
        // Bit 0 is set if a quote char at the start of the block may open a quoted field.
        let mut valid_open_carry = 1u64;

        let mut tail = [0u8; BLOCK_SIZE];
        let mut block_start = 0;
        while block_start < bytes.len() {
            let (block, in_bounds) = match bytes.get(block_start..block_start + BLOCK_SIZE) {
                Some(block) => (block.try_into().unwrap(), u64::MAX),
                None => {
                    let n = bytes.len() - block_start;
                    tail[..n].copy_from_slice(&bytes[block_start..]);
                    (&tail, (1u64 << n) - 1)
                },
            };

            let separators = eq_mask(block, self.separator) & in_bounds;
            let eols = eq_mask(block, self.eol_char) & in_bounds;
            let mut ends = separators | eols;
            let mut line_ends = eols;
            let mut stop = u64::MAX;
            if self.quoting {
                let quotes = eq_mask(block, self.quote_char) & in_bounds;
                if quotes | in_quotes_carry != 0 {
                    let in_quotes = prefix_xorsum_inclusive(quotes) ^ in_quotes_carry;
                    in_quotes_carry = ((in_quotes as i64) >> 63) as u64;
                    let valid_opens = ((ends | quotes) << 1) | valid_open_carry;
                    let stray_quotes = quotes & in_quotes & !valid_opens;
                    if stray_quotes != 0 {
                        stop = stray_quotes.isolate_lowest_one().wrapping_sub(1);
                    }
                    ends &= !in_quotes & stop;
                    line_ends &= !in_quotes;
                }
                valid_open_carry = (separators | eols | quotes) >> 63;
            }

            while ends != 0 {
                let bit = ends.trailing_zeros() as usize;
                ends &= ends - 1;
                let end = block_start + bit;

                if field_idx == next_projected {
                    let mut field_end = end;
                    if field_end > field_start && bytes[field_end - 1] == b'\r' {
                        field_end -= 1;
                    }
                    let Some(pos) =
                        FieldPos::new(batch_offset + field_start, field_end - field_start)
                    else {
                        fields.truncate(line_fields);
                        return (line_start, lines);
                    };
                    fields.push(pos);
                    projected += 1;
                    next_projected = projection.get(projected).copied().unwrap_or(usize::MAX);
                }
                field_idx += 1;

                if (line_ends >> bit) & 1 != 0 {
                    if field_idx > self.schema_len && !self.truncate_ragged_lines {
                        fields.truncate(line_fields);
                        return (line_start, lines);
                    }
                    for _ in projected..projection.len() {
                        fields.push(FieldPos::MISSING);
                    }
                    lines += 1;
                    line_start = end + 1;
                    line_fields = fields.len();
                    field_idx = 0;
                    projected = 0;
                    next_projected = projection[0];
                    if lines == max_lines
                        || batch_offset + line_start >= max_batch_bytes
                        || fields.len() >= max_fields
                    {
                        return (line_start, lines);
                    }
                }
                field_start = end + 1;
            }

            if stop != u64::MAX {
                break;
            }
            block_start += BLOCK_SIZE;
        }

        fields.truncate(line_fields);
        (line_start, lines)
    }
}
