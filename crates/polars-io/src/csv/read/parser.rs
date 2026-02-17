use std::cmp;

use memchr::memchr2_iter;
use polars_buffer::Buffer;
use polars_core::POOL;
use polars_core::prelude::*;
use polars_error::feature_gated;
use polars_utils::mmap::MMapSemaphore;
use polars_utils::pl_path::PlRefPath;
use polars_utils::select::select_unpredictable;
use rayon::prelude::*;

use super::CsvParseOptions;
use super::builder::Builder;
use super::options::{CommentPrefix, NullValuesCompiled};
use super::splitfields::SplitFields;
use crate::csv::read::read_until_start_and_infer_schema;
use crate::prelude::CsvReadOptions;
use crate::utils::compression::CompressedReader;

/// Read the number of rows without parsing columns
/// useful for count(*) queries
#[allow(clippy::too_many_arguments)]
pub fn count_rows(
    path: PlRefPath,
    quote_char: Option<u8>,
    comment_prefix: Option<&CommentPrefix>,
    eol_char: u8,
    has_header: bool,
    skip_lines: usize,
    skip_rows_before_header: usize,
    skip_rows_after_header: usize,
) -> PolarsResult<usize> {
    let file = if path.has_scheme() || polars_config::config().force_async() {
        feature_gated!("cloud", {
            crate::file_cache::FILE_CACHE
                .get_entry(path)
                // Safety: This was initialized by schema inference.
                .unwrap()
                .try_open_assume_latest()?
        })
    } else {
        polars_utils::open_file(path.as_std_path())?
    };

    let mmap = MMapSemaphore::new_from_file(&file).unwrap();

    count_rows_from_slice_par(
        Buffer::from_owner(mmap),
        quote_char,
        comment_prefix,
        eol_char,
        has_header,
        skip_lines,
        skip_rows_before_header,
        skip_rows_after_header,
    )
}

/// Read the number of rows without parsing columns
/// useful for count(*) queries
#[allow(clippy::too_many_arguments)]
pub fn count_rows_from_slice_par(
    buffer: Buffer<u8>,
    quote_char: Option<u8>,
    comment_prefix: Option<&CommentPrefix>,
    eol_char: u8,
    has_header: bool,
    skip_lines: usize,
    skip_rows_before_header: usize,
    skip_rows_after_header: usize,
) -> PolarsResult<usize> {
    let mut reader = CompressedReader::try_new(buffer)?;

    let reader_options = CsvReadOptions {
        parse_options: Arc::new(CsvParseOptions {
            quote_char,
            comment_prefix: comment_prefix.cloned(),
            eol_char,
            ..Default::default()
        }),
        has_header,
        skip_lines,
        skip_rows: skip_rows_before_header,
        skip_rows_after_header,
        ..Default::default()
    };

    let (_, mut leftover) =
        read_until_start_and_infer_schema(&reader_options, None, None, &mut reader)?;

    const BYTES_PER_CHUNK: usize = if cfg!(debug_assertions) {
        128
    } else {
        512 * 1024
    };

    let count = CountLines::new(quote_char, eol_char, comment_prefix.cloned());
    POOL.install(|| {
        let mut states = Vec::new();
        let eof_unterminated_row;

        if comment_prefix.is_none() {
            let mut last_slice = Buffer::new();
            let mut err = None;

            let streaming_iter = std::iter::from_fn(|| {
                let (slice, read_n) = match reader.read_next_slice(&leftover, BYTES_PER_CHUNK) {
                    Ok(tup) => tup,
                    Err(e) => {
                        err = Some(e);
                        return None;
                    },
                };

                leftover = Buffer::new();
                if slice.is_empty() && read_n == 0 {
                    return None;
                }

                last_slice = slice.clone();
                Some(slice)
            });

            states = streaming_iter
                .enumerate()
                .par_bridge()
                .map(|(id, slice)| (count.analyze_chunk(&slice), id))
                .collect::<Vec<_>>();

            if let Some(e) = err {
                return Err(e.into());
            }

            // par_bridge does not guarantee order, but is mostly sorted so `slice::sort` is a
            // decent fit.
            states.sort_by_key(|(_, id)| *id);

            // Technically this is broken if the input has a comment line at the end that is longer
            // than `BYTES_PER_CHUNK`, but in practice this ought to be fine.
            eof_unterminated_row = ends_in_unterminated_row(&last_slice, eol_char, comment_prefix);
        } else {
            // For the non-compressed case this is a zero-copy op.
            // TODO: Implement streaming chunk logic.
            let (bytes, _) = reader.read_next_slice(&leftover, usize::MAX)?;

            let num_chunks = bytes.len().div_ceil(BYTES_PER_CHUNK);
            (0..num_chunks)
                .into_par_iter()
                .map(|chunk_idx| {
                    let mut start_offset = chunk_idx * BYTES_PER_CHUNK;
                    let next_start_offset = (start_offset + BYTES_PER_CHUNK).min(bytes.len());

                    if start_offset != 0 {
                        // Ensure we start at the start of a line.
                        if let Some(nl_off) = bytes[start_offset..next_start_offset]
                            .iter()
                            .position(|b| *b == eol_char)
                        {
                            start_offset += nl_off + 1;
                        } else {
                            return (count.analyze_chunk(&[]), 0);
                        }
                    }

                    let stop_offset = if let Some(nl_off) = bytes[next_start_offset..]
                        .iter()
                        .position(|b| *b == eol_char)
                    {
                        next_start_offset + nl_off + 1
                    } else {
                        bytes.len()
                    };

                    (count.analyze_chunk(&bytes[start_offset..stop_offset]), 0)
                })
                .collect_into_vec(&mut states);

            eof_unterminated_row = ends_in_unterminated_row(&bytes, eol_char, comment_prefix);
        }

        let mut n = 0;
        let mut in_string = false;
        for (pair, _) in states {
            n += pair[in_string as usize].newline_count;
            in_string = pair[in_string as usize].end_inside_string;
        }
        n += eof_unterminated_row as usize;

        Ok(n)
    })
}

/// Checks if a line in a CSV file is a comment based on the given comment prefix configuration.
///
/// This function is used during CSV parsing to determine whether a line should be ignored based on its starting characters.
#[inline]
pub fn is_comment_line(line: &[u8], comment_prefix: Option<&CommentPrefix>) -> bool {
    match comment_prefix {
        Some(CommentPrefix::Single(c)) => line.first() == Some(c),
        Some(CommentPrefix::Multi(s)) => line.starts_with(s.as_bytes()),
        None => false,
    }
}

/// Find the nearest next line position.
/// Does not check for new line characters embedded in String fields.
pub(super) fn next_line_position_naive(input: &[u8], eol_char: u8) -> Option<usize> {
    let pos = memchr::memchr(eol_char, input)? + 1;
    if input.len() - pos == 0 {
        return None;
    }
    Some(pos)
}

/// Find the nearest next line position that is not embedded in a String field.
pub(super) fn next_line_position(
    mut input: &[u8],
    mut expected_fields: Option<usize>,
    separator: u8,
    quote_char: Option<u8>,
    eol_char: u8,
) -> Option<usize> {
    fn accept_line(
        line: &[u8],
        expected_fields: usize,
        separator: u8,
        eol_char: u8,
        quote_char: Option<u8>,
    ) -> bool {
        let mut count = 0usize;
        for (field, _) in SplitFields::new(line, separator, quote_char, eol_char) {
            if memchr2_iter(separator, eol_char, field).count() >= expected_fields {
                return false;
            }
            count += 1;
        }

        // if the latest field is missing
        // e.g.:
        // a,b,c
        // vala,valb,
        // SplitFields returns a count that is 1 less
        // There fore we accept:
        // expected == count
        // and
        // expected == count - 1
        expected_fields.wrapping_sub(count) <= 1
    }

    // we check 3 subsequent lines for `accept_line` before we accept
    // if 3 groups are rejected we reject completely
    let mut rejected_line_groups = 0u8;

    let mut total_pos = 0;
    if input.is_empty() {
        return None;
    }
    let mut lines_checked = 0u8;
    loop {
        if rejected_line_groups >= 3 {
            return None;
        }
        lines_checked = lines_checked.wrapping_add(1);
        // headers might have an extra value
        // So if we have churned through enough lines
        // we try one field less.
        if lines_checked == u8::MAX {
            if let Some(ef) = expected_fields {
                expected_fields = Some(ef.saturating_sub(1))
            }
        };
        let pos = memchr::memchr(eol_char, input)? + 1;
        if input.len() - pos == 0 {
            return None;
        }
        debug_assert!(pos <= input.len());
        let new_input = unsafe { input.get_unchecked(pos..) };
        let mut lines = SplitLines::new(new_input, quote_char, eol_char, None);
        let line = lines.next();

        match (line, expected_fields) {
            // count the fields, and determine if they are equal to what we expect from the schema
            (Some(line), Some(expected_fields)) => {
                if accept_line(line, expected_fields, separator, eol_char, quote_char) {
                    let mut valid = true;
                    for line in lines.take(2) {
                        if !accept_line(line, expected_fields, separator, eol_char, quote_char) {
                            valid = false;
                            break;
                        }
                    }
                    if valid {
                        return Some(total_pos + pos);
                    } else {
                        rejected_line_groups += 1;
                    }
                } else {
                    debug_assert!(pos < input.len());
                    unsafe {
                        input = input.get_unchecked(pos + 1..);
                    }
                    total_pos += pos + 1;
                }
            },
            // don't count the fields
            (Some(_), None) => return Some(total_pos + pos),
            // // no new line found, check latest line (without eol) for number of fields
            _ => return None,
        }
    }
}

#[inline(always)]
pub(super) fn is_whitespace(b: u8) -> bool {
    b == b' ' || b == b'\t'
}

/// May have false-positives, but not false negatives.
#[inline(always)]
pub(super) fn could_be_whitespace_fast(b: u8) -> bool {
    // We're interested in \t (ASCII 9) and " " (ASCII 32), both of which are
    // <= 32. In that range there aren't a lot of other common symbols (besides
    // newline), so this is a quick test which can be worth doing to avoid the
    // exact test.
    b <= 32
}

#[inline]
fn skip_condition<F>(input: &[u8], f: F) -> &[u8]
where
    F: Fn(u8) -> bool,
{
    if input.is_empty() {
        return input;
    }

    let read = input.iter().position(|b| !f(*b)).unwrap_or(input.len());
    &input[read..]
}

/// Remove whitespace from the start of buffer.
/// Makes sure that the bytes stream starts with
///     'field_1,field_2'
/// and not with
///     '\nfield_1,field_1'
#[inline]
pub(super) fn skip_whitespace(input: &[u8]) -> &[u8] {
    skip_condition(input, is_whitespace)
}

/// An adapted version of std::iter::Split.
/// This exists solely because we cannot split the file in lines naively as
///
/// ```text
///    for line in bytes.split(b'\n') {
/// ```
///
/// This will fail when strings fields are have embedded end line characters.
/// For instance: "This is a valid field\nI have multiples lines" is a valid string field, that contains multiple lines.
pub struct SplitLines<'a> {
    v: &'a [u8],
    quote_char: u8,
    eol_char: u8,
    #[cfg(feature = "simd")]
    simd_eol_char: SimdVec,
    #[cfg(feature = "simd")]
    simd_quote_char: SimdVec,
    #[cfg(feature = "simd")]
    previous_valid_eols: u64,
    total_index: usize,
    quoting: bool,
    comment_prefix: Option<&'a CommentPrefix>,
}

#[cfg(feature = "simd")]
const SIMD_SIZE: usize = 64;
#[cfg(feature = "simd")]
use std::simd::prelude::*;

#[cfg(feature = "simd")]
use polars_utils::clmul::prefix_xorsum_inclusive;

#[cfg(feature = "simd")]
type SimdVec = u8x64;

impl<'a> SplitLines<'a> {
    pub fn new(
        slice: &'a [u8],
        quote_char: Option<u8>,
        eol_char: u8,
        comment_prefix: Option<&'a CommentPrefix>,
    ) -> Self {
        let quoting = quote_char.is_some();
        let quote_char = quote_char.unwrap_or(b'\"');
        #[cfg(feature = "simd")]
        let simd_eol_char = SimdVec::splat(eol_char);
        #[cfg(feature = "simd")]
        let simd_quote_char = SimdVec::splat(quote_char);
        Self {
            v: slice,
            quote_char,
            eol_char,
            #[cfg(feature = "simd")]
            simd_eol_char,
            #[cfg(feature = "simd")]
            simd_quote_char,
            #[cfg(feature = "simd")]
            previous_valid_eols: 0,
            total_index: 0,
            quoting,
            comment_prefix,
        }
    }
}

impl<'a> SplitLines<'a> {
    // scalar as in non-simd
    fn next_scalar(&mut self) -> Option<&'a [u8]> {
        if self.v.is_empty() {
            return None;
        }
        if is_comment_line(self.v, self.comment_prefix) {
            return self.next_comment_line();
        }
        {
            let mut pos = 0u32;
            let mut iter = self.v.iter();
            let mut in_field = false;
            loop {
                match iter.next() {
                    Some(&c) => {
                        pos += 1;

                        if self.quoting && c == self.quote_char {
                            // toggle between string field enclosure
                            //      if we encounter a starting '"' -> in_field = true;
                            //      if we encounter a closing '"' -> in_field = false;
                            in_field = !in_field;
                        }
                        // if we are not in a string and we encounter '\n' we can stop at this position.
                        else if c == self.eol_char && !in_field {
                            break;
                        }
                    },
                    None => {
                        let remainder = self.v;
                        self.v = &[];
                        return Some(remainder);
                    },
                }
            }

            unsafe {
                debug_assert!((pos as usize) <= self.v.len());

                // return line up to this position
                let ret = Some(
                    self.v
                        .get_unchecked(..(self.total_index + pos as usize - 1)),
                );
                // skip the '\n' token and update slice.
                self.v = self.v.get_unchecked(self.total_index + pos as usize..);
                ret
            }
        }
    }
    fn next_comment_line(&mut self) -> Option<&'a [u8]> {
        if let Some(pos) = next_line_position_naive(self.v, self.eol_char) {
            unsafe {
                // return line up to this position
                let ret = Some(self.v.get_unchecked(..(pos - 1)));
                // skip the '\n' token and update slice.
                self.v = self.v.get_unchecked(pos..);
                ret
            }
        } else {
            let remainder = self.v;
            self.v = &[];
            Some(remainder)
        }
    }
}

impl<'a> Iterator for SplitLines<'a> {
    type Item = &'a [u8];

    #[inline]
    #[cfg(not(feature = "simd"))]
    fn next(&mut self) -> Option<&'a [u8]> {
        self.next_scalar()
    }

    #[inline]
    #[cfg(feature = "simd")]
    fn next(&mut self) -> Option<&'a [u8]> {
        // First check cached value
        if self.previous_valid_eols != 0 {
            let pos = self.previous_valid_eols.trailing_zeros() as usize;
            self.previous_valid_eols >>= (pos + 1) as u64;

            unsafe {
                debug_assert!((pos) <= self.v.len());

                // return line up to this position
                let ret = Some(self.v.get_unchecked(..pos));
                // skip the '\n' token and update slice.
                self.v = self.v.get_unchecked(pos + 1..);
                return ret;
            }
        }
        if self.v.is_empty() {
            return None;
        }
        if self.comment_prefix.is_some() {
            return self.next_scalar();
        }

        self.total_index = 0;
        let mut not_in_field_previous_iter = true;

        loop {
            let bytes = unsafe { self.v.get_unchecked(self.total_index..) };
            if bytes.len() > SIMD_SIZE {
                let lane: [u8; SIMD_SIZE] = unsafe {
                    bytes
                        .get_unchecked(0..SIMD_SIZE)
                        .try_into()
                        .unwrap_unchecked()
                };
                let simd_bytes = SimdVec::from(lane);
                let eol_mask = simd_bytes.simd_eq(self.simd_eol_char).to_bitmask();

                let valid_eols = if self.quoting {
                    let quote_mask = simd_bytes.simd_eq(self.simd_quote_char).to_bitmask();
                    let mut not_in_quote_field = prefix_xorsum_inclusive(quote_mask);

                    if not_in_field_previous_iter {
                        not_in_quote_field = !not_in_quote_field;
                    }
                    not_in_field_previous_iter = (not_in_quote_field & (1 << (SIMD_SIZE - 1))) > 0;
                    eol_mask & not_in_quote_field
                } else {
                    eol_mask
                };

                if valid_eols != 0 {
                    let pos = valid_eols.trailing_zeros() as usize;
                    if pos == SIMD_SIZE - 1 {
                        self.previous_valid_eols = 0;
                    } else {
                        self.previous_valid_eols = valid_eols >> (pos + 1) as u64;
                    }

                    unsafe {
                        let pos = self.total_index + pos;
                        debug_assert!((pos) <= self.v.len());

                        // return line up to this position
                        let ret = Some(self.v.get_unchecked(..pos));
                        // skip the '\n' token and update slice.
                        self.v = self.v.get_unchecked(pos + 1..);
                        return ret;
                    }
                } else {
                    self.total_index += SIMD_SIZE;
                }
            } else {
                // Denotes if we are in a string field, started with a quote
                let mut in_field = !not_in_field_previous_iter;
                let mut pos = 0u32;
                let mut iter = bytes.iter();
                loop {
                    match iter.next() {
                        Some(&c) => {
                            pos += 1;

                            if self.quoting && c == self.quote_char {
                                // toggle between string field enclosure
                                //      if we encounter a starting '"' -> in_field = true;
                                //      if we encounter a closing '"' -> in_field = false;
                                in_field = !in_field;
                            }
                            // if we are not in a string and we encounter '\n' we can stop at this position.
                            else if c == self.eol_char && !in_field {
                                break;
                            }
                        },
                        None => {
                            let remainder = self.v;
                            self.v = &[];
                            return Some(remainder);
                        },
                    }
                }

                unsafe {
                    debug_assert!((pos as usize) <= self.v.len());

                    // return line up to this position
                    let ret = Some(
                        self.v
                            .get_unchecked(..(self.total_index + pos as usize - 1)),
                    );
                    // skip the '\n' token and update slice.
                    self.v = self.v.get_unchecked(self.total_index + pos as usize..);
                    return ret;
                }
            }
        }
    }
}

pub struct CountLines {
    quote_char: u8,
    eol_char: u8,
    #[cfg(feature = "simd")]
    simd_eol_char: SimdVec,
    #[cfg(feature = "simd")]
    simd_quote_char: SimdVec,
    quoting: bool,
    comment_prefix: Option<CommentPrefix>,
}

#[derive(Copy, Clone, Debug, Default)]
pub struct LineStats {
    pub newline_count: usize,
    pub last_newline_offset: usize,
    pub end_inside_string: bool,
}

impl CountLines {
    pub fn new(
        quote_char: Option<u8>,
        eol_char: u8,
        comment_prefix: Option<CommentPrefix>,
    ) -> Self {
        let quoting = quote_char.is_some();
        let quote_char = quote_char.unwrap_or(b'\"');
        #[cfg(feature = "simd")]
        let simd_eol_char = SimdVec::splat(eol_char);
        #[cfg(feature = "simd")]
        let simd_quote_char = SimdVec::splat(quote_char);
        Self {
            quote_char,
            eol_char,
            #[cfg(feature = "simd")]
            simd_eol_char,
            #[cfg(feature = "simd")]
            simd_quote_char,
            quoting,
            comment_prefix,
        }
    }

    /// Analyzes a chunk of CSV data.
    ///
    /// Returns (newline_count, last_newline_offset, end_inside_string) twice,
    /// the first is assuming the start of the chunk is *not* inside a string,
    /// the second assuming the start is inside a string.
    ///
    /// If comment_prefix is not None the start of bytes must be at the start of
    /// a line (and thus not in the middle of a comment).
    pub fn analyze_chunk(&self, bytes: &[u8]) -> [LineStats; 2] {
        let mut states = [
            LineStats {
                newline_count: 0,
                last_newline_offset: 0,
                end_inside_string: false,
            },
            LineStats {
                newline_count: 0,
                last_newline_offset: 0,
                end_inside_string: false,
            },
        ];

        // If we have to deal with comments we can't use SIMD and have to explicitly do two passes.
        if self.comment_prefix.is_some() {
            states[0] = self.analyze_chunk_with_comment(bytes, false);
            states[1] = self.analyze_chunk_with_comment(bytes, true);
            return states;
        }

        // False if even number of quotes seen so far, true otherwise.
        #[allow(unused_assignments)]
        let mut global_quote_parity = false;
        let mut scan_offset = 0;

        #[cfg(feature = "simd")]
        {
            // 0 if even number of quotes seen so far, u64::MAX otherwise.
            let mut global_quote_parity_mask = 0;
            while scan_offset + 64 <= bytes.len() {
                let block: [u8; 64] = unsafe {
                    bytes
                        .get_unchecked(scan_offset..scan_offset + 64)
                        .try_into()
                        .unwrap_unchecked()
                };
                let simd_bytes = SimdVec::from(block);
                let eol_mask = simd_bytes.simd_eq(self.simd_eol_char).to_bitmask();
                if self.quoting {
                    let quote_mask = simd_bytes.simd_eq(self.simd_quote_char).to_bitmask();
                    let quote_parity =
                        prefix_xorsum_inclusive(quote_mask) ^ global_quote_parity_mask;
                    global_quote_parity_mask = ((quote_parity as i64) >> 63) as u64;

                    let start_outside_string_eol_mask = eol_mask & !quote_parity;
                    states[0].newline_count += start_outside_string_eol_mask.count_ones() as usize;
                    states[0].last_newline_offset = select_unpredictable(
                        start_outside_string_eol_mask != 0,
                        (scan_offset + 63)
                            .wrapping_sub(start_outside_string_eol_mask.leading_zeros() as usize),
                        states[0].last_newline_offset,
                    );

                    let start_inside_string_eol_mask = eol_mask & quote_parity;
                    states[1].newline_count += start_inside_string_eol_mask.count_ones() as usize;
                    states[1].last_newline_offset = select_unpredictable(
                        start_inside_string_eol_mask != 0,
                        (scan_offset + 63)
                            .wrapping_sub(start_inside_string_eol_mask.leading_zeros() as usize),
                        states[1].last_newline_offset,
                    );
                } else {
                    states[0].newline_count += eol_mask.count_ones() as usize;
                    states[0].last_newline_offset = select_unpredictable(
                        eol_mask != 0,
                        (scan_offset + 63).wrapping_sub(eol_mask.leading_zeros() as usize),
                        states[0].last_newline_offset,
                    );
                }

                scan_offset += 64;
            }

            global_quote_parity = global_quote_parity_mask > 0;
        }

        while scan_offset < bytes.len() {
            let c = unsafe { *bytes.get_unchecked(scan_offset) };
            global_quote_parity ^= (c == self.quote_char) & self.quoting;

            let state = &mut states[global_quote_parity as usize];
            state.newline_count += (c == self.eol_char) as usize;
            state.last_newline_offset =
                select_unpredictable(c == self.eol_char, scan_offset, state.last_newline_offset);

            scan_offset += 1;
        }

        states[0].end_inside_string = global_quote_parity;
        states[1].end_inside_string = !global_quote_parity;
        states
    }

    // bytes must begin at the start of a line.
    fn analyze_chunk_with_comment(&self, bytes: &[u8], mut in_string: bool) -> LineStats {
        let pre_s = match self.comment_prefix.as_ref().unwrap() {
            CommentPrefix::Single(pc) => core::slice::from_ref(pc),
            CommentPrefix::Multi(ps) => ps.as_bytes(),
        };

        let mut state = LineStats::default();
        let mut scan_offset = 0;
        while scan_offset < bytes.len() {
            // Skip comment line if needed.
            while bytes[scan_offset..].starts_with(pre_s) {
                scan_offset += pre_s.len();
                let Some(nl_off) = bytes[scan_offset..]
                    .iter()
                    .position(|c| *c == self.eol_char)
                else {
                    break;
                };
                scan_offset += nl_off + 1;
            }

            while scan_offset < bytes.len() {
                let c = unsafe { *bytes.get_unchecked(scan_offset) };
                in_string ^= (c == self.quote_char) & self.quoting;

                if c == self.eol_char && !in_string {
                    state.newline_count += 1;
                    state.last_newline_offset = scan_offset;
                    scan_offset += 1;
                    break;
                } else {
                    scan_offset += 1;
                }
            }
        }

        state.end_inside_string = in_string;
        state
    }

    pub fn find_next(&self, bytes: &[u8], chunk_size: &mut usize) -> (usize, usize) {
        loop {
            let b = unsafe { bytes.get_unchecked(..(*chunk_size).min(bytes.len())) };

            let (count, offset) = if self.comment_prefix.is_some() {
                let stats = self.analyze_chunk_with_comment(b, false);
                (stats.newline_count, stats.last_newline_offset)
            } else {
                self.count(b)
            };

            if count > 0 || b.len() == bytes.len() {
                return (count, offset);
            }

            *chunk_size = chunk_size.saturating_mul(2);
        }
    }

    pub fn count_rows(&self, bytes: &[u8], is_eof: bool) -> (usize, usize) {
        let stats = if self.comment_prefix.is_some() {
            self.analyze_chunk_with_comment(bytes, false)
        } else {
            self.analyze_chunk(bytes)[0]
        };

        let mut count = stats.newline_count;
        let mut offset = stats.last_newline_offset;

        if count > 0 {
            offset = cmp::min(offset + 1, bytes.len());
        } else {
            debug_assert!(offset == 0);
        }

        if is_eof {
            count += ends_in_unterminated_row(bytes, self.eol_char, self.comment_prefix.as_ref())
                as usize;
            offset = bytes.len();
        }

        (count, offset)
    }

    /// Returns count and offset to split for remainder in slice.
    #[cfg(feature = "simd")]
    pub fn count(&self, bytes: &[u8]) -> (usize, usize) {
        let mut total_idx = 0;
        let original_bytes = bytes;
        let mut count = 0;
        let mut position = 0;
        let mut not_in_field_previous_iter = true;

        loop {
            let bytes = unsafe { original_bytes.get_unchecked(total_idx..) };

            if bytes.len() > SIMD_SIZE {
                let lane: [u8; SIMD_SIZE] = unsafe {
                    bytes
                        .get_unchecked(0..SIMD_SIZE)
                        .try_into()
                        .unwrap_unchecked()
                };
                let simd_bytes = SimdVec::from(lane);
                let eol_mask = simd_bytes.simd_eq(self.simd_eol_char).to_bitmask();

                let valid_eols = if self.quoting {
                    let quote_mask = simd_bytes.simd_eq(self.simd_quote_char).to_bitmask();
                    let mut not_in_quote_field = prefix_xorsum_inclusive(quote_mask);

                    if not_in_field_previous_iter {
                        not_in_quote_field = !not_in_quote_field;
                    }
                    not_in_field_previous_iter = (not_in_quote_field & (1 << (SIMD_SIZE - 1))) > 0;
                    eol_mask & not_in_quote_field
                } else {
                    eol_mask
                };

                if valid_eols != 0 {
                    count += valid_eols.count_ones() as usize;
                    position = total_idx + 63 - valid_eols.leading_zeros() as usize;
                    debug_assert_eq!(original_bytes[position], self.eol_char)
                }
                total_idx += SIMD_SIZE;
            } else if bytes.is_empty() {
                debug_assert!(count == 0 || original_bytes[position] == self.eol_char);
                return (count, position);
            } else {
                let (c, o) = self.count_no_simd(bytes, !not_in_field_previous_iter);

                let (count, position) = if c > 0 {
                    (count + c, total_idx + o)
                } else {
                    (count, position)
                };
                debug_assert!(count == 0 || original_bytes[position] == self.eol_char);

                return (count, position);
            }
        }
    }

    #[cfg(not(feature = "simd"))]
    pub fn count(&self, bytes: &[u8]) -> (usize, usize) {
        self.count_no_simd(bytes, false)
    }

    fn count_no_simd(&self, bytes: &[u8], in_field: bool) -> (usize, usize) {
        let iter = bytes.iter();
        let mut in_field = in_field;
        let mut count = 0;
        let mut position = 0;

        for b in iter {
            let c = *b;
            if self.quoting && c == self.quote_char {
                // toggle between string field enclosure
                //      if we encounter a starting '"' -> in_field = true;
                //      if we encounter a closing '"' -> in_field = false;
                in_field = !in_field;
            }
            // If we are not in a string and we encounter '\n' we can stop at this position.
            else if c == self.eol_char && !in_field {
                position = (b as *const _ as usize) - (bytes.as_ptr() as usize);
                count += 1;
            }
        }
        debug_assert!(count == 0 || bytes[position] == self.eol_char);

        (count, position)
    }
}

fn ends_in_unterminated_row(
    bytes: &[u8],
    eol_char: u8,
    comment_prefix: Option<&CommentPrefix>,
) -> bool {
    if !bytes.is_empty() && bytes.last().copied().unwrap() != eol_char {
        // We can do a simple backwards-scan to find the start of last line if it is a
        // comment line, since comment lines can't escape new-lines.
        let last_new_line_post = memchr::memrchr(eol_char, bytes).unwrap_or(0);
        let last_line_is_comment_line = bytes
            .get(last_new_line_post + 1..)
            .map(|line| is_comment_line(line, comment_prefix))
            .unwrap_or(false);

        return !last_line_is_comment_line;
    }

    false
}

#[inline]
fn find_quoted(bytes: &[u8], quote_char: u8, needle: u8) -> Option<usize> {
    let mut in_field = false;

    let mut idx = 0u32;
    // micro optimizations
    #[allow(clippy::explicit_counter_loop)]
    for &c in bytes.iter() {
        if c == quote_char {
            // toggle between string field enclosure
            //      if we encounter a starting '"' -> in_field = true;
            //      if we encounter a closing '"' -> in_field = false;
            in_field = !in_field;
        }

        if !in_field && c == needle {
            return Some(idx as usize);
        }
        idx += 1;
    }
    None
}

#[inline]
pub(super) fn skip_this_line(bytes: &[u8], quote: Option<u8>, eol_char: u8) -> &[u8] {
    let pos = match quote {
        Some(quote) => find_quoted(bytes, quote, eol_char),
        None => bytes.iter().position(|x| *x == eol_char),
    };
    match pos {
        None => &[],
        Some(pos) => &bytes[pos + 1..],
    }
}

#[inline]
pub(super) fn skip_this_line_naive(input: &[u8], eol_char: u8) -> &[u8] {
    if let Some(pos) = next_line_position_naive(input, eol_char) {
        unsafe { input.get_unchecked(pos..) }
    } else {
        &[]
    }
}

/// Parse CSV.
///
/// # Arguments
/// * `bytes` - input to parse
/// * `offset` - offset in bytes in total input. This is 0 if single threaded. If multi-threaded every
///   thread has a different offset.
/// * `projection` - Indices of the columns to project.
/// * `buffers` - Parsed output will be written to these buffers. Except for UTF8 data. The offsets of the
///   fields are written to the buffers. The UTF8 data will be parsed later.
///
/// Returns the number of bytes parsed successfully.
#[allow(clippy::too_many_arguments)]
pub(super) fn parse_lines(
    mut bytes: &[u8],
    parse_options: &CsvParseOptions,
    offset: usize,
    ignore_errors: bool,
    null_values: Option<&NullValuesCompiled>,
    projection: &[usize],
    buffers: &mut [Builder],
    n_lines: usize,
    // length of original schema
    schema_len: usize,
    schema: &Schema,
) -> PolarsResult<usize> {
    assert!(
        !projection.is_empty(),
        "at least one column should be projected"
    );
    let mut truncate_ragged_lines = parse_options.truncate_ragged_lines;
    // During projection pushdown we are not checking other csv fields.
    // This would be very expensive and we don't care as we only want
    // the projected columns.
    if projection.len() != schema_len {
        truncate_ragged_lines = true
    }

    // we use the pointers to track the no of bytes read.
    let start = bytes.as_ptr() as usize;
    let original_bytes_len = bytes.len();
    let n_lines = n_lines as u32;

    let mut line_count = 0u32;
    loop {
        if line_count > n_lines {
            let end = bytes.as_ptr() as usize;
            return Ok(end - start);
        }

        if bytes.is_empty() {
            return Ok(original_bytes_len);
        } else if is_comment_line(bytes, parse_options.comment_prefix.as_ref()) {
            // deal with comments
            let bytes_rem = skip_this_line_naive(bytes, parse_options.eol_char);
            bytes = bytes_rem;
            continue;
        }

        // Every line we only need to parse the columns that are projected.
        // Therefore we check if the idx of the field is in our projected columns.
        // If it is not, we skip the field.
        let mut projection_iter = projection.iter().copied();
        let mut next_projected = unsafe { projection_iter.next().unwrap_unchecked() };
        let mut processed_fields = 0;

        let mut iter = SplitFields::new(
            bytes,
            parse_options.separator,
            parse_options.quote_char,
            parse_options.eol_char,
        );
        let mut idx = 0u32;
        let mut read_sol = 0;
        loop {
            match iter.next() {
                // end of line
                None => {
                    bytes = unsafe { bytes.get_unchecked(std::cmp::min(read_sol, bytes.len())..) };
                    break;
                },
                Some((mut field, needs_escaping)) => {
                    let field_len = field.len();

                    // +1 is the split character that is consumed by the iterator.
                    read_sol += field_len + 1;

                    if idx == next_projected as u32 {
                        // the iterator is finished when it encounters a `\n`
                        // this could be preceded by a '\r'
                        unsafe {
                            if field_len > 0 && *field.get_unchecked(field_len - 1) == b'\r' {
                                field = field.get_unchecked(..field_len - 1);
                            }
                        }

                        debug_assert!(processed_fields < buffers.len());
                        let buf = unsafe {
                            // SAFETY: processed fields index can never exceed the projection indices.
                            buffers.get_unchecked_mut(processed_fields)
                        };
                        let mut add_null = false;

                        // if we have null values argument, check if this field equal null value
                        if let Some(null_values) = null_values {
                            let field = if needs_escaping && !field.is_empty() {
                                unsafe { field.get_unchecked(1..field.len() - 1) }
                            } else {
                                field
                            };

                            // SAFETY:
                            // process fields is in bounds
                            add_null = unsafe { null_values.is_null(field, idx as usize) }
                        }
                        if add_null {
                            buf.add_null(!parse_options.missing_is_null && field.is_empty())
                        } else {
                            buf.add(field, ignore_errors, needs_escaping, parse_options.missing_is_null)
                                .map_err(|e| {
                                    let bytes_offset = offset + field.as_ptr() as usize - start;
                                    let unparsable = String::from_utf8_lossy(field);
                                    let column_name = schema.get_at_index(idx as usize).unwrap().0;
                                    polars_err!(
                                        ComputeError:
                                        "could not parse `{}` as dtype `{}` at column '{}' (column number {})\n\n\
                                        The current offset in the file is {} bytes.\n\
                                        \n\
                                        You might want to try:\n\
                                        - increasing `infer_schema_length` (e.g. `infer_schema_length=10000`),\n\
                                        - specifying correct dtype with the `schema_overrides` argument\n\
                                        - setting `ignore_errors` to `True`,\n\
                                        - adding `{}` to the `null_values` list.\n\n\
                                        Original error: ```{}```",
                                        &unparsable,
                                        buf.dtype(),
                                        column_name,
                                        idx + 1,
                                        bytes_offset,
                                        &unparsable,
                                        e
                                    )
                                })?;
                        }
                        processed_fields += 1;

                        // if we have all projected columns we are done with this line
                        match projection_iter.next() {
                            Some(p) => next_projected = p,
                            None => {
                                if bytes.get(read_sol - 1) == Some(&parse_options.eol_char) {
                                    bytes = unsafe { bytes.get_unchecked(read_sol..) };
                                } else {
                                    if !truncate_ragged_lines && read_sol < bytes.len() {
                                        polars_bail!(ComputeError: r#"found more fields than defined in 'Schema'

Consider setting 'truncate_ragged_lines={}'."#, polars_error::constants::TRUE)
                                    }
                                    let bytes_rem = skip_this_line(
                                        unsafe { bytes.get_unchecked(read_sol - 1..) },
                                        parse_options.quote_char,
                                        parse_options.eol_char,
                                    );
                                    bytes = bytes_rem;
                                }
                                break;
                            },
                        }
                    }
                    idx += 1;
                },
            }
        }

        // there can be lines that miss fields (also the comma values)
        // this means the splitter won't process them.
        // We traverse them to read them as null values.
        while processed_fields < projection.len() {
            debug_assert!(processed_fields < buffers.len());
            let buf = unsafe {
                // SAFETY: processed fields index can never exceed the projection indices.
                buffers.get_unchecked_mut(processed_fields)
            };
            buf.add_null(!parse_options.missing_is_null);
            processed_fields += 1;
        }
        line_count += 1;
    }
}

#[cfg(test)]
mod test {
    use super::SplitLines;

    #[test]
    fn test_splitlines() {
        let input = "1,\"foo\n\"\n2,\"foo\n\"\n";
        let mut lines = SplitLines::new(input.as_bytes(), Some(b'"'), b'\n', None);
        assert_eq!(lines.next(), Some("1,\"foo\n\"".as_bytes()));
        assert_eq!(lines.next(), Some("2,\"foo\n\"".as_bytes()));
        assert_eq!(lines.next(), None);

        let input2 = "1,'foo\n'\n2,'foo\n'\n";
        let mut lines2 = SplitLines::new(input2.as_bytes(), Some(b'\''), b'\n', None);
        assert_eq!(lines2.next(), Some("1,'foo\n'".as_bytes()));
        assert_eq!(lines2.next(), Some("2,'foo\n'".as_bytes()));
        assert_eq!(lines2.next(), None);
    }
}
