use std::path::Path;

use memchr::memchr2_iter;
use num_traits::Pow;
use polars_core::prelude::*;
use polars_core::{config, POOL};
use polars_error::feature_gated;
use polars_utils::index::Bounded;
use polars_utils::select::select_unpredictable;
use rayon::prelude::*;

use super::buffer::Buffer;
use super::options::{CommentPrefix, NullValuesCompiled};
use super::splitfields::SplitFields;
use super::utils::get_file_chunks;
use super::CsvParseOptions;
use crate::path_utils::is_cloud_url;
use crate::utils::compression::maybe_decompress_bytes;

/// Read the number of rows without parsing columns
/// useful for count(*) queries
pub fn count_rows(
    path: &Path,
    separator: u8,
    quote_char: Option<u8>,
    comment_prefix: Option<&CommentPrefix>,
    eol_char: u8,
    has_header: bool,
) -> PolarsResult<usize> {
    let file = if is_cloud_url(path) || config::force_async() {
        feature_gated!("cloud", {
            crate::file_cache::FILE_CACHE
                .get_entry(path.to_str().unwrap())
                // Safety: This was initialized by schema inference.
                .unwrap()
                .try_open_assume_latest()?
        })
    } else {
        polars_utils::open_file(path)?
    };

    let mmap = unsafe { memmap::Mmap::map(&file).unwrap() };
    let owned = &mut vec![];
    let reader_bytes = maybe_decompress_bytes(mmap.as_ref(), owned)?;

    count_rows_from_slice(
        reader_bytes,
        separator,
        quote_char,
        comment_prefix,
        eol_char,
        has_header,
    )
}

/// Read the number of rows without parsing columns
/// useful for count(*) queries
pub fn count_rows_from_slice(
    mut bytes: &[u8],
    separator: u8,
    quote_char: Option<u8>,
    comment_prefix: Option<&CommentPrefix>,
    eol_char: u8,
    has_header: bool,
) -> PolarsResult<usize> {
    for _ in 0..bytes.len() {
        if bytes[0] != eol_char {
            break;
        }

        bytes = &bytes[1..];
    }

    const MIN_ROWS_PER_THREAD: usize = 1024;
    let max_threads = POOL.current_num_threads();

    // Determine if parallelism is beneficial and how many threads
    let n_threads = get_line_stats(
        bytes,
        MIN_ROWS_PER_THREAD,
        eol_char,
        None,
        separator,
        quote_char,
    )
    .map(|(mean, std)| {
        let n_rows = (bytes.len() as f32 / (mean - 0.01 * std)) as usize;
        (n_rows / MIN_ROWS_PER_THREAD).clamp(1, max_threads)
    })
    .unwrap_or(1);

    let file_chunks: Vec<(usize, usize)> =
        get_file_chunks(bytes, n_threads, None, separator, quote_char, eol_char);

    let iter = file_chunks.into_par_iter().map(|(start, stop)| {
        let local_bytes = &bytes[start..stop];
        let row_iterator = SplitLines::new(local_bytes, quote_char, eol_char, comment_prefix);
        if comment_prefix.is_some() {
            Ok(row_iterator
                .filter(|line| !line.is_empty() && !is_comment_line(line, comment_prefix))
                .count())
        } else {
            Ok(row_iterator.count())
        }
    });

    let count_result: PolarsResult<usize> = POOL.install(|| iter.sum());

    match count_result {
        Ok(val) => Ok(val - (has_header as usize)),
        Err(err) => Err(err),
    }
}

/// Skip the utf-8 Byte Order Mark.
/// credits to csv-core
pub(super) fn skip_bom(input: &[u8]) -> &[u8] {
    if input.len() >= 3 && &input[0..3] == b"\xef\xbb\xbf" {
        &input[3..]
    } else {
        input
    }
}

/// Checks if a line in a CSV file is a comment based on the given comment prefix configuration.
///
/// This function is used during CSV parsing to determine whether a line should be ignored based on its starting characters.
#[inline]
pub(super) fn is_comment_line(line: &[u8], comment_prefix: Option<&CommentPrefix>) -> bool {
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

pub(super) fn skip_lines_naive(mut input: &[u8], eol_char: u8, skip: usize) -> &[u8] {
    for _ in 0..skip {
        if let Some(pos) = next_line_position_naive(input, eol_char) {
            input = &input[pos..];
        } else {
            return input;
        }
    }
    input
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

pub(super) fn is_line_ending(b: u8, eol_char: u8) -> bool {
    b == eol_char || b == b'\r'
}

pub(super) fn is_whitespace(b: u8) -> bool {
    b == b' ' || b == b'\t'
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

#[inline]
pub(super) fn skip_line_ending(input: &[u8], eol_char: u8) -> &[u8] {
    skip_condition(input, |b| is_line_ending(b, eol_char))
}

/// Get the mean and standard deviation of length of lines in bytes
pub(super) fn get_line_stats(
    bytes: &[u8],
    n_lines: usize,
    eol_char: u8,
    expected_fields: Option<usize>,
    separator: u8,
    quote_char: Option<u8>,
) -> Option<(f32, f32)> {
    let mut lengths = Vec::with_capacity(n_lines);

    let mut bytes_trunc;
    let n_lines_per_iter = n_lines / 2;

    let mut n_read = 0;

    // sample from start and 75% in the file
    for offset in [0, (bytes.len() as f32 * 0.75) as usize] {
        bytes_trunc = &bytes[offset..];
        let pos = next_line_position(
            bytes_trunc,
            expected_fields,
            separator,
            quote_char,
            eol_char,
        )?;
        bytes_trunc = &bytes_trunc[pos + 1..];

        for _ in offset..(offset + n_lines_per_iter) {
            let pos = next_line_position_naive(bytes_trunc, eol_char)? + 1;
            n_read += pos;
            lengths.push(pos);
            bytes_trunc = &bytes_trunc[pos..];
        }
    }

    let n_samples = lengths.len();

    let mean = (n_read as f32) / (n_samples as f32);
    let mut std = 0.0;
    for &len in lengths.iter() {
        std += (len as f32 - mean).pow(2.0)
    }
    std = (std / n_samples as f32).sqrt();
    Some((mean, std))
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
pub(super) struct SplitLines<'a> {
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
    pub(super) fn new(
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
}

#[derive(Copy, Clone, Debug)]
pub struct LineStats {
    newline_count: usize,
    last_newline_offset: usize,
    end_inside_string: bool,
}

impl CountLines {
    pub fn new(quote_char: Option<u8>, eol_char: u8) -> Self {
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
        }
    }

    /// Analyzes a chunk of CSV data.
    ///
    /// Returns (newline_count, last_newline_offset, end_inside_string) twice,
    /// the first is assuming the start of the chunk is *not* inside a string,
    /// the second assuming the start is inside a string.
    pub fn analyze_chunk(&self, bytes: &[u8]) -> [LineStats; 2] {
        let mut scan_offset = 0;
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

        // false if even number of quotes seen so far, true otherwise.
        #[allow(unused_assignments)]
        let mut global_quote_parity = false;

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

    pub fn find_next(&self, bytes: &[u8], chunk_size: &mut usize) -> (usize, usize) {
        loop {
            let b = unsafe { bytes.get_unchecked(..(*chunk_size).min(bytes.len())) };

            let (count, offset) = self.count(b);

            if count > 0 || b.len() == bytes.len() {
                return (count, offset);
            }

            *chunk_size *= 2;
        }
    }

    // Returns count and offset in slice
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
///              thread has a different offset.
/// * `projection` - Indices of the columns to project.
/// * `buffers` - Parsed output will be written to these buffers. Except for UTF8 data. The offsets of the
///               fields are written to the buffers. The UTF8 data will be parsed later.
#[allow(clippy::too_many_arguments)]
pub(super) fn parse_lines(
    mut bytes: &[u8],
    parse_options: &CsvParseOptions,
    offset: usize,
    ignore_errors: bool,
    null_values: Option<&NullValuesCompiled>,
    projection: &[usize],
    buffers: &mut [Buffer],
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
                                    bytes = &bytes[read_sol..];
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
