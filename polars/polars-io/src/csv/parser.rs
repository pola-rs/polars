use num::traits::Pow;
use polars_core::prelude::*;

use super::buffer::*;
use crate::csv::read::NullValuesCompiled;

/// Skip the utf-8 Byte Order Mark.
/// credits to csv-core
pub(crate) fn skip_bom(input: &[u8]) -> &[u8] {
    if input.len() >= 3 && &input[0..3] == b"\xef\xbb\xbf" {
        &input[3..]
    } else {
        input
    }
}

/// Find the nearest next line position.
/// Does not check for new line characters embedded in String fields.
pub(crate) fn next_line_position_naive(input: &[u8], eol_char: u8) -> Option<usize> {
    let pos = memchr::memchr(eol_char, input)? + 1;
    if input.len() - pos == 0 {
        return None;
    }
    Some(pos)
}

/// Find the nearest next line position that is not embedded in a String field.
pub(crate) fn next_line_position(
    mut input: &[u8],
    mut expected_fields: Option<usize>,
    delimiter: u8,
    quote_char: Option<u8>,
    eol_char: u8,
) -> Option<usize> {
    let mut total_pos = 0;
    if input.is_empty() {
        return None;
    }
    let mut lines_checked = 0u16;
    loop {
        lines_checked += 1;
        // headers might have an extra value
        // So if we have churned through enough lines
        // we try one field less.
        if lines_checked == 256 {
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
        let line = SplitLines::new(new_input, quote_char.unwrap_or(b'"'), eol_char).next();

        let count_fields =
            |line: &[u8]| SplitFields::new(line, delimiter, quote_char, eol_char).count();

        match (line, expected_fields) {
            // count the fields, and determine if they are equal to what we expect from the schema
            (Some(line), Some(expected_fields)) if { count_fields(line) == expected_fields } => {
                return Some(total_pos + pos)
            }
            (Some(_), Some(_)) => {
                debug_assert!(pos < input.len());
                unsafe {
                    input = input.get_unchecked(pos + 1..);
                }
                total_pos += pos + 1;
            }
            // don't count the fields
            (Some(_), None) => return Some(total_pos + pos),
            // no new line found, check latest line (without eol) for number of fields
            (None, Some(expected_fields)) if { count_fields(new_input) == expected_fields } => {
                return Some(total_pos + pos)
            }
            _ => return None,
        }
    }
}

pub(crate) fn is_line_ending(b: u8, eol_char: u8) -> bool {
    b == eol_char || b == b'\r'
}

pub(crate) fn is_whitespace(b: u8) -> bool {
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

/// Makes sure that the bytes stream starts with
///     'field_1,field_2'
/// and not with
///     '\nfield_1,field_1'
pub(crate) fn skip_header(input: &[u8], eol_char: u8) -> (&[u8], usize) {
    match next_line_position_naive(input, eol_char) {
        Some(mut pos) => {
            if input[pos] == eol_char {
                pos += 1;
            }
            (&input[pos..], pos)
        }
        // no lines in the file, so skipping the header is skipping all.
        None => (&[], input.len()),
    }
}

/// Remove whitespace from the start of buffer.
#[inline]
pub(crate) fn skip_whitespace(input: &[u8]) -> &[u8] {
    skip_condition(input, is_whitespace)
}

#[inline]
/// Can be used to skip whitespace, but exclude the delimiter
pub(crate) fn skip_whitespace_exclude(input: &[u8], exclude: u8) -> &[u8] {
    skip_condition(input, |b| b != exclude && (is_whitespace(b)))
}

#[inline]
/// Can be used to skip whitespace, but exclude the delimiter
pub(crate) fn skip_whitespace_line_ending_exclude(
    input: &[u8],
    exclude: u8,
    eol_char: u8,
) -> &[u8] {
    skip_condition(input, |b| {
        b != exclude && (is_whitespace(b) || is_line_ending(b, eol_char))
    })
}

#[inline]
pub(crate) fn skip_line_ending(input: &[u8], eol_char: u8) -> &[u8] {
    skip_condition(input, |b| is_line_ending(b, eol_char))
}

/// Get the mean and standard deviation of length of lines in bytes
pub(crate) fn get_line_stats(
    bytes: &[u8],
    n_lines: usize,
    eol_char: u8,
    expected_fields: usize,
    delimiter: u8,
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
            Some(expected_fields),
            delimiter,
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
pub(crate) struct SplitLines<'a> {
    v: &'a [u8],
    quote_char: u8,
    end_line_char: u8,
}

impl<'a> SplitLines<'a> {
    pub(crate) fn new(slice: &'a [u8], quote_char: u8, end_line_char: u8) -> Self {
        Self {
            v: slice,
            quote_char,
            end_line_char,
        }
    }
}

impl<'a> Iterator for SplitLines<'a> {
    type Item = &'a [u8];

    #[inline]
    fn next(&mut self) -> Option<&'a [u8]> {
        if self.v.is_empty() {
            return None;
        }

        // denotes if we are in a string field, started with a quote
        let mut in_field = false;
        let mut pos = 0u32;
        let mut iter = self.v.iter();
        loop {
            match iter.next() {
                Some(&c) => {
                    pos += 1;

                    if c == self.quote_char {
                        // toggle between string field enclosure
                        //      if we encounter a starting '"' -> in_field = true;
                        //      if we encounter a closing '"' -> in_field = false;
                        in_field = !in_field;
                    }
                    // if we are not in a string and we encounter '\n' we can stop at this position.
                    else if c == self.end_line_char && !in_field {
                        break;
                    }
                }
                None => {
                    // no new line found we are done
                    // the rest will be done by last line specific code.
                    return None;
                }
            }
        }

        unsafe {
            debug_assert!((pos as usize) <= self.v.len());
            // return line up to this position
            let ret = Some(self.v.get_unchecked(..(pos - 1) as usize));
            // skip the '\n' token and update slice.
            self.v = self.v.get_unchecked(pos as usize..);
            ret
        }
    }
}

/// An adapted version of std::iter::Split.
/// This exists solely because we cannot split the lines naively as
pub(crate) struct SplitFields<'a> {
    v: &'a [u8],
    delimiter: u8,
    finished: bool,
    quote_char: u8,
    quoting: bool,
    eol_char: u8,
}

impl<'a> SplitFields<'a> {
    pub(crate) fn new(
        slice: &'a [u8],
        delimiter: u8,
        quote_char: Option<u8>,
        eol_char: u8,
    ) -> Self {
        Self {
            v: slice,
            delimiter,
            finished: false,
            quote_char: quote_char.unwrap_or(b'"'),
            quoting: quote_char.is_some(),
            eol_char,
        }
    }

    unsafe fn finish_eol(&mut self, need_escaping: bool, idx: usize) -> Option<(&'a [u8], bool)> {
        self.finished = true;
        debug_assert!(idx <= self.v.len());
        Some((self.v.get_unchecked(..idx), need_escaping))
    }

    fn finish(&mut self, need_escaping: bool) -> Option<(&'a [u8], bool)> {
        self.finished = true;
        Some((self.v, need_escaping))
    }

    fn eof_oel(&self, current_ch: u8) -> bool {
        current_ch == self.delimiter || current_ch == self.eol_char
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

impl<'a> Iterator for SplitFields<'a> {
    // the bool is used to indicate that it requires escaping
    type Item = (&'a [u8], bool);

    #[inline]
    fn next(&mut self) -> Option<(&'a [u8], bool)> {
        if self.v.is_empty() || self.finished {
            return None;
        }

        let mut needs_escaping = false;
        // There can be strings with delimiters:
        // "Street, City",

        // Safety:
        // we have checked bounds
        let pos = if self.quoting && unsafe { *self.v.get_unchecked(0) } == self.quote_char {
            needs_escaping = true;
            // There can be pair of double-quotes within string.
            // Each of the embedded double-quote characters must be represented
            // by a pair of double-quote characters:
            // e.g. 1997,Ford,E350,"Super, ""luxurious"" truck",20020

            // denotes if we are in a string field, started with a quote
            let mut in_field = false;

            let mut idx = 0u32;
            let mut current_idx = 0u32;
            // micro optimizations
            #[allow(clippy::explicit_counter_loop)]
            for &c in self.v.iter() {
                if c == self.quote_char {
                    // toggle between string field enclosure
                    //      if we encounter a starting '"' -> in_field = true;
                    //      if we encounter a closing '"' -> in_field = false;
                    in_field = !in_field;
                }

                if !in_field && self.eof_oel(c) {
                    if c == self.eol_char {
                        // safety
                        // we are in bounds
                        return unsafe { self.finish_eol(needs_escaping, current_idx as usize) };
                    }
                    idx = current_idx;
                    break;
                }
                current_idx += 1;
            }

            if idx == 0 {
                return self.finish(needs_escaping);
            }

            idx as usize
        } else {
            match memchr::memchr2(self.delimiter, self.eol_char, self.v) {
                None => return self.finish(needs_escaping),
                Some(idx) => unsafe {
                    // Safety:
                    // idx was just found
                    if *self.v.get_unchecked(idx) == self.eol_char {
                        return self.finish_eol(needs_escaping, idx);
                    } else {
                        idx
                    }
                },
            }
        };

        unsafe {
            debug_assert!(pos <= self.v.len());
            // safety
            // we are in bounds
            let ret = Some((self.v.get_unchecked(..pos), needs_escaping));
            self.v = self.v.get_unchecked(pos + 1..);
            ret
        }
    }
}

#[inline]
fn skip_this_line(bytes: &[u8], quote: Option<u8>, eol_char: u8) -> &[u8] {
    let pos = match quote {
        Some(quote) => find_quoted(bytes, quote, eol_char),
        None => bytes.iter().position(|x| *x == eol_char),
    };
    match pos {
        None => &[],
        Some(pos) => &bytes[pos + 1..],
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
pub(super) fn parse_lines<'a>(
    mut bytes: &'a [u8],
    offset: usize,
    delimiter: u8,
    comment_char: Option<u8>,
    quote_char: Option<u8>,
    eol_char: u8,
    null_values: Option<&NullValuesCompiled>,
    missing_is_null: bool,
    projection: &[usize],
    buffers: &mut [Buffer<'a>],
    ignore_errors: bool,
    n_lines: usize,
    // length or original schema
    schema_len: usize,
) -> PolarsResult<usize> {
    assert!(
        !projection.is_empty(),
        "at least one column should be projected"
    );

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

        // only when we have one column \n should not be skipped
        // other widths should have commas.
        bytes = if schema_len > 1 {
            skip_whitespace_line_ending_exclude(bytes, delimiter, eol_char)
        } else {
            skip_whitespace_exclude(bytes, delimiter)
        };
        if bytes.is_empty() {
            return Ok(original_bytes_len);
        }

        // deal with comments
        if let Some(c) = comment_char {
            // line is a comment -> skip
            if bytes[0] == c {
                let bytes_rem = skip_this_line(bytes, quote_char, eol_char);
                bytes = bytes_rem;
                continue;
            }
        }

        // Every line we only need to parse the columns that are projected.
        // Therefore we check if the idx of the field is in our projected columns.
        // If it is not, we skip the field.
        let mut projection_iter = projection.iter().copied();
        let mut next_projected = unsafe { projection_iter.next().unwrap_unchecked() };
        let mut processed_fields = 0;

        let mut iter = SplitFields::new(bytes, delimiter, quote_char, eol_char);
        let mut idx = 0u32;
        let mut read_sol = 0;
        loop {
            match iter.next() {
                // end of line
                None => {
                    bytes = &bytes[std::cmp::min(read_sol, bytes.len())..];
                    break;
                }
                Some((mut field, needs_escaping)) => {
                    idx += 1;
                    let field_len = field.len();

                    // +1 is the split character that is consumed by the iterator.
                    read_sol += field_len + 1;

                    if (idx - 1) == next_projected as u32 {
                        // the iterator is finished when it encounters a `\n`
                        // this could be preceded by a '\r'
                        if field_len > 0 && field[field_len - 1] == b'\r' {
                            field = &field[..field_len - 1];
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
                                &field[1..field.len() - 1]
                            } else {
                                field
                            };

                            // safety:
                            // process fields is in bounds
                            add_null = unsafe { null_values.is_null(field, processed_fields) }
                        }
                        if add_null {
                            buf.add_null(!missing_is_null && field.is_empty())
                        } else {
                            buf.add(field, ignore_errors, needs_escaping, missing_is_null)
                                .map_err(|_| {
                                    let bytes_offset = offset + field.as_ptr() as usize - start;
                                    let unparsable = String::from_utf8_lossy(field);
                                    PolarsError::ComputeError(
                                        format!(
                                            "Could not parse `{}` as dtype {:?} at column {}.\n\
                                            The current offset in the file is {} bytes.\n\
                                            \n\
                                            Consider specifying the correct dtype, increasing\n\
                                            the number of records used to infer the schema,\n\
                                            enabling the `ignore_errors` flag, or adding\n\
                                            `{}` to the `null_values` list.",
                                            &unparsable,
                                            buf.dtype(),
                                            idx,
                                            bytes_offset,
                                            &unparsable,
                                        )
                                        .into(),
                                    )
                                })?;
                        }
                        processed_fields += 1;

                        // if we have all projected columns we are done with this line
                        match projection_iter.next() {
                            Some(p) => next_projected = p,
                            None => {
                                if bytes.get(read_sol - 1) == Some(&eol_char) {
                                    bytes = &bytes[read_sol..];
                                } else {
                                    let bytes_rem = skip_this_line(
                                        &bytes[read_sol - 1..],
                                        quote_char,
                                        eol_char,
                                    );
                                    bytes = bytes_rem;
                                }
                                break;
                            }
                        }
                    }
                }
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
            buf.add_null(!missing_is_null);
            processed_fields += 1;
        }
        line_count += 1;
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_splitfields() {
        let input = "\"foo\",\"bar\"";
        let mut fields = SplitFields::new(input.as_bytes(), b',', Some(b'"'), b'\n');

        assert_eq!(fields.next(), Some(("\"foo\"".as_bytes(), true)));
        assert_eq!(fields.next(), Some(("\"bar\"".as_bytes(), true)));
        assert_eq!(fields.next(), None);

        let input2 = "\"foo\n bar\";\"baz\";12345";
        let mut fields2 = SplitFields::new(input2.as_bytes(), b';', Some(b'"'), b'\n');

        assert_eq!(fields2.next(), Some(("\"foo\n bar\"".as_bytes(), true)));
        assert_eq!(fields2.next(), Some(("\"baz\"".as_bytes(), true)));
        assert_eq!(fields2.next(), Some(("12345".as_bytes(), false)));
        assert_eq!(fields2.next(), None);
    }

    #[test]
    fn test_splitlines() {
        let input = "1,\"foo\n\"\n2,\"foo\n\"\n";
        let mut lines = SplitLines::new(input.as_bytes(), b'"', b'\n');
        assert_eq!(lines.next(), Some("1,\"foo\n\"".as_bytes()));
        assert_eq!(lines.next(), Some("2,\"foo\n\"".as_bytes()));
        assert_eq!(lines.next(), None);

        let input2 = "1,'foo\n'\n2,'foo\n'\n";
        let mut lines2 = SplitLines::new(input2.as_bytes(), b'\'', b'\n');
        assert_eq!(lines2.next(), Some("1,'foo\n'".as_bytes()));
        assert_eq!(lines2.next(), Some("2,'foo\n'".as_bytes()));
        assert_eq!(lines2.next(), None);
    }
}
