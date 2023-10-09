use memchr::memchr2_iter;
use num_traits::Pow;
use polars_core::prelude::*;

use super::buffer::*;
use crate::csv::read::NullValuesCompiled;
use crate::csv::splitfields::SplitFields;

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
    let mut lines_checked = 0u16;
    loop {
        if rejected_line_groups >= 3 {
            return None;
        }
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
        let mut lines = SplitLines::new(new_input, quote_char.unwrap_or(b'"'), eol_char);
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
pub(crate) fn skip_header(input: &[u8], quote: Option<u8>, eol_char: u8) -> &[u8] {
    skip_this_line(input, quote, eol_char)
}

/// Remove whitespace from the start of buffer.
#[inline]
pub(crate) fn skip_whitespace(input: &[u8]) -> &[u8] {
    skip_condition(input, is_whitespace)
}

#[inline]
/// Can be used to skip whitespace, but exclude the separator
pub(crate) fn skip_whitespace_exclude(input: &[u8], exclude: u8) -> &[u8] {
    skip_condition(input, |b| b != exclude && (is_whitespace(b)))
}

#[inline]
/// Can be used to skip whitespace, but exclude the separator
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
            Some(expected_fields),
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
                },
                None => {
                    // no new line found we are done
                    // the rest will be done by last line specific code.
                    return None;
                },
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
    separator: u8,
    comment_char: Option<u8>,
    quote_char: Option<u8>,
    eol_char: u8,
    missing_is_null: bool,
    ignore_errors: bool,
    mut truncate_ragged_lines: bool,
    null_values: Option<&NullValuesCompiled>,
    projection: &[usize],
    buffers: &mut [Buffer<'a>],
    n_lines: usize,
    // length of original schema
    schema_len: usize,
    schema: &Schema,
) -> PolarsResult<usize> {
    assert!(
        !projection.is_empty(),
        "at least one column should be projected"
    );
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

        // only when we have one column \n should not be skipped
        // other widths should have commas.
        bytes = if schema_len > 1 {
            skip_whitespace_line_ending_exclude(bytes, separator, eol_char)
        } else {
            skip_whitespace_exclude(bytes, separator)
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

        let mut iter = SplitFields::new(bytes, separator, quote_char, eol_char);
        let mut idx = 0u32;
        let mut read_sol = 0;
        loop {
            match iter.next() {
                // end of line
                None => {
                    bytes = &bytes[std::cmp::min(read_sol, bytes.len())..];
                    break;
                },
                Some((mut field, needs_escaping)) => {
                    let field_len = field.len();

                    // +1 is the split character that is consumed by the iterator.
                    read_sol += field_len + 1;

                    if idx == next_projected as u32 {
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
                                    let column_name = schema.get_at_index(idx as usize).unwrap().0;
                                    polars_err!(
                                        ComputeError:
                                        "Could not parse `{}` as dtype `{}` at column '{}' (column number {}).\n\
                                        The current offset in the file is {} bytes.\n\
                                        \n\
                                        You might want to try:\n\
                                        - increasing `infer_schema_length` (e.g. `infer_schema_length=10000`),\n\
                                        - specifying correct dtype with the `dtypes` argument\n\
                                        - setting `ignore_errors` to `True`,\n\
                                        - adding `{}` to the `null_values` list.",
                                        &unparsable,
                                        buf.dtype(),
                                        column_name,
                                        idx + 1,
                                        bytes_offset,
                                        &unparsable,
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
                                    if !truncate_ragged_lines && read_sol < bytes.len() {
                                        polars_bail!(ComputeError: r#"found more fields than defined in 'Schema'

Consider setting 'truncate_ragged_lines={}'."#, polars_error::constants::TRUE)
                                    }
                                    let bytes_rem = skip_this_line(
                                        &bytes[read_sol - 1..],
                                        quote_char,
                                        eol_char,
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
