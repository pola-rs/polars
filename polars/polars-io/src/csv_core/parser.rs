use super::buffer::*;
use num::traits::Pow;
use polars_core::prelude::*;

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
pub(crate) fn next_line_position_naive(input: &[u8]) -> Option<usize> {
    let pos = input.iter().position(|b| *b == b'\n')? + 1;
    if input.len() - pos == 0 {
        return None;
    }
    input.get(pos + 1).and_then(|&b| {
        Option::from({
            if b == b'\r' {
                pos + 1
            } else {
                pos
            }
        })
    })
}

/// Find the nearest next line position that is not embedded in a String field.
pub(crate) fn next_line_position(
    mut input: &[u8],
    expected_fields: usize,
    delimiter: u8,
    quote_char: Option<u8>,
) -> Option<usize> {
    let mut total_pos = 0;
    if input.is_empty() {
        return None;
    }
    loop {
        let pos = input.iter().position(|b| *b == b'\n')? + 1;
        if input.len() - pos == 0 {
            return None;
        }
        let line = SplitLines::new(&input[pos..], b'\n').next();
        if let Some(line) = line {
            if SplitFields::new(line, delimiter, quote_char)
                .into_iter()
                .count()
                == expected_fields
            {
                return input.get(pos + 1).and_then(|&b| {
                    Option::from({
                        if b == b'\r' {
                            total_pos + pos + 1
                        } else {
                            total_pos + pos
                        }
                    })
                });
            } else {
                input = &input[pos + 1..];
                total_pos += pos + 1;
            }
        } else {
            return None;
        }
    }
}

pub(crate) fn is_line_ending(b: u8) -> bool {
    b == b'\n' || b == b'\r'
}

pub(crate) fn is_whitespace(b: u8) -> bool {
    b == b' ' || b == b'\t'
}

#[inline]
fn skip_condition<F>(input: &[u8], f: F) -> (&[u8], usize)
where
    F: Fn(u8) -> bool,
{
    if input.is_empty() {
        return (input, 0);
    }
    let mut read = 0;
    let len = input.len();
    while read < len {
        let b = input[read];
        if !f(b) {
            break;
        }
        read += 1;
    }
    (&input[read..], read)
}

/// Makes sure that the bytes stream starts with
///     'field_1,field_2'
/// and not with
///     '\nfield_1,field_1'
pub(crate) fn skip_header(input: &[u8]) -> (&[u8], usize) {
    let mut pos = next_line_position_naive(input).expect("no lines in the file");
    if input[pos] == b'\n' {
        pos += 1;
    }
    (&input[pos..], pos)
}

/// Remove whitespace and line endings from the start of file.
#[inline]
pub(crate) fn skip_whitespace(input: &[u8]) -> (&[u8], usize) {
    skip_condition(input, |b| is_whitespace(b) || is_line_ending(b))
}

#[inline]
/// Can be used to skip whitespace, but exclude the delimiter
pub(crate) fn skip_whitespace_exclude(input: &[u8], exclude: u8) -> (&[u8], usize) {
    skip_condition(input, |b| {
        b != exclude && (is_whitespace(b) || is_line_ending(b))
    })
}

/// Local version of slice::starts_with (as it won't inline)
#[inline]
fn starts_with(bytes: &[u8], needle: u8) -> bool {
    !bytes.is_empty() && bytes[0] == needle
}

/// Slice `"100"` to `100`, if slice starts with `"` it does not check that it ends with `"`, but
/// assumes this. Be aware of this.
#[inline]
pub(crate) fn drop_quotes(input: &[u8]) -> &[u8] {
    if starts_with(input, b'"') {
        &input[1..input.len() - 1]
    } else {
        input
    }
}

#[inline]
pub(crate) fn skip_line_ending(input: &[u8]) -> (&[u8], usize) {
    skip_condition(input, is_line_ending)
}

/// Get the mean and standard deviation of length of lines in bytes
pub(crate) fn get_line_stats(bytes: &[u8], n_lines: usize) -> Option<(f32, f32)> {
    let mut n_read = 0;
    let mut lengths = Vec::with_capacity(n_lines);
    let file_len = bytes.len();
    let mut bytes_trunc;

    for _ in 0..n_lines {
        if n_read >= file_len {
            return None;
        }
        bytes_trunc = &bytes[n_read..];
        match bytes_trunc.iter().position(|&b| b == b'\n') {
            Some(position) => {
                n_read += position + 1;
                lengths.push(position + 1);
            }
            None => {
                return None;
            }
        }
    }
    let mean = (n_read as f32) / (n_lines as f32);
    let mut std = 0.0;
    for &len in lengths.iter().take(n_lines) {
        std += (len as f32 - mean).pow(2.0)
    }
    std = (std / n_lines as f32).pow(0.5);
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
    end_line_char: u8,
}

impl<'a> SplitLines<'a> {
    pub(crate) fn new(slice: &'a [u8], end_line_char: u8) -> Self {
        Self {
            v: slice,
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
                    if c == b'"' {
                        // toggle between string field enclosure
                        //      if we encounter a starting '"' -> in_field = true;
                        //      if we encounter a closing '"' -> in_field = false;
                        in_field = !in_field;
                    }
                    // if we are not in a string and we encounter '\n' we can stop at this position.
                    if c == self.end_line_char && !in_field {
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
        // return line up to this position
        let ret = Some(&self.v[..(pos - 1) as usize]);
        // skip the '\n' token and update slice.
        self.v = &self.v[pos as usize..];
        ret
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
}

impl<'a> SplitFields<'a> {
    pub(crate) fn new(slice: &'a [u8], delimiter: u8, quote_char: Option<u8>) -> Self {
        Self {
            v: slice,
            delimiter,
            finished: false,
            quote_char: quote_char.unwrap_or(b'"'),
            quoting: quote_char.is_some(),
        }
    }

    fn finish_eol(&mut self, need_escaping: bool, idx: usize) -> Option<(&'a [u8], bool)> {
        self.finished = true;
        Some((&self.v[..idx], need_escaping))
    }

    fn finish(&mut self, need_escaping: bool) -> Option<(&'a [u8], bool)> {
        self.finished = true;
        Some((self.v, need_escaping))
    }

    fn eof_oel(&self, current_ch: u8) -> bool {
        current_ch == self.delimiter || current_ch == b'\n'
    }
}

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

        let pos = if self.quoting && self.v[0] == self.quote_char {
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
                    if c == b'\n' {
                        return self.finish_eol(needs_escaping, current_idx as usize);
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
            match memchr::memchr2(self.delimiter, b'\n', self.v) {
                None => return self.finish(needs_escaping),
                Some(idx) => match self.v[idx] {
                    b'\n' => return self.finish_eol(needs_escaping, idx),
                    _ => idx,
                },
            }
        };

        let ret = Some((&self.v[..pos], needs_escaping));
        self.v = &self.v[pos + 1..];
        ret
    }
}

fn skip_this_line(bytes: &[u8], quote: Option<u8>, offset: usize) -> (&[u8], usize) {
    let pos = match quote {
        Some(quote) => find_quoted(bytes, quote, b'\n'),
        None => bytes.iter().position(|x| *x == b'\n'),
    };
    match pos {
        None => (&[], bytes.len() + offset),
        Some(pos) => (&bytes[pos + 1..], pos + 1 + offset),
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
pub(crate) fn parse_lines(
    mut bytes: &[u8],
    offset: usize,
    delimiter: u8,
    comment_char: Option<u8>,
    quote_char: Option<u8>,
    null_values: Option<&Vec<String>>,
    projection: &[usize],
    buffers: &mut [Buffer],
    ignore_parser_errors: bool,
    n_lines: usize,
) -> Result<usize> {
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
        let (b, _) = skip_whitespace_exclude(bytes, delimiter);
        bytes = b;
        if bytes.is_empty() {
            return Ok(original_bytes_len);
        }

        // deal with comments
        if let Some(c) = comment_char {
            // line is a comment -> skip
            if bytes[0] == c {
                let (bytes_rem, _) = skip_this_line(bytes, quote_char, 0);
                bytes = bytes_rem;
                continue;
            }
        }

        // Every line we only need to parse the columns that are projected.
        // Therefore we check if the idx of the field is in our projected columns.
        // If it is not, we skip the field.
        let mut projection_iter = projection.iter().copied();
        let mut next_projected = projection_iter
            .next()
            .expect("at least one column should be projected");
        let mut processed_fields = 0;

        let mut iter = SplitFields::new(bytes, delimiter, quote_char);
        let mut idx = 0u32;
        loop {
            let mut read_sol = 0;

            let track_bytes = |bytes: &mut &[u8], read_sol: usize| {
                // benchmarking showed it is 6% faster to take the min of these two
                // not needed for correctness.
                *bytes = &bytes[std::cmp::min(read_sol, bytes.len())..];
            };

            match iter.next() {
                None => break,
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
                        if let Some(null_values) = &null_values {
                            if let Some(null_value) = null_values.get(processed_fields) {
                                if field == null_value.as_bytes() {
                                    add_null = true;
                                }
                            }
                        }
                        if add_null {
                            buf.add_null()
                        } else {
                            buf.add(field, ignore_parser_errors, needs_escaping)
                                .map_err(|_| {
                                    let bytes_offset = offset + field.as_ptr() as usize - start;
                                    let unparsable = String::from_utf8_lossy(field);
                                    PolarsError::ComputeError(
                                        format!(
                                            r#"Could not parse {} as dtype {:?} at column {}.
                                            The total offset in the file is {} bytes.

                                            Consider running the parser `with_ignore_parser_errors=true`
                                            or consider adding {} to the `null_values` list."#,
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

                        let a = projection_iter.next();

                        // if we have all projected columns we are done with this line
                        match a {
                            Some(p) => {
                                track_bytes(&mut bytes, read_sol);
                                next_projected = p
                            }
                            None => {
                                if let Some(b'\n') = bytes.get(0) {
                                    bytes = &bytes[read_sol..];
                                } else {
                                    let (bytes_rem, _) = skip_this_line(bytes, quote_char, 0);
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

            buf.add_null();
            processed_fields += 1;
        }

        line_count += 1;
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_skip() {
        let input = b"    hello";
        assert_eq!(skip_whitespace(input).0, b"hello");
        let input = b"\n        hello";
        assert_eq!(skip_whitespace(input).0, b"hello");
        let input = b"\t\n\r
        hello";
        assert_eq!(skip_whitespace(input).0, b"hello");
    }

    #[test]
    fn test_splitfields() {
        let input = "\"foo\",\"bar\"";
        let mut fields = SplitFields::new(input.as_bytes(), b',', Some(b'"'));

        assert_eq!(fields.next(), Some(("\"foo\"".as_bytes(), true)));
        assert_eq!(fields.next(), Some(("\"bar\"".as_bytes(), true)));
        assert_eq!(fields.next(), None);

        let input2 = "\"foo\n bar\";\"baz\";12345";
        let mut fields2 = SplitFields::new(input2.as_bytes(), b';', Some(b'"'));

        assert_eq!(fields2.next(), Some(("\"foo\n bar\"".as_bytes(), true)));
        assert_eq!(fields2.next(), Some(("\"baz\"".as_bytes(), true)));
        assert_eq!(fields2.next(), Some(("12345".as_bytes(), false)));
        assert_eq!(fields2.next(), None);
    }
}
