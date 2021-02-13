use super::buffer::*;
use num::traits::Pow;
use polars_core::prelude::*;

/// Skip the utf-8 Byte Order Mark.
/// credits to csv-core
pub(crate) fn skip_bom(input: &[u8]) -> &[u8] {
    if input.len() >= 3 && &input[0..3] == b"\xef\xbb\xbf" {
        &input[..3]
    } else {
        input
    }
}

pub(crate) fn next_line_position(input: &[u8]) -> Option<usize> {
    let pos = input.iter().position(|b| *b == b'\n')?;
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

pub(crate) fn is_line_ending(b: u8) -> bool {
    b == b'\n' || b == b'\r'
}

pub(crate) fn is_whitespace(b: u8) -> bool {
    b == b' ' || b == b'\t'
}

fn skip_condition<F>(input: &[u8], f: F) -> (&[u8], usize)
where
    F: Fn(u8) -> bool,
{
    if input.is_empty() {
        return (input, 0);
    }
    let mut read = 0;
    loop {
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
    let mut pos = next_line_position(input).expect("no lines in the file");
    if input[pos + 1] == b'\n' {
        pos += 1;
    }
    (&input[pos + 1..], pos + 1)
}

/// Remove whitespace and line endings from the start of file.
#[inline]
pub(crate) fn skip_whitespace(input: &[u8]) -> (&[u8], usize) {
    skip_condition(input, |b| is_whitespace(b) || is_line_ending(b))
}

pub(crate) fn skip_line_ending(input: &[u8]) -> (&[u8], usize) {
    skip_condition(input, is_line_ending)
}

/// Get the mean and standard deviation of length of lines in bytes
pub(crate) fn get_line_stats(mut bytes: &[u8], n_lines: usize) -> Option<(f32, f32)> {
    let mut n_read = 0;
    let mut lengths = Vec::with_capacity(n_lines);

    for _ in 0..n_lines {
        if n_read >= bytes.len() {
            return None;
        }
        bytes = &bytes[n_read..];
        match bytes.iter().position(|&b| b == b'\n') {
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
/// This exists solely because we cannot split the lines naively as
///
/// ```text
///    lines.split(b',').for_each(do_stuff)
/// ```
///
/// This will fail when strings have contained delimiters.
/// For instance: "Street, City, Country" is a valid string field, that contains multiple delimiters.
struct SplitFields<'a> {
    v: &'a [u8],
    delimiter: u8,
    // escaped string field ",
    str_delimiter: [u8; 2],
    finished: bool,
}

impl<'a> SplitFields<'a> {
    fn new(slice: &'a [u8], delimiter: u8) -> Self {
        Self {
            v: slice,
            delimiter,
            str_delimiter: [b'"', delimiter],
            finished: false,
        }
    }

    fn finish(&mut self) -> Option<&'a [u8]> {
        if self.finished {
            None
        } else {
            self.finished = true;
            Some(self.v)
        }
    }
}

impl<'a> Iterator for SplitFields<'a> {
    type Item = &'a [u8];

    #[inline]
    fn next(&mut self) -> Option<&'a [u8]> {
        if self.finished {
            return None;
        }
        // There can be strings with delimiters:
        // "Street, City",
        let pos = if !self.v.is_empty() && self.v[0] == b'"' {
            // we offset 1 because "," is a valid field and we don't want to match position 0.
            match self.v[1..].windows(2).position(|x| x == self.str_delimiter) {
                None => return self.finish(),
                Some(idx) => idx + 2,
            }
        } else {
            match self.v.iter().position(|x| *x == self.delimiter) {
                None => return self.finish(),
                Some(idx) => idx,
            }
        };

        let ret = Some(&self.v[..pos]);
        self.v = &self.v[pos + 1..];
        ret
    }
}

/// Parse CSV.
///
/// # Arguments
/// * `bytes` - input to parse
/// * `offset` - offset in bytes in total input. This is 0 if single threaded. If multithreaded every
///              thread has a different offset.
/// * `projection` - Indices of the columns to project.
/// * `buffers` - Parsed output will be written to these buffers. Except for UTF8 data. The offsets of the
///               fields are written to the buffers. The UTF8 data will be parsed later.
pub(crate) fn parse_lines(
    bytes: &[u8],
    offset: usize,
    delimiter: u8,
    projection: &[usize],
    buffers: &mut [Buffer],
    ignore_parser_errors: bool,
) -> Result<()> {
    // This variable will store the number of bytes we read. It is important to do this bookkeeping
    // to be able to correctly parse the strings later.
    let mut read = offset;

    // We split the lines by the new line characters in the outer loop.
    // in the inner loop we deal with the fields/columns.
    // Any primitive type is directly parsed and stored in a Vec buffer.
    // String types are not parsed. We store strings the starting index in the bytes array and store
    // the length of the string field. We also store the total length of processed string fields per column.
    // Later we use that meta information to exactly allocate the required buffers and parse the strings.
    for mut line in bytes.split(|b| *b == b'\n') {
        let len = line.len();

        // two adjacent '\n\n' will lead to an empty line.
        if len == 0 {
            read += 1;
            continue;
        }
        // including the '\n' character
        let line_length = len + 1;

        let trailing_byte = line[len - 1];
        if trailing_byte == b'\r' {
            line = &line[..len - 1];
        }

        // read at start of the line
        let read_sol = read;
        // // +1 is the split character
        // read += 1;

        // Every line we only need to parse the columns that are projected.
        // Therefore we check if the idx of the field is in our projected columns.
        // If it is not, we skip the field.
        let mut projection_iter = projection.iter().copied();
        let mut next_projected = projection_iter
            .next()
            .expect("at least one column should be projected");
        let mut processed_fields = 0;

        let iter = SplitFields::new(line, delimiter);

        for (idx, field) in iter.enumerate() {
            if idx == next_projected {
                debug_assert!(processed_fields < buffers.len());
                let buf = unsafe {
                    // SAFETY: processed fields index can never exceed the projection indices.
                    buffers.get_unchecked_mut(processed_fields)
                };
                // let buf = &mut buffers[processed_fields];
                buf.add(field, ignore_parser_errors, read).map_err(|e| {
                    PolarsError::Other(
                        format!(
                            "{:?} on thread line {}; on input: {}",
                            e,
                            idx,
                            String::from_utf8_lossy(field)
                        )
                        .into(),
                    )
                })?;

                processed_fields += 1;

                // if we have all projected columns we are done with this line
                match projection_iter.next() {
                    Some(p) => next_projected = p,
                    None => {
                        break;
                    }
                }
            }
            // +1 is the split character that is consumed by the iterator.
            read += field.len() + 1;
        }
        // this way we also include the trailing '\n' or '\r\n' in the bytes read
        // and any skipped fields.
        read = read_sol + line_length;
    }
    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;
    use std::io::Read;

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
    fn test_parse_lines() {
        let path = "../../examples/aggregate_multiple_files_in_chunks/datasets/foods1.csv";
        let mut file = std::fs::File::open(path).unwrap();
        let mut input = vec![];
        file.read_to_end(&mut input).unwrap();
        let bytes = skip_header(skip_whitespace(&input).0).0;

        // after a skip header, we should not have a new line char.
        assert_ne!(bytes[0], b'\n');
        dbg!(std::str::from_utf8(bytes).unwrap());

        let mut buffers = vec![
            Buffer::Utf8(vec![], 0),
            Buffer::UInt64(vec![]),
            Buffer::Float64(vec![]),
            Buffer::Float64(vec![]),
        ];

        macro_rules! call_buff_method {
            ($buf: expr, $method:ident) => {{
                use Buffer::*;
                match $buf {
                    Boolean(a) => a.$method(),
                    Int32(a) => a.$method(),
                    Int64(a) => a.$method(),
                    UInt64(a) => a.$method(),
                    UInt32(a) => a.$method(),
                    Float32(a) => a.$method(),
                    Float64(a) => a.$method(),
                    Utf8(a, _) => a.$method(),
                }
            }};
        }

        let projection = &[0, 1, 2, 3];

        parse_lines(&bytes, 0, b',', projection, &mut buffers, false).unwrap();
        // check if all buffers are correctly filled.
        for buf in &buffers {
            let len = call_buff_method!(buf, len);
            assert_eq!(len, 27);
        }

        dbg!(&buffers[0]);
        // check if we can reconstruct the correct strings from the accumulated offsets.
        if let Buffer::Utf8(buf, len) = &buffers[0] {
            let v = buf
                .iter()
                .map(|utf8_field| {
                    let sub_slice = utf8_field.get_long_subslice(bytes);
                    std::str::from_utf8(&sub_slice[..sub_slice.len() - 1]).unwrap()
                })
                .collect::<Vec<_>>();

            let total_len: usize = v.iter().map(|s| s.len()).sum();
            assert_eq!(total_len, *len);

            assert_eq!(&v[0], &"vegetables");
            assert_eq!(&v[v.len() - 1], &"fruit");
            dbg!(v);
        }
    }
}
