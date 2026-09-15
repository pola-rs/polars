use arrow::array::{BINVIEW_ARROW_BUFFER_LEN_LIMIT, BINVIEW_MAX_ROW_BYTE_LEN, Utf8ViewArray, View};
use arrow::datatypes::ArrowDataType;
use polars_buffer::Buffer;
use polars_core::prelude::DataType;
use polars_core::series::Series;
use polars_error::{PolarsResult, polars_bail, polars_ensure};
use polars_utils::pl_str::PlSmallStr;

const CR: u8 = b'\r';
const LF: u8 = b'\n';

pub fn count_lines(full_bytes: &[u8]) -> usize {
    let mut n: usize = full_bytes.iter().map(|c| (*c == LF) as usize).sum();

    if let Some(c) = full_bytes.last()
        && *c != LF
    {
        n += 1;
    }

    n
}

/// Splits `bytes` into a String column containing one row per line.
///
/// The returned `Series` references `bytes` directly - lines that are too long to be stored inline
/// are stored as views into `bytes` instead of being copied into a newly allocated buffer.
pub fn split_lines_to_rows(bytes: Buffer<u8>) -> PolarsResult<Series> {
    split_lines_to_rows_impl(bytes, BINVIEW_MAX_ROW_BYTE_LEN)
}

fn split_lines_to_rows_impl(bytes: Buffer<u8>, max_row_size: usize) -> PolarsResult<Series> {
    if bytes.is_empty() {
        return Ok(Series::new_empty(PlSmallStr::EMPTY, &DataType::String));
    };

    // Validate the full chunk in a single pass rather than per-line. This is equivalent, as line
    // terminators are ASCII and splitting valid UTF-8 on them yields valid UTF-8.
    if simdutf8::basic::from_utf8(&bytes).is_err() {
        polars_bail!(ComputeError: "invalid utf8")
    }

    let first_line_len = memchr::memchr(LF, &bytes).unwrap_or(bytes.len());
    let last_line_len = memchr::memrchr(LF, &bytes).map_or(bytes.len(), |i| bytes.len() - 1 - i);

    let n_lines_estimate = bytes
        .len()
        .div_ceil(first_line_len.min(last_line_len).max(1));

    let mut views: Vec<View> = Vec::with_capacity(n_lines_estimate);
    // Slices of `bytes` referenced by the non-inline views. A single slice covering all of them is
    // enough unless the chunk exceeds the maximum buffer length.
    let mut data_buffers: Vec<Buffer<u8>> = Vec::new();
    let mut total_bytes_len: usize = 0;
    let mut total_buffer_len: usize = 0;
    // Range of `bytes` covered by the data buffer that is currently being filled.
    let mut active_buffer: Option<(usize, usize)> = None;

    let bytes = if bytes.last() == Some(&LF) {
        let len = bytes.len();
        bytes.sliced(..len - 1)
    } else {
        bytes
    };

    let slice: &[u8] = &bytes;
    let mut line_start: usize = 0;

    // Iterate over the end offsets of the line terminators, with a trailing sentinel for the last
    // line (which is not terminated after the trailing newline was stripped above).
    for line_end in memchr::memchr_iter(LF, slice)
        .map(|i| i + 1)
        .chain(std::iter::once(slice.len() + 1))
    {
        let start = line_start;
        let mut end = line_end - 1;
        line_start = line_end;

        if end > start && slice[end - 1] == CR {
            end -= 1;
        }

        let len = end - start;

        polars_ensure!(
            len <= max_row_size,
            ComputeError:
            "line byte length {} exceeds max row byte length {}",
            len, max_row_size,
        );

        total_bytes_len += len;

        // SAFETY: `start <= end <= slice.len()`.
        let line_bytes = unsafe { slice.get_unchecked(start..end) };

        let view = if len <= View::MAX_INLINE_SIZE as usize {
            // SAFETY: Length checked above.
            unsafe { View::new_inline_unchecked(line_bytes) }
        } else {
            if let Some((buffer_start, buffer_end)) = active_buffer
                && end - buffer_start > BINVIEW_ARROW_BUFFER_LEN_LIMIT
            {
                total_buffer_len += buffer_end - buffer_start;
                data_buffers.push(bytes.clone().sliced(buffer_start..buffer_end));
                active_buffer = None;
            }

            let buffer_start = active_buffer.map_or(start, |(buffer_start, _)| buffer_start);
            active_buffer = Some((buffer_start, end));

            // SAFETY: Length checked above. The offset fits in a `u32` as the active buffer is
            // flushed before it can exceed `BINVIEW_ARROW_BUFFER_LEN_LIMIT`.
            unsafe {
                View::new_noninline_unchecked(
                    line_bytes,
                    data_buffers.len() as u32,
                    (start - buffer_start) as u32,
                )
            }
        };

        views.push(view);
    }

    if let Some((buffer_start, buffer_end)) = active_buffer {
        total_buffer_len += buffer_end - buffer_start;
        data_buffers.push(bytes.sliced(buffer_start..buffer_end));
    }

    // SAFETY: The views are constructed from, and point into, `data_buffers`. The data was
    // validated to be UTF-8 above.
    let arr = unsafe {
        Utf8ViewArray::new_unchecked(
            ArrowDataType::Utf8View,
            views.into(),
            data_buffers.into(),
            None,
            Some(total_bytes_len),
            total_buffer_len,
        )
    };

    Ok(unsafe {
        Series::_try_from_arrow_unchecked(
            PlSmallStr::EMPTY,
            vec![arr.boxed()],
            &ArrowDataType::Utf8View,
        )?
    })
}

#[cfg(test)]
mod tests {
    use arrow::array::BINVIEW_MAX_ROW_BYTE_LEN;
    use polars_buffer::Buffer;
    use polars_error::PolarsError;

    use super::{CR, LF};
    use crate::scan_lines::split_lines_to_rows_impl;

    #[test]
    fn test_split_lines_to_rows_impl() {
        let data: &'static [u8] = b"
AAAAABBBBBCCCCCDDDDD

EEEEEFFFFFGGGGGHHHHH

";

        let out = split_lines_to_rows_impl(Buffer::from_static(data), 20).unwrap();
        let out = out.str().unwrap();

        assert_eq!(
            out.iter().collect::<Vec<_>>().as_slice(),
            &[
                Some(""),
                Some("AAAAABBBBBCCCCCDDDDD"),
                Some(""),
                Some("EEEEEFFFFFGGGGGHHHHH"),
                Some(""),
            ]
        );

        let v: Vec<&[Buffer<u8>]> = out
            .downcast_iter()
            .map(|array| array.data_buffers().as_ref())
            .collect();

        // The data buffer is a slice of the input spanning the non-inline lines.
        assert_eq!(
            v.as_slice(),
            &[&[Buffer::from_static(
                b"AAAAABBBBBCCCCCDDDDD\n\nEEEEEFFFFFGGGGGHHHHH"
            )]]
        );

        let PolarsError::ComputeError(err_str) =
            split_lines_to_rows_impl(Buffer::from_static(data), 19).unwrap_err()
        else {
            unreachable!()
        };

        assert_eq!(
            &*err_str,
            "line byte length 20 exceeds max row byte length 19"
        );
    }

    #[test]
    fn test_split_lines_to_rows_impl_all_inline() {
        let data: Vec<u8> = [
            b"AAAABBBBCCCC\n".as_slice(),
            b"            \n".as_slice(),
            b"DDDDEEEEFFFF\n".as_slice(),
            b"            ".as_slice(),
        ]
        .concat();

        let out = split_lines_to_rows_impl(Buffer::from_vec(data), 12).unwrap();
        let out = out.str().unwrap();

        assert_eq!(
            out.iter().collect::<Vec<_>>().as_slice(),
            &[
                Some("AAAABBBBCCCC"),
                Some("            "),
                Some("DDDDEEEEFFFF"),
                Some("            "),
            ]
        );

        let v: Vec<&[Buffer<u8>]> = out
            .downcast_iter()
            .map(|array| array.data_buffers().as_ref())
            .collect();

        assert_eq!(v.as_slice(), &[&[][..]]);
    }

    #[test]
    fn test_split_lines_to_rows_impl_crlf() {
        let data: &'static [u8] = b"AAAAABBBBBCCCCCDDDDD\r\n\r\nshort\r\n";

        let out = split_lines_to_rows_impl(Buffer::from_static(data), 20).unwrap();
        let out = out.str().unwrap();

        assert_eq!(
            out.iter().collect::<Vec<_>>().as_slice(),
            &[Some("AAAAABBBBBCCCCCDDDDD"), Some(""), Some("short")]
        );
    }

    #[test]
    fn test_split_lines_to_rows_impl_invalid_utf8() {
        let data: &'static [u8] = b"abc\n\xff\xfe\n";

        let PolarsError::ComputeError(err_str) =
            split_lines_to_rows_impl(Buffer::from_static(data), 20).unwrap_err()
        else {
            unreachable!()
        };

        assert_eq!(&*err_str, "invalid utf8");
    }

    /// Compares against a straightforward reference implementation over randomized inputs.
    #[test]
    fn test_split_lines_to_rows_impl_random() {
        // Simple xorshift, so that a failure is reproducible.
        let mut state: u64 = 0x9E3779B97F4A7C15;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        for _ in 0..200 {
            let mut data: Vec<u8> = Vec::new();

            for _ in 0..(next() % 32) {
                // Mix of inline (<= 12 bytes) and non-inline lines.
                let line_len = (next() % 40) as usize;

                for _ in 0..line_len {
                    data.push(b'a' + (next() % 26) as u8);
                }

                if next() % 4 == 0 {
                    data.push(CR);
                }

                data.push(LF);
            }

            // Randomly leave the last line unterminated.
            if next() % 2 == 0 {
                data.pop();
            }

            let expected: Vec<&str> = {
                let trimmed = if data.last() == Some(&LF) {
                    &data[..data.len() - 1]
                } else {
                    &data[..]
                };

                if data.is_empty() {
                    vec![]
                } else {
                    trimmed
                        .split(|c| *c == LF)
                        .map(|line| {
                            let line = if line.last() == Some(&CR) {
                                &line[..line.len() - 1]
                            } else {
                                line
                            };
                            std::str::from_utf8(line).unwrap()
                        })
                        .collect()
                }
            };

            let out =
                split_lines_to_rows_impl(Buffer::from_vec(data.clone()), BINVIEW_MAX_ROW_BYTE_LEN)
                    .unwrap();
            let out = out.str().unwrap();

            assert_eq!(
                out.iter().map(|x| x.unwrap()).collect::<Vec<_>>(),
                expected,
                "input: {:?}",
                std::str::from_utf8(&data).unwrap()
            );
        }
    }

    /// The zero-copy views must remain valid after the input `Buffer` handle is dropped.
    #[test]
    fn test_split_lines_to_rows_impl_owns_buffer() {
        let data: Vec<u8> = b"AAAAABBBBBCCCCCDDDDD\nEEEEEFFFFFGGGGGHHHHH\n".to_vec();
        let buffer = Buffer::from_vec(data);

        let out = split_lines_to_rows_impl(buffer.clone(), 20).unwrap();
        drop(buffer);

        assert_eq!(
            out.str().unwrap().iter().collect::<Vec<_>>().as_slice(),
            &[Some("AAAAABBBBBCCCCCDDDDD"), Some("EEEEEFFFFFGGGGGHHHHH")]
        );
    }
}
