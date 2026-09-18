use polars_arrow::array::{
    BINVIEW_ARROW_BUFFER_LEN_LIMIT, BINVIEW_MAX_ROW_BYTE_LEN, Utf8ViewArray, View,
};
use polars_arrow::datatypes::ArrowDataType;
use polars_buffer::Buffer;
use polars_core::prelude::DataType;
use polars_core::series::Series;
use polars_error::{PolarsResult, polars_bail, polars_ensure};
use polars_utils::pl_str::PlSmallStr;

const CR: u8 = b'\r';
const LF: u8 = b'\n';
const BUFFER_SPLIT_THRESHOLD: usize = 4096;

pub fn count_lines(full_bytes: &[u8]) -> usize {
    let mut n: usize = full_bytes.iter().map(|c| (*c == LF) as usize).sum();

    if let Some(c) = full_bytes.last()
        && *c != LF
    {
        n += 1;
    }

    n
}

pub fn split_lines_to_rows(bytes: Buffer<u8>) -> PolarsResult<Series> {
    split_lines_to_rows_impl(bytes, BINVIEW_MAX_ROW_BYTE_LEN)
}

fn split_lines_to_rows_impl(bytes: Buffer<u8>, max_row_size: usize) -> PolarsResult<Series> {
    if bytes.is_empty() {
        return Ok(Series::new_empty(PlSmallStr::EMPTY, &DataType::String));
    };

    if simdutf8::basic::from_utf8(&bytes).is_err() {
        polars_bail!(ComputeError: "invalid utf8")
    }

    let first_line_len = memchr::memchr(LF, &bytes).unwrap_or(bytes.len());
    let last_line_len = memchr::memrchr(LF, &bytes).map_or(bytes.len(), |i| bytes.len() - 1 - i);

    let n_lines_estimate = bytes
        .len()
        .div_ceil(first_line_len.min(last_line_len).max(1));

    let mut views: Vec<View> = Vec::with_capacity(n_lines_estimate);
    let mut data_buffers: Vec<Buffer<u8>> = Vec::new();
    let mut total_bytes_len: usize = 0;
    let mut total_buffer_len: usize = 0;
    let mut active_buffer: Option<(usize, usize)> = None;

    let bytes = if bytes.last() == Some(&LF) {
        let len = bytes.len();
        bytes.sliced(..len - 1)
    } else {
        bytes
    };

    let slice: &[u8] = &bytes;
    let mut line_start: usize = 0;

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

        let line_bytes = unsafe { slice.get_unchecked(start..end) };

        let view = if len <= View::MAX_INLINE_SIZE as usize {
            unsafe { View::new_inline_unchecked(line_bytes) }
        } else {
            // Note: `start > buffer_end`, there is always at least a line terminator in between.
            if let Some((buffer_start, buffer_end)) = active_buffer
                && (end - buffer_start > BINVIEW_ARROW_BUFFER_LEN_LIMIT
                    || start - buffer_end > BUFFER_SPLIT_THRESHOLD)
            {
                total_buffer_len += buffer_end - buffer_start;
                data_buffers.push(bytes.clone().sliced(buffer_start..buffer_end));
                active_buffer = None;
            }

            let buffer_start = active_buffer.map_or(start, |(buffer_start, _)| buffer_start);
            active_buffer = Some((buffer_start, end));

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
    use polars_buffer::Buffer;
    use polars_error::PolarsError;

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
}
