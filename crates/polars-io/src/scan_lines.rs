use arrow::array::BinaryViewArrayGenericBuilder;
use arrow::array::builder::StaticArrayBuilder;
use arrow::datatypes::ArrowDataType;
use polars_core::prelude::DataType;
use polars_core::series::Series;
use polars_error::{PolarsResult, polars_ensure};
use polars_utils::pl_str::PlSmallStr;

type BinviewArrayBuilder = BinaryViewArrayGenericBuilder<[u8]>;

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

pub fn split_lines_to_rows(bytes: &[u8]) -> PolarsResult<Series> {
    split_lines_to_rows_impl(bytes, BinviewArrayBuilder::MAX_ROW_BYTE_LEN)
}

fn split_lines_to_rows_impl(bytes: &[u8], max_buffer_size: usize) -> PolarsResult<Series> {
    if bytes.is_empty() {
        return Ok(Series::new_empty(PlSmallStr::EMPTY, &DataType::String));
    };

    let first_line_len = bytes.split(|c| *c == LF).next().unwrap().len();
    let last_line_len = bytes.rsplit(|c| *c == LF).next().unwrap().len();

    let n_lines_estimate = bytes
        .len()
        .div_ceil(first_line_len.min(last_line_len).max(1));

    let mut builder: BinviewArrayBuilder = BinviewArrayBuilder::new(ArrowDataType::BinaryView);
    builder.reserve(n_lines_estimate);

    let bytes = if bytes.last() == Some(&LF) {
        &bytes[..bytes.len() - 1]
    } else {
        bytes
    };

    for line_bytes in bytes.split(|c| *c == LF) {
        let line_bytes = if line_bytes.last() == Some(&CR) {
            &line_bytes[..line_bytes.len() - 1]
        } else {
            line_bytes
        };

        polars_ensure!(
            line_bytes.len() <= max_buffer_size,
            ComputeError:
            "line byte length {} exceeds max row byte length {}",
            line_bytes.len(), max_buffer_size,
        );

        builder.push_value_ignore_validity(line_bytes);
    }

    let arr = builder.freeze();

    // Performs UTF-8 validation.
    let arr = arr.to_utf8view()?;

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

        let out = split_lines_to_rows_impl(data, 20).unwrap();
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
                b"AAAAABBBBBCCCCCDDDDDEEEEEFFFFFGGGGGHHHHH"
            )]]
        );

        let PolarsError::ComputeError(err_str) = split_lines_to_rows_impl(data, 19).unwrap_err()
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

        let out = split_lines_to_rows_impl(&data, 12).unwrap();
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
