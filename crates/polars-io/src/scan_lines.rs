use arrow::array::BinaryViewArrayGenericBuilder;
use arrow::datatypes::ArrowDataType;
use polars_core::prelude::DataType;
use polars_core::series::Series;
use polars_error::{PolarsResult, polars_bail};
use polars_utils::pl_str::PlSmallStr;

const EOL_CHAR: u8 = b'\n';

pub fn count_lines(full_bytes: &[u8]) -> usize {
    let mut n: usize = full_bytes.iter().map(|c| (*c == EOL_CHAR) as usize).sum();

    if let Some(c) = full_bytes.last()
        && *c != EOL_CHAR
    {
        n += 1;
    }

    n
}

pub fn split_lines_to_rows(bytes: &[u8]) -> PolarsResult<Series> {
    split_lines_to_rows_impl(bytes, u32::MAX as usize)
}

fn split_lines_to_rows_impl(bytes: &[u8], max_buffer_size: usize) -> PolarsResult<Series> {
    if bytes.is_empty() {
        return Ok(Series::new_empty(PlSmallStr::EMPTY, &DataType::String));
    };

    let first_line_len = bytes.split(|c| *c == EOL_CHAR).next().unwrap().len();
    let last_line_len = bytes.rsplit(|c| *c == EOL_CHAR).next().unwrap().len();

    let n_lines_estimate = bytes
        .len()
        .div_ceil(first_line_len.min(last_line_len).max(1));

    use arrow::array::builder::StaticArrayBuilder;

    let mut builder: BinaryViewArrayGenericBuilder<[u8]> =
        BinaryViewArrayGenericBuilder::new(ArrowDataType::BinaryView);
    builder.reserve(n_lines_estimate);

    for line_bytes in bytes
        .strip_suffix(&[EOL_CHAR])
        .unwrap_or(bytes)
        .split(|c| *c == EOL_CHAR)
    {
        if line_bytes.len() > max_buffer_size {
            polars_bail!(
                ComputeError:
                "line byte length {} exceeds max buffer size {}",
                line_bytes.len(), max_buffer_size,
            )
        }

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
    use arrow::buffer::Buffer;
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

        assert_eq!(&*err_str, "line byte length 20 exceeds max buffer size 19");
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
