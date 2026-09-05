use std::borrow::Cow;

use polars_array::builder::StaticArrayBuilder;
use polars_array::bitmap::PlBitmap;
use polars_array::{PlArray as _, PlUtf8ViewArrayBuilder, StaticArray as _};
use polars_core::prelude::{Column, DataType, IntoColumn, StringChunked};
use polars_core::scalar::Scalar;
use polars_error::{PolarsContext, PolarsResult};
use polars_utils::broadcast::broadcast_len;
use polars_utils::pl_str::PlSmallStr;

#[inline(always)]
fn opt_str_to_string(s: Option<&str>) -> &str {
    s.unwrap_or("null")
}

pub fn str_format(cs: &mut [Column], format: &str, insertions: &[usize]) -> PolarsResult<Column> {
    assert_eq!(cs.len(), insertions.len());
    assert!(!cs.is_empty()); // Checked at IR construction

    let output_name = cs[0].name().clone();
    let output_length = broadcast_len(cs.iter()).context("str.format")?;

    let mut validity = None;
    let mut num_scalar_inputs = 0;
    for c in cs.iter_mut() {
        if let Some(c_validity) = c.rechunk_validity() {
            // Column with only nulls means output is only nulls.
            if c.null_count() == c.len() {
                return Ok(Column::full_null(
                    output_name,
                    output_length,
                    &DataType::String,
                ));
            }

            match &mut validity {
                v @ None => *v = Some(c_validity),
                Some(v) => *v = arrow::bitmap::and(v, &c_validity),
            }
        }

        *c = c.cast(&DataType::String)?;
        num_scalar_inputs += usize::from(c.len() == 1);
    }

    let mut format = Cow::Borrowed(format);
    let mut insertions = Cow::Borrowed(insertions);

    // Fill in any constants into the format string.
    if num_scalar_inputs > 0 {
        let mut filled_format = String::new();
        filled_format.push_str(&format[..*insertions.first().unwrap()]);
        insertions = Cow::Owned(
            cs.iter()
                .enumerate()
                .filter_map(|(i, c)| {
                    let v = if c.len() == 1 {
                        filled_format.push_str(opt_str_to_string(c.str().unwrap().get(0)));
                        None
                    } else {
                        Some(filled_format.len())
                    };

                    let s = if i == cs.len() - 1 {
                        &format[insertions[i]..]
                    } else {
                        &format[insertions[i]..insertions[i + 1]]
                    };
                    filled_format.push_str(s);

                    v
                })
                .collect(),
        );
        format = filled_format.into();
    }

    let format = format.as_ref();
    let insertions = insertions.as_ref();

    // If the format string is constant.
    if num_scalar_inputs == cs.len() {
        let sc = Scalar::from(PlSmallStr::from_str(format));
        return Ok(Column::new_scalar(output_name, sc, output_length));
    }

    let mut builder = PlUtf8ViewArrayBuilder::new();
    builder.reserve(output_length);

    let mut arrays = cs
        .iter()
        .filter(|c| c.len() != 1)
        .map(|c| {
            let ca = c.str().unwrap();
            let mut iter = ca.downcast_iter();
            let arr = iter.next().unwrap();
            (iter, arr, 0)
        })
        .collect::<Vec<_>>();

    // @Performance. There is some smarter stuff that can be done with views and stuff. Don't think
    // it is worth the complexity.

    // Amortize the format string allocation.
    let mut s = String::new();
    for i in 0..output_length {
        if validity
            .as_ref()
            .is_some_and(|v| !unsafe { v.get_bit_unchecked(i) })
        {
            // The value of a null element is undetermined, so anything at all does.
            builder.push_value("");

            for (iter, arr, elem_idx) in arrays.iter_mut() {
                *elem_idx += 1;
                if i + 1 != output_length && *elem_idx == arr.len() {
                    *arr = iter.next().unwrap();
                    *elem_idx = 0;
                }
            }

            continue;
        }

        s.clear();
        s.push_str(&format[..insertions[0]]);

        for (j, (iter, arr, elem_idx)) in arrays.iter_mut().enumerate() {
            s.push_str(opt_str_to_string(arr.get(*elem_idx)));
            let start = insertions[j];
            let end = insertions.get(j + 1).copied().unwrap_or(format.len());
            s.push_str(&format[start..end]);

            *elem_idx += 1;
            if i + 1 != output_length && *elem_idx == arr.len() {
                *arr = iter.next().unwrap();
                *elem_idx = 0;
            }
        }

        builder.push_value(&s);
    }

    // TODO(polars-array-scalar): the rows are formatted one at a time, so a repeated row is
    // written out once per element rather than formatted once and broadcast.
    let array = builder
        .freeze()
        .with_validity(validity.map(PlBitmap::from_bitmap))
        .into_boxed();
    Ok(unsafe { StringChunked::from_chunks(output_name, vec![array]) }.into_column())
}
