//! Decoding of `DELTA_LENGTH_BYTE_ARRAY` and `DELTA_BYTE_ARRAY` pages.
//!
//! The values of a page are decoded into views in bulk first, from
//! which the rows are taken (as for a plain-encoded primitive column).

use polars_arrow::array::View;
use polars_arrow::bitmap::Bitmap;
use polars_arrow::types::Bytes16Alignment4;
use polars_compute::filter::filter_boolean_kernel;
use polars_utils::vec::with_cast_mut_vec;

use super::{
    DecodedState, inlined_are_valid_utf8, invalid_input_err, invalid_utf8_err, new_inline_view,
};
use crate::parquet::encoding::{delta_byte_array, delta_length_byte_array};
use crate::parquet::error::ParquetResult;
use crate::read::Filter;
use crate::read::deserialize::primitive::plain::decode_aligned_bytes_dispatch;
use crate::read::deserialize::utils::array_chunks::ArrayChunks;

/// Whether `value` starts with a UTF-8 continuation byte.
fn starts_with_continuation_byte(value: &[u8]) -> bool {
    value
        .first()
        .is_some_and(|b| b & 0b1100_0000 == 0b1000_0000)
}

/// The values of a page that the rows of a filter select, by index among its non-null values:
/// those from `start` to `end` that `mask`, if any, has.
struct Selection {
    start: usize,
    end: usize,
    mask: Option<Bitmap>,
}

impl Selection {
    /// The selection of `filter`, or `None` if it selects all values.
    fn new(
        num_values: usize,
        page_validity: Option<&Bitmap>,
        filter: Option<&Filter>,
    ) -> Option<Self> {
        // The number of values in the first `num_rows` rows.
        let num_values_in = |num_rows: usize| {
            page_validity.map_or(num_rows, |validity| {
                validity.clone().sliced(0, num_rows).set_bits()
            })
        };
        match filter {
            Some(Filter::Range(rows)) => {
                let (start, end) = (num_values_in(rows.start), num_values_in(rows.end));
                (start > 0 || end < num_values).then_some(Self {
                    start,
                    end,
                    mask: None,
                })
            },
            Some(Filter::Mask(rows)) if rows.unset_bits() > 0 => {
                // The bits of the rows that hold a value.
                let mask = page_validity.map_or_else(
                    || rows.clone(),
                    |validity| filter_boolean_kernel(rows, validity),
                );
                Some(Self {
                    start: mask.leading_zeros(),
                    end: mask.len() - mask.trailing_zeros(),
                    mask: Some(mask),
                })
            },
            _ => None,
        }
    }

    /// Whether the value at index `i`, which is less than `end`, is selected.
    fn contains(&self, i: usize) -> bool {
        i >= self.start && self.mask.as_ref().is_none_or(|mask| mask.get_bit(i))
    }
}

/// Decodes a `DELTA_LENGTH_BYTE_ARRAY` page into `state`, for the rows that `page_validity` and
/// `filter` select.
pub(super) fn decode_delta_length(
    mut decoder: delta_length_byte_array::Decoder<'_>,
    verify_utf8: bool,
    state: &mut DecodedState,
    is_optional: bool,
    page_validity: Option<&Bitmap>,
    filter: Option<Filter>,
) -> ParquetResult<()> {
    let num_values = decoder.len();
    let selection = Selection::new(num_values, page_validity, filter.as_ref());
    let is_selected = |i| selection.as_ref().is_none_or(|s| s.contains(i));
    // Values after the last selected one are not needed.
    let end = selection.as_ref().map_or(num_values, |s| s.end);
    let mut lengths = Vec::with_capacity(end);
    decoder.lengths.collect_n(&mut lengths, end)?;
    let values = decoder.values;

    // A page is at most `i32::MAX` bytes, so the offsets fit in the views.
    assert!(values.len() <= u32::MAX as usize);

    state.binview.finish_in_progress();
    let buffer_idx = state.binview.completed_buffers().len() as u32;
    let mut views = Vec::with_capacity(num_values);
    let mut offset = 0usize;
    let mut noninlined_bytes_len = 0;

    // If no value starts with a continuation byte, the values are valid UTF-8 exactly when their
    // concatenation is.
    let mut none_starting_with_continuation_byte = true;
    for &length in &lengths {
        let value = usize::try_from(length)
            .ok()
            .and_then(|length| values.get(offset..offset.checked_add(length)?))
            .ok_or_else(invalid_input_err)?;
        none_starting_with_continuation_byte &= !starts_with_continuation_byte(value);

        let length = value.len();
        if length <= View::MAX_INLINE_SIZE as usize {
            views.push(new_inline_view(&values[offset..], length));
        } else if is_selected(views.len()) {
            noninlined_bytes_len += length;
            views.push(View::new_from_bytes(value, buffer_idx, offset as u32));
        } else {
            // Only the selected non-inlined values are kept.
            views.push(View::default());
        }
        offset += length;
    }
    views.resize(num_values, View::default());

    let values = &values[..offset];
    if verify_utf8
        && !(none_starting_with_continuation_byte && simdutf8::basic::from_utf8(values).is_ok())
    {
        return Err(invalid_utf8_err());
    }
    if noninlined_bytes_len > 0 {
        // Copying all values at once is fastest, unless the others take up much of them.
        let buffer = if noninlined_bytes_len >= values.len() - values.len() / 4 {
            values.to_vec()
        } else {
            let mut buffer = Vec::with_capacity(noninlined_bytes_len);
            for view in views.iter_mut() {
                if view.length > View::MAX_INLINE_SIZE {
                    let start = view.offset as usize;
                    view.offset = buffer.len() as u32;
                    buffer.extend_from_slice(&values[start..start + view.length as usize]);
                }
            }
            buffer
        };
        let pushed_idx = state.binview.push_buffer(buffer.into());
        debug_assert_eq!(pushed_idx, buffer_idx);
    }

    // SAFETY: The non-inlined values of the views, if any, are in the buffer pushed above, at
    // `buffer_idx`.
    unsafe { extend_from_page_views(state, &views, is_optional, page_validity, filter) }
}

/// Decodes a `DELTA_BYTE_ARRAY` page into `state`, for the rows that `page_validity` and `filter`
/// select.
pub(super) fn decode_delta_bytes(
    mut decoder: delta_byte_array::Decoder<'_>,
    verify_utf8: bool,
    state: &mut DecodedState,
    is_optional: bool,
    page_validity: Option<&Bitmap>,
    filter: Option<Filter>,
) -> ParquetResult<()> {
    if decoder.prefix_lengths.len() != decoder.suffix_lengths.len() {
        return Err(invalid_input_err());
    }
    let num_values = decoder.prefix_lengths.len();
    let selection = Selection::new(num_values, page_validity, filter.as_ref());
    let is_selected = |i| selection.as_ref().is_none_or(|s| s.contains(i));
    // Values after the last selected one are not needed.
    let end = selection.as_ref().map_or(num_values, |s| s.end);
    let mut prefix_lengths = Vec::with_capacity(end);
    let mut suffix_lengths = Vec::with_capacity(end);
    decoder.prefix_lengths.collect_n(&mut prefix_lengths, end)?;
    decoder.suffix_lengths.collect_n(&mut suffix_lengths, end)?;
    let suffixes = decoder.values;

    let target = &mut state.binview;
    target.finish_in_progress();
    let first_buffer_idx = target.completed_buffers().len();
    let mut buffer_idx = first_buffer_idx as u32;
    let mut buffer = Vec::new();
    let mut views = Vec::with_capacity(num_values);

    // The previous value, followed by at least 16 bytes, so that a short value can be cut from a
    // block of 16 bytes and a short suffix copied as one.
    let mut last = vec![0; 64];
    let mut last_len = 0;
    let mut offset = 0usize;
    let mut none_starting_with_continuation_byte = true;
    for (&prefix_length, &suffix_length) in prefix_lengths.iter().zip(&suffix_lengths) {
        let prefix_length = usize::try_from(prefix_length)
            .ok()
            .filter(|&prefix_length| prefix_length <= last_len)
            .ok_or_else(invalid_input_err)?;
        let suffix = usize::try_from(suffix_length)
            .ok()
            .and_then(|suffix_length| suffixes.get(offset..offset.checked_add(suffix_length)?))
            .ok_or_else(invalid_input_err)?;

        let length = prefix_length + suffix.len();
        if last.len() < length + 16 {
            last.resize(2 * (length + 16), 0);
        }
        match suffixes[offset..].first_chunk::<16>() {
            // Assigned as an array: with `copy_from_slice`, this and the copy below are merged into
            // one `memcpy` call of either length.
            Some(block) if suffix.len() <= 16 => {
                *last[prefix_length..].first_chunk_mut::<16>().unwrap() = *block
            },
            _ => last[prefix_length..length].copy_from_slice(suffix),
        }
        offset += suffix.len();
        last_len = length;

        if length <= View::MAX_INLINE_SIZE as usize {
            views.push(new_inline_view(&last, length));
        } else if is_selected(views.len()) {
            let value = &last[..length];
            none_starting_with_continuation_byte &= !starts_with_continuation_byte(value);
            // Offsets are `u32`, so a buffer holds less than 4GiB.
            if buffer.len() + length > u32::MAX as usize {
                let pushed_idx = target.push_buffer(std::mem::take(&mut buffer).into());
                debug_assert_eq!(pushed_idx, buffer_idx);
                buffer_idx += 1;
            }
            views.push(View::new_from_bytes(value, buffer_idx, buffer.len() as u32));
            buffer.extend_from_slice(value);
        } else {
            // Only the selected non-inlined values are kept, as shared prefixes can make the values
            // much larger than the page.
            views.push(View::default());
        }
    }
    views.resize(num_values, View::default());
    if !buffer.is_empty() {
        let pushed_idx = target.push_buffer(buffer.into());
        debug_assert_eq!(pushed_idx, buffer_idx);
    }
    if verify_utf8
        && !(none_starting_with_continuation_byte
            && target.completed_buffers()[first_buffer_idx..]
                .iter()
                .all(|buffer| simdutf8::basic::from_utf8(buffer).is_ok())
            && inlined_are_valid_utf8(&views))
    {
        return Err(invalid_utf8_err());
    }

    // SAFETY: The non-inlined values of the views, if any, are in the buffers pushed above, from
    // `first_buffer_idx` on.
    unsafe { extend_from_page_views(state, &views, is_optional, page_validity, filter) }
}

/// Pushes the views of the non-null values of a page, `views`, for the rows that `page_validity`
/// and `filter` select.
///
/// # Safety
/// The non-inlined values of `views` must be in the data buffers of `state.binview`.
unsafe fn extend_from_page_views(
    state: &mut DecodedState,
    views: &[View],
    is_optional: bool,
    page_validity: Option<&Bitmap>,
    filter: Option<Filter>,
) -> ParquetResult<()> {
    let target = &mut state.binview;
    let start_length = target.views().len();
    let views = ArrayChunks::<Bytes16Alignment4>::new(bytemuck::cast_slice(views)).unwrap();

    // SAFETY: Only views of `views`, which are valid for the data buffers as the caller ensures,
    // and empty inline views for nulls are appended, and the total length is updated below, also
    // on error.
    let target_views = unsafe { target.views_mut() };
    let result =
        with_cast_mut_vec::<View, Bytes16Alignment4, _, _>(target_views, |aligned_views| {
            decode_aligned_bytes_dispatch(
                views,
                is_optional,
                page_validity,
                filter,
                &mut state.validity,
                aligned_views,
            )
        });

    let total_bytes_len: usize = target.views()[start_length..]
        .iter()
        .map(|view| view.length as usize)
        .sum();

    // SAFETY: The total length of the views grows by that of the views appended above.
    unsafe { target.set_total_bytes_len(target.total_bytes_len() + total_bytes_len) };
    result
}
