//! Casting the offset-backed binary arrays of `polars-array`.

use arrow::array::{BINVIEW_MAX_ROW_BYTE_LEN, View};
use arrow::types::NativeType;
#[cfg(feature = "dtype-f16")]
use num_traits::AsPrimitive;
use polars_array::{
    PlBinaryArray, PlBinaryViewArray, PlBitmap, PlFixedSizeBinaryArray, PlListArray,
    PlPrimitiveArray,
};
use polars_buffer::Buffer;
use polars_utils::unitvec;

use super::{CastOptionsImpl, map_bytes_fallible};

/// The number a text stands for, which is how a cast off a string reads its elements.
pub trait Parse {
    fn parse(val: &[u8]) -> Option<Self>
    where
        Self: Sized;
}

macro_rules! impl_parse {
    ($primitive_type:ident) => {
        impl Parse for $primitive_type {
            fn parse(val: &[u8]) -> Option<Self> {
                atoi_simd::parse::<_, true, true>(val).ok()
            }
        }
    };
}
impl_parse!(i8);
impl_parse!(i16);
impl_parse!(i32);
impl_parse!(i64);
#[cfg(feature = "dtype-i128")]
impl_parse!(i128);

impl_parse!(u8);
impl_parse!(u16);
impl_parse!(u32);
impl_parse!(u64);
#[cfg(feature = "dtype-u128")]
impl_parse!(u128);

#[cfg(feature = "dtype-f16")]
impl Parse for polars_utils::float16::pf16 {
    fn parse(val: &[u8]) -> Option<Self>
    where
        Self: Sized,
    {
        fast_float2::parse(val).ok().map(|f: f32| f.as_())
    }
}

impl Parse for f32 {
    fn parse(val: &[u8]) -> Option<Self>
    where
        Self: Sized,
    {
        fast_float2::parse(val).ok()
    }
}
impl Parse for f64 {
    fn parse(val: &[u8]) -> Option<Self>
    where
        Self: Sized,
    {
        fast_float2::parse(val).ok()
    }
}

/// Reads the text of every element as the number it stands for, leaving a null where it stands for
/// none.
pub fn binary_to_parsed<T: NativeType + Parse>(
    from: &PlBinaryArray,
    options: CastOptionsImpl,
) -> PlPrimitiveArray<T> {
    if options.partial {
        unimplemented!()
    }

    map_bytes_fallible(
        from.len(),
        from.scalar_value_ignore_validity(),
        from.broadcast_values_iter(from.len()),
        from.validity(),
        T::parse,
    )
}

// A buffer of an Arrow view array is addressed with a `u32`, and is limited to an `i32` so that
// the consumers that read the offset as a signed integer can read it too.
const ARROW_MAX_OFFSET: usize = if cfg!(test) {
    // Used to test buffer splitting.
    i8::MAX as usize
} else {
    i32::MAX as usize
};

/// Reads the bytes of every element through a view of them, which is what a binary view array is.
pub fn binary_to_binview(from: &PlBinaryArray) -> PlBinaryViewArray {
    // The one value every element of a scalar chunk reads is viewed once, and the view repeats it.
    if let Some(value) = from.scalar_value_ignore_validity() {
        return PlBinaryViewArray::new_scalar(value, from.len())
            .with_validity(from.validity().map(PlBitmap::from));
    }

    let offsets = from.flat_offsets().unwrap();
    let mut views = Vec::with_capacity(from.len());

    let mut buffers = unitvec![];
    let mut current_buffer_range: std::ops::Range<usize> = 0..0;

    for row_byte_range in offsets
        .windows(2)
        .map(|window| window[0] as usize..window[1] as usize)
    {
        let row_byte_len: usize = row_byte_range.len();
        assert!(
            row_byte_len <= BINVIEW_MAX_ROW_BYTE_LEN,
            "max string/binary length exceeded"
        );

        // SAFETY: an offset of an array of this crate holds a range within its values.
        let row_byte_values = unsafe { from.values().get_unchecked(row_byte_range.clone()) };

        let view = if row_byte_len <= View::MAX_INLINE_SIZE as usize {
            // SAFETY: the bytes were just checked to fit a view of their own.
            unsafe { View::new_inline_unchecked(row_byte_values) }
        } else {
            if row_byte_range.end > current_buffer_range.end {
                let new_buffer_end = usize::min(
                    from.values().len(),
                    usize::max(
                        row_byte_range.end,
                        row_byte_range.start.saturating_add(ARROW_MAX_OFFSET),
                    ),
                );

                let new_buffer_range = row_byte_range.start..new_buffer_end;

                assert!(
                    buffers.len() < u32::MAX as usize,
                    "max string/binary buffers exceeded"
                );
                // SAFETY: the range ends within the values, which is where it came from.
                buffers.push(unsafe {
                    from.values()
                        .clone()
                        .sliced_unchecked(new_buffer_range.clone())
                });
                current_buffer_range = new_buffer_range;
            }

            let offset: u32 = (row_byte_range.start - current_buffer_range.start) as u32;
            // SAFETY: the bytes were just checked not to fit a view of their own, and the buffer
            // they are read out of was just pushed.
            unsafe {
                View::new_noninline_unchecked(row_byte_values, (buffers.len() - 1) as u32, offset)
            }
        };

        views.push(view);
    }

    // SAFETY: every view was just built against the buffer it reads.
    unsafe {
        PlBinaryViewArray::new_unchecked(
            views.into(),
            Buffer::from_owner(buffers),
            from.len(),
            from.validity().map(PlBitmap::from),
        )
    }
}

/// Reads the bytes of every element through a view of them, one view per element.
pub fn fixed_size_binary_to_binview(from: &PlFixedSizeBinaryArray) -> PlBinaryViewArray {
    // The one value every element of a scalar chunk reads is viewed once, and the view repeats it.
    if let Some(value) = from.scalar_value_ignore_validity() {
        return PlBinaryViewArray::new_scalar(value, from.len())
            .with_validity(from.validity().map(PlBitmap::from));
    }

    let width = from.width();
    let values = from.flat_values().unwrap();

    // Fast path: every element fits a view of its own, so no buffer is read at all.
    if width <= View::MAX_INLINE_SIZE as usize {
        let mut views = Vec::new();
        // SAFETY: the width was just checked to fit a view.
        View::extend_with_inlinable_strided(&mut views, values.as_slice(), width as u8);
        return PlBinaryViewArray::from_views(views.into(), Buffer::new())
            .with_validity(from.validity().map(PlBitmap::from));
    }

    // The values already lie end to end, which is the layout of a view array's buffer, so the
    // cast reads them out of the buffer they are in — split up where a view can no longer address
    // it.
    let max_bytes_per_buffer = if width <= ARROW_MAX_OFFSET {
        ARROW_MAX_OFFSET
    } else {
        BINVIEW_MAX_ROW_BYTE_LEN
    };
    assert!(width <= max_bytes_per_buffer);
    let elements_per_buffer = max_bytes_per_buffer / width;
    let split_point = elements_per_buffer * width;
    let num_buffers = (from.len() * width).div_ceil(max_bytes_per_buffer);
    assert!(num_buffers < u32::MAX as usize);

    let mut buffer = values.clone();
    let mut buffers = Vec::with_capacity(num_buffers);
    if let Some(splits) = num_buffers.checked_sub(1) {
        for _ in 0..splits {
            let slice;
            (slice, buffer) = buffer.split_at(split_point);
            buffers.push(slice);
        }
        buffers.push(buffer);
    }

    let mut views = Vec::with_capacity(from.len());
    let mut iter = from.values_iter();
    let iter = iter.by_ref();
    for buffer_idx in 0..num_buffers {
        views.extend(
            iter.take(elements_per_buffer)
                .enumerate()
                .map(|(i, value)| {
                    // SAFETY: the width was just checked not to fit a view, and the buffer holds
                    // the element at that offset.
                    unsafe {
                        View::new_noninline_unchecked(value, buffer_idx as u32, (i * width) as u32)
                    }
                }),
        );
    }

    PlBinaryViewArray::from_views(views.into(), buffers.into())
        .with_validity(from.validity().map(PlBitmap::from))
}

/// Reads the bytes of every element as its own values, which is how a binary reads as a list of
/// bytes.
pub fn binary_to_list(from: &PlBinaryArray) -> PlListArray {
    let values = Box::new(PlPrimitiveArray::from_values(from.values().clone()));
    let validity = from.validity().map(PlBitmap::from);

    match from.scalar_offsets() {
        Some(range) => PlListArray::new_broadcast(
            values,
            Buffer::from(vec![range.start as u64, range.end as u64]),
            from.len(),
            validity,
        ),
        None => PlListArray::new(
            values,
            from.flat_offsets().unwrap().clone(),
            from.len(),
            validity,
        ),
    }
}

#[cfg(test)]
mod tests {
    use polars_array::arrow::export;

    use super::*;

    /// The views of a long element read a buffer at an offset an Arrow consumer reads as an `i32`,
    /// so the values are split across buffers once they no longer fit one.
    #[test]
    fn binary_to_binview_splits_its_buffers() {
        let values = [
            "lksafjdlkakjslkjsafkjdalkjfalkdsalkjfaslkfjlkakdsjfkajfksdaj", // 0 (offset)
            "123",                                                          // inline
            "lksafjdlkakjslkjsafkjdalkjfalkdsalkjfaslkfjlkakdsjfkajfksdaj", // 60
            "lksafjdlkakjslkjsafkjdalkjfalkdsalkjfaslkfjlkakdsjfkajfksdaj", // 0 (new buffer)
            "lksafjdlkakjslkjsafkjdalkjfalkdsalkjfaslkfjlkakdsjfkajfksdaj", // 60
            "234",                                                          // inline
            "lksafjdlkakjslkjsafkjdalkjfalkdsalkjfaslkfjlkakdsjfkajfksdaj", // 0 (new buffer)
            "lksafjdlkakjslkjsafkjdalkjfalkdsalkjfaslkfjlkakdsjfkajfksdaj", // 60
            "lksafjdlkakjslkjsafkjdalkjfalkdsalkjfaslkfjlkakdsjfkajfksdaj", // 0 (new buffer)
            "lksafjdlkakjslkjsafkjdalkjfalkdsalkjfaslkfjlkakdsjfkajfksdaj", // 60
            "324",                                                          // inline
        ];
        let array = PlBinaryArray::from_values_iter(values.iter().map(|value| value.as_bytes()));

        let out = binary_to_binview(&array);

        // Ensure we hit the multiple buffers part.
        assert_eq!(
            export::binview_to_arrow_binview(&out).data_buffers().len(),
            4
        );
        // Ensure we created a valid binview.
        let read = out
            .values_iter()
            .map(|value| std::str::from_utf8(value).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(read, values);
    }

    /// A chunk that repeats one value is viewed once, and the view repeats it in turn.
    #[test]
    fn binary_to_binview_keeps_a_scalar_chunk_scalar() {
        let array = PlBinaryArray::new_scalar(b"the value every element reads", 8);

        let out = binary_to_binview(&array);

        assert!(out.is_scalar());
        assert_eq!(out.len(), 8);
        assert_eq!(out.value(7), b"the value every element reads");
    }
}
