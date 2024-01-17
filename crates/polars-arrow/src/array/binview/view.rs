use polars_error::*;

use crate::buffer::Buffer;

pub struct View {
    /// The length of the string/bytes.
    pub length: u32,
    /// First 4 bytes of string/bytes data.
    pub prefix: u32,
    /// The buffer index.
    pub buffer_idx: u32,
    /// The offset into the buffer.
    pub offset: u32,
}

impl From<u128> for View {
    #[inline]
    fn from(value: u128) -> Self {
        Self {
            length: value as u32,
            prefix: (value >> 64) as u32,
            buffer_idx: (value >> 64) as u32,
            offset: (value >> 96) as u32,
        }
    }
}

impl From<View> for u128 {
    #[inline]
    fn from(value: View) -> Self {
        value.length as u128
            | ((value.prefix as u128) << 32)
            | ((value.buffer_idx as u128) << 64)
            | ((value.offset as u128) << 96)
    }
}

fn validate_view<F>(views: &[u128], buffers: &[Buffer<u8>], validate_bytes: F) -> PolarsResult<()>
where
    F: Fn(&[u8]) -> PolarsResult<()>,
{
    for view in views {
        let len = *view as u32;
        if len <= 12 {
            if len < 12 && view >> (32 + len * 8) != 0 {
                polars_bail!(ComputeError: "view contained non-zero padding in prefix");
            }

            validate_bytes(&view.to_le_bytes()[4..4 + len as usize])?;
        } else {
            let view = View::from(*view);

            let data = buffers.get(view.buffer_idx as usize).ok_or_else(|| {
                polars_err!(OutOfBounds: "view index out of bounds\n\nGot: {} buffers and index: {}", buffers.len(), view.buffer_idx)
            })?;

            let start = view.offset as usize;
            let end = start + len as usize;
            let b = data
                .as_slice()
                .get(start..end)
                .ok_or_else(|| polars_err!(OutOfBounds: "buffer slice out of bounds"))?;

            polars_ensure!(b.starts_with(&view.prefix.to_le_bytes()), ComputeError: "prefix does not match string data");
            validate_bytes(b)?;
        };
    }

    Ok(())
}

pub(super) fn validate_binary_view(views: &[u128], buffers: &[Buffer<u8>]) -> PolarsResult<()> {
    validate_view(views, buffers, |_| Ok(()))
}

fn validate_utf8(b: &[u8]) -> PolarsResult<()> {
    match simdutf8::basic::from_utf8(b) {
        Ok(_) => Ok(()),
        Err(_) => Err(polars_err!(ComputeError: "invalid utf8")),
    }
}

pub(super) fn validate_utf8_view(views: &[u128], buffers: &[Buffer<u8>]) -> PolarsResult<()> {
    validate_view(views, buffers, validate_utf8)
}

pub(super) fn validate_utf8_only(views: &[u128], buffers: &[Buffer<u8>]) -> PolarsResult<()> {
    for view in views {
        let len = *view as u32;
        if len <= 12 {
            validate_utf8(&view.to_le_bytes()[4..4 + len as usize])?;
        } else {
            let view = View::from(*view);
            let data = &buffers[view.buffer_idx as usize];

            let start = view.offset as usize;
            let end = start + len as usize;
            let b = &data.as_slice()[start..end];
            validate_utf8(b)?;
        };
    }

    Ok(())
}
