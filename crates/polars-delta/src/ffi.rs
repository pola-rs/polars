//! `delta.v2` deletion-vector decoders over the C ABI, served from [`crate::codec`].
use polars_io_ext_ffi::common::{FfiBuf, FfiError, FfiErrorKind, FfiResult, FfiSlice};
use polars_io_ext_ffi::delta_v2::KeepMask;

use crate::codec::{self, DecodeError};

fn mask(result: Result<codec::KeepMask, DecodeError>) -> Result<KeepMask, FfiError> {
    let m = result.map_err(|e| FfiError::new(FfiErrorKind::OTHER, e.to_string()))?;
    Ok(KeepMask {
        len: m.len as u64,
        // Ownership moves across without a copy; the host drops it through the release callback.
        bytes: FfiBuf::from_vec(m.bytes),
    })
}

fn max_rows(v: u64) -> usize {
    usize::try_from(v).unwrap_or(usize::MAX)
}

pub(crate) unsafe extern "C" fn decode_inline(
    data: FfiSlice<'_, u8>,
    size: u32,
    cardinality: u64,
    max_rows_: u64,
) -> FfiResult<KeepMask> {
    crate::catch_panic("deletion vector decoding", || {
        mask(codec::decode_inline(
            data.as_slice(),
            size,
            cardinality,
            max_rows(max_rows_),
        ))
    })
    .into()
}

pub(crate) unsafe extern "C" fn decode_frame(
    frame: FfiSlice<'_, u8>,
    offset: u64,
    expected_size: u32,
    cardinality: u64,
    max_rows_: u64,
) -> FfiResult<KeepMask> {
    crate::catch_panic("deletion vector decoding", || {
        codec::validate_frame_offset(offset)
            .map_err(|e| FfiError::new(FfiErrorKind::OTHER, e.to_string()))?;
        mask(codec::decode_framed(
            frame.as_slice(),
            Some(expected_size),
            Some(cardinality),
            max_rows(max_rows_),
        ))
    })
    .into()
}

pub(crate) unsafe extern "C" fn check_file_version(header: FfiSlice<'_, u8>) -> FfiResult<()> {
    crate::catch_panic("deletion vector decoding", || {
        codec::validate_file_version(header.as_slice())
            .map_err(|e| FfiError::new(FfiErrorKind::OTHER, e.to_string()))
    })
    .into()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bitmap() -> Vec<u8> {
        // Rows 3 and 5 of a single array container.
        let mut b = 1681511377u32.to_le_bytes().to_vec();
        b.extend_from_slice(&1u64.to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes());
        b.extend_from_slice(&12346u32.to_le_bytes());
        b.extend_from_slice(&1u32.to_le_bytes());
        b.extend_from_slice(&0u16.to_le_bytes());
        b.extend_from_slice(&1u16.to_le_bytes());
        b.extend_from_slice(&16u32.to_le_bytes());
        b.extend_from_slice(&3u16.to_le_bytes());
        b.extend_from_slice(&5u16.to_le_bytes());
        b
    }

    fn frame(body: &[u8]) -> Vec<u8> {
        let mut f = (body.len() as u32).to_be_bytes().to_vec();
        f.extend_from_slice(body);
        f.extend_from_slice(&crc32fast::hash(body).to_be_bytes());
        f
    }

    #[test]
    fn decode_over_the_abi() {
        let body = bitmap();
        let f = frame(&body);
        let size = body.len() as u32;
        for mask in [
            unsafe { decode_inline(FfiSlice::new(&body), size, 2, 100) },
            unsafe { decode_frame(FfiSlice::new(&f), 1, size, 2, 100) },
        ] {
            let mask = mask.into_result().unwrap();
            assert_eq!(mask.len, 6);
            assert_eq!(mask.bytes.as_slice(), [0b1101_0111]);
        }
        let mut bad = f.clone();
        bad[6] ^= 1;
        let Err(err) = unsafe { decode_frame(FfiSlice::new(&bad), 1, size, 2, 100) }.into_result()
        else {
            panic!("CRC mismatch must fail")
        };
        assert!(err.message().contains("CRC"));
        // Wrong size prefix, short frame and max_rows.
        assert!(
            unsafe { decode_frame(FfiSlice::new(&f), 1, size + 1, 2, 100) }
                .into_result()
                .is_err()
        );
        assert!(
            unsafe { decode_frame(FfiSlice::new(&f[..f.len() - 1]), 1, size, 2, 100) }
                .into_result()
                .is_err()
        );
        assert!(
            unsafe { decode_frame(FfiSlice::new(&f), 1, size, 2, 5) }
                .into_result()
                .is_err()
        );
        assert!(
            unsafe { check_file_version(FfiSlice::new(&[1])) }
                .into_result()
                .is_ok()
        );
        assert!(
            unsafe { check_file_version(FfiSlice::new(&[2])) }
                .into_result()
                .is_err()
        );
    }
}
