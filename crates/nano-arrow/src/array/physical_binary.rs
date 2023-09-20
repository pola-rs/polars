use crate::bitmap::MutableBitmap;
use crate::offset::{Offset, Offsets};

/// # Safety
/// The caller must ensure that `iterator` is `TrustedLen`.
#[inline]
#[allow(clippy::type_complexity)]
pub(crate) unsafe fn try_trusted_len_unzip<E, I, P, O>(
    iterator: I,
) -> std::result::Result<(Option<MutableBitmap>, Offsets<O>, Vec<u8>), E>
where
    O: Offset,
    P: AsRef<[u8]>,
    I: Iterator<Item = std::result::Result<Option<P>, E>>,
{
    let (_, upper) = iterator.size_hint();
    let len = upper.expect("trusted_len_unzip requires an upper limit");

    let mut null = MutableBitmap::with_capacity(len);
    let mut offsets = Vec::<O>::with_capacity(len + 1);
    let mut values = Vec::<u8>::new();

    let mut length = O::default();
    let mut dst = offsets.as_mut_ptr();
    std::ptr::write(dst, length);
    dst = dst.add(1);
    for item in iterator {
        if let Some(item) = item? {
            null.push_unchecked(true);
            let s = item.as_ref();
            length += O::from_usize(s.len()).unwrap();
            values.extend_from_slice(s);
        } else {
            null.push_unchecked(false);
        };

        std::ptr::write(dst, length);
        dst = dst.add(1);
    }
    assert_eq!(
        dst.offset_from(offsets.as_ptr()) as usize,
        len + 1,
        "Trusted iterator length was not accurately reported"
    );
    offsets.set_len(len + 1);

    Ok((null.into(), Offsets::new_unchecked(offsets), values))
}

/// Creates [`MutableBitmap`] and two [`Vec`]s from an iterator of `Option`.
/// The first buffer corresponds to a offset buffer, the second one
/// corresponds to a values buffer.
/// # Safety
/// The caller must ensure that `iterator` is `TrustedLen`.
#[inline]
pub(crate) unsafe fn trusted_len_unzip<O, I, P>(
    iterator: I,
) -> (Option<MutableBitmap>, Offsets<O>, Vec<u8>)
where
    O: Offset,
    P: AsRef<[u8]>,
    I: Iterator<Item = Option<P>>,
{
    let (_, upper) = iterator.size_hint();
    let len = upper.expect("trusted_len_unzip requires an upper limit");

    let mut offsets = Offsets::<O>::with_capacity(len);
    let mut values = Vec::<u8>::new();
    let mut validity = MutableBitmap::new();

    extend_from_trusted_len_iter(&mut offsets, &mut values, &mut validity, iterator);

    let validity = if validity.unset_bits() > 0 {
        Some(validity)
    } else {
        None
    };

    (validity, offsets, values)
}

/// Creates two [`Buffer`]s from an iterator of `&[u8]`.
/// The first buffer corresponds to a offset buffer, the second to a values buffer.
/// # Safety
/// The caller must ensure that `iterator` is [`TrustedLen`].
#[inline]
pub(crate) unsafe fn trusted_len_values_iter<O, I, P>(iterator: I) -> (Offsets<O>, Vec<u8>)
where
    O: Offset,
    P: AsRef<[u8]>,
    I: Iterator<Item = P>,
{
    let (_, upper) = iterator.size_hint();
    let len = upper.expect("trusted_len_unzip requires an upper limit");

    let mut offsets = Offsets::<O>::with_capacity(len);
    let mut values = Vec::<u8>::new();

    extend_from_trusted_len_values_iter(&mut offsets, &mut values, iterator);

    (offsets, values)
}

// Populates `offsets` and `values` [`Vec`]s with information extracted
// from the incoming `iterator`.
// # Safety
// The caller must ensure the `iterator` is [`TrustedLen`]
#[inline]
pub(crate) unsafe fn extend_from_trusted_len_values_iter<I, P, O>(
    offsets: &mut Offsets<O>,
    values: &mut Vec<u8>,
    iterator: I,
) where
    O: Offset,
    P: AsRef<[u8]>,
    I: Iterator<Item = P>,
{
    let lengths = iterator.map(|item| {
        let s = item.as_ref();
        // Push new entries for both `values` and `offsets` buffer
        values.extend_from_slice(s);
        s.len()
    });
    offsets.try_extend_from_lengths(lengths).unwrap();
}

// Populates `offsets` and `values` [`Vec`]s with information extracted
// from the incoming `iterator`.
// the return value indicates how many items were added.
#[inline]
pub(crate) fn extend_from_values_iter<I, P, O>(
    offsets: &mut Offsets<O>,
    values: &mut Vec<u8>,
    iterator: I,
) -> usize
where
    O: Offset,
    P: AsRef<[u8]>,
    I: Iterator<Item = P>,
{
    let (size_hint, _) = iterator.size_hint();

    offsets.reserve(size_hint);

    let start_index = offsets.len_proxy();

    for item in iterator {
        let bytes = item.as_ref();
        values.extend_from_slice(bytes);
        offsets.try_push_usize(bytes.len()).unwrap();
    }
    offsets.len_proxy() - start_index
}

// Populates `offsets`, `values`, and `validity` [`Vec`]s with
// information extracted from the incoming `iterator`.
//
// # Safety
// The caller must ensure that `iterator` is [`TrustedLen`]
#[inline]
pub(crate) unsafe fn extend_from_trusted_len_iter<O, I, P>(
    offsets: &mut Offsets<O>,
    values: &mut Vec<u8>,
    validity: &mut MutableBitmap,
    iterator: I,
) where
    O: Offset,
    P: AsRef<[u8]>,
    I: Iterator<Item = Option<P>>,
{
    let (_, upper) = iterator.size_hint();
    let additional = upper.expect("extend_from_trusted_len_iter requires an upper limit");

    offsets.reserve(additional);
    validity.reserve(additional);

    let lengths = iterator.map(|item| {
        if let Some(item) = item {
            let bytes = item.as_ref();
            values.extend_from_slice(bytes);
            validity.push_unchecked(true);
            bytes.len()
        } else {
            validity.push_unchecked(false);
            0
        }
    });
    offsets.try_extend_from_lengths(lengths).unwrap();
}

/// Creates two [`Vec`]s from an iterator of `&[u8]`.
/// The first buffer corresponds to a offset buffer, the second to a values buffer.
#[inline]
pub(crate) fn values_iter<O, I, P>(iterator: I) -> (Offsets<O>, Vec<u8>)
where
    O: Offset,
    P: AsRef<[u8]>,
    I: Iterator<Item = P>,
{
    let (lower, _) = iterator.size_hint();

    let mut offsets = Offsets::<O>::with_capacity(lower);
    let mut values = Vec::<u8>::new();

    for item in iterator {
        let s = item.as_ref();
        values.extend_from_slice(s);
        offsets.try_push_usize(s.len()).unwrap();
    }
    (offsets, values)
}

/// Extends `validity` with all items from `other`
pub(crate) fn extend_validity(
    length: usize,
    validity: &mut Option<MutableBitmap>,
    other: &Option<MutableBitmap>,
) {
    if let Some(other) = other {
        if let Some(validity) = validity {
            let slice = other.as_slice();
            // safety: invariant offset + length <= slice.len()
            unsafe { validity.extend_from_slice_unchecked(slice, 0, other.len()) }
        } else {
            let mut new_validity = MutableBitmap::from_len_set(length);
            new_validity.extend_from_slice(other.as_slice(), 0, other.len());
            *validity = Some(new_validity);
        }
    }
}
