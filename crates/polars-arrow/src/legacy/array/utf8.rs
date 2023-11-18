use crate::array::{BinaryArray, Utf8Array};
use crate::datatypes::ArrowDataType;
use crate::legacy::trusted_len::TrustedLenPush;
use crate::offset::Offsets;

#[inline]
unsafe fn extend_from_trusted_len_values_iter<I, P>(
    offsets: &mut Vec<i64>,
    values: &mut Vec<u8>,
    iterator: I,
) where
    P: AsRef<[u8]>,
    I: Iterator<Item = P>,
{
    let mut total_length = 0;
    offsets.push(total_length);
    iterator.for_each(|item| {
        let s = item.as_ref();
        // Push new entries for both `values` and `offsets` buffer
        values.extend_from_slice(s);

        total_length += s.len() as i64;
        offsets.push_unchecked(total_length);
    });
}

/// # Safety
/// reported `len` must be correct.
#[inline]
unsafe fn fill_offsets_and_values<I, P>(
    iterator: I,
    value_capacity: usize,
    len: usize,
) -> (Offsets<i64>, Vec<u8>)
where
    P: AsRef<[u8]>,
    I: Iterator<Item = P>,
{
    let mut offsets = Vec::with_capacity(len + 1);
    let mut values = Vec::<u8>::with_capacity(value_capacity);

    extend_from_trusted_len_values_iter(&mut offsets, &mut values, iterator);

    (Offsets::new_unchecked(offsets), values)
}

struct StrAsBytes<P>(P);
impl<T: AsRef<str>> AsRef<[u8]> for StrAsBytes<T> {
    #[inline(always)]
    fn as_ref(&self) -> &[u8] {
        self.0.as_ref().as_bytes()
    }
}

pub trait Utf8FromIter {
    #[inline]
    fn from_values_iter<I, S>(iter: I, len: usize, size_hint: usize) -> Utf8Array<i64>
    where
        S: AsRef<str>,
        I: Iterator<Item = S>,
    {
        let iter = iter.map(StrAsBytes);
        let (offsets, values) = unsafe { fill_offsets_and_values(iter, size_hint, len) };
        unsafe {
            Utf8Array::new_unchecked(
                ArrowDataType::LargeUtf8,
                offsets.into(),
                values.into(),
                None,
            )
        }
    }
}

impl Utf8FromIter for Utf8Array<i64> {}

pub trait BinaryFromIter {
    #[inline]
    fn from_values_iter<I, S>(iter: I, len: usize, value_cap: usize) -> BinaryArray<i64>
    where
        S: AsRef<[u8]>,
        I: Iterator<Item = S>,
    {
        let (offsets, values) = unsafe { fill_offsets_and_values(iter, value_cap, len) };
        BinaryArray::new(
            ArrowDataType::LargeBinary,
            offsets.into(),
            values.into(),
            None,
        )
    }
}

impl BinaryFromIter for BinaryArray<i64> {}
