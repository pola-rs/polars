use std::hint::unreachable_unchecked;
use std::sync::Arc;

use polars_error::{polars_bail, PolarsResult};

use super::utils::{
    count_zeros, fmt, get_bit, set, set_bit, BitChunk, BitChunks, BitChunksExactMut, BitmapIter,
};
use super::{intersects_with_mut, Bitmap};
use crate::bitmap::utils::{get_bit_unchecked, merge_reversed, set_bit_unchecked};
use crate::trusted_len::TrustedLen;

/// A container of booleans. [`MutableBitmap`] is semantically equivalent
/// to [`Vec<bool>`].
///
/// The two main differences against [`Vec<bool>`] is that each element stored as a single bit,
/// thereby:
/// * it uses 8x less memory
/// * it cannot be represented as `&[bool]` (i.e. no pointer arithmetics).
///
/// A [`MutableBitmap`] can be converted to a [`Bitmap`] at `O(1)`.
/// # Examples
/// ```
/// use polars_arrow::bitmap::MutableBitmap;
///
/// let bitmap = MutableBitmap::from([true, false, true]);
/// assert_eq!(bitmap.iter().collect::<Vec<_>>(), vec![true, false, true]);
///
/// // creation directly from bytes
/// let mut bitmap = MutableBitmap::try_new(vec![0b00001101], 5).unwrap();
/// // note: the first bit is the left-most of the first byte
/// assert_eq!(bitmap.iter().collect::<Vec<_>>(), vec![true, false, true, true, false]);
/// // we can also get the slice:
/// assert_eq!(bitmap.as_slice(), [0b00001101u8].as_ref());
/// // debug helps :)
/// assert_eq!(format!("{:?}", bitmap), "Bitmap { len: 5, offset: 0, bytes: [0b___01101] }");
///
/// // It supports mutation in place
/// bitmap.set(0, false);
/// assert_eq!(format!("{:?}", bitmap), "Bitmap { len: 5, offset: 0, bytes: [0b___01100] }");
/// // and `O(1)` random access
/// assert_eq!(bitmap.get(0), false);
/// ```
/// # Implementation
/// This container is internally a [`Vec<u8>`].
#[derive(Clone)]
pub struct MutableBitmap {
    buffer: Vec<u8>,
    // invariant: length.saturating_add(7) / 8 == buffer.len();
    length: usize,
}

impl std::fmt::Debug for MutableBitmap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fmt(&self.buffer, 0, self.len(), f)
    }
}

impl PartialEq for MutableBitmap {
    fn eq(&self, other: &Self) -> bool {
        self.iter().eq(other.iter())
    }
}

impl MutableBitmap {
    /// Initializes an empty [`MutableBitmap`].
    #[inline]
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            length: 0,
        }
    }

    /// Initializes a new [`MutableBitmap`] from a [`Vec<u8>`] and a length.
    /// # Errors
    /// This function errors iff `length > bytes.len() * 8`
    #[inline]
    pub fn try_new(mut bytes: Vec<u8>, length: usize) -> PolarsResult<Self> {
        if length > bytes.len().saturating_mul(8) {
            polars_bail!(InvalidOperation:
                "The length of the bitmap ({}) must be `<=` to the number of bytes times 8 ({})",
                length,
                bytes.len().saturating_mul(8)
            )
        }

        // Ensure invariant holds.
        let min_byte_length_needed = length.div_ceil(8);
        bytes.drain(min_byte_length_needed..);
        Ok(Self {
            length,
            buffer: bytes,
        })
    }

    /// Initializes a [`MutableBitmap`] from a [`Vec<u8>`] and a length.
    /// This function is `O(1)`.
    /// # Panic
    /// Panics iff the length is larger than the length of the buffer times 8.
    #[inline]
    pub fn from_vec(buffer: Vec<u8>, length: usize) -> Self {
        Self::try_new(buffer, length).unwrap()
    }

    /// Initializes a pre-allocated [`MutableBitmap`] with capacity for `capacity` bits.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity.saturating_add(7) / 8),
            length: 0,
        }
    }

    /// Pushes a new bit to the [`MutableBitmap`], re-sizing it if necessary.
    #[inline]
    pub fn push(&mut self, value: bool) {
        if self.length % 8 == 0 {
            self.buffer.push(0);
        }
        let byte = unsafe { self.buffer.as_mut_slice().last_mut().unwrap_unchecked() };
        *byte = set(*byte, self.length % 8, value);
        self.length += 1;
    }

    /// Pop the last bit from the [`MutableBitmap`].
    /// Note if the [`MutableBitmap`] is empty, this method will return None.
    #[inline]
    pub fn pop(&mut self) -> Option<bool> {
        if self.is_empty() {
            return None;
        }

        self.length -= 1;
        let value = unsafe { self.get_unchecked(self.length) };
        if self.length % 8 == 0 {
            self.buffer.pop();
        }
        Some(value)
    }

    /// Returns whether the position `index` is set.
    /// # Panics
    /// Panics iff `index >= self.len()`.
    #[inline]
    pub fn get(&self, index: usize) -> bool {
        get_bit(&self.buffer, index)
    }

    /// Returns whether the position `index` is set.
    ///
    /// # Safety
    /// The caller must ensure `index < self.len()`.
    #[inline]
    pub unsafe fn get_unchecked(&self, index: usize) -> bool {
        get_bit_unchecked(&self.buffer, index)
    }

    /// Sets the position `index` to `value`
    /// # Panics
    /// Panics iff `index >= self.len()`.
    #[inline]
    pub fn set(&mut self, index: usize, value: bool) {
        set_bit(self.buffer.as_mut_slice(), index, value)
    }

    /// constructs a new iterator over the bits of [`MutableBitmap`].
    pub fn iter(&self) -> BitmapIter {
        BitmapIter::new(&self.buffer, 0, self.length)
    }

    /// Empties the [`MutableBitmap`].
    #[inline]
    pub fn clear(&mut self) {
        self.length = 0;
        self.buffer.clear();
    }

    /// Extends [`MutableBitmap`] by `additional` values of constant `value`.
    /// # Implementation
    /// This function is an order of magnitude faster than pushing element by element.
    #[inline]
    pub fn extend_constant(&mut self, additional: usize, value: bool) {
        if additional == 0 {
            return;
        }

        if value {
            self.extend_set(additional)
        } else {
            self.extend_unset(additional)
        }
    }

    /// Initializes a zeroed [`MutableBitmap`].
    #[inline]
    pub fn from_len_zeroed(length: usize) -> Self {
        Self {
            buffer: vec![0; length.saturating_add(7) / 8],
            length,
        }
    }

    /// Initializes a [`MutableBitmap`] with all values set to valid/ true.
    #[inline]
    pub fn from_len_set(length: usize) -> Self {
        Self {
            buffer: vec![u8::MAX; length.saturating_add(7) / 8],
            length,
        }
    }

    /// Reserves `additional` bits in the [`MutableBitmap`], potentially re-allocating its buffer.
    #[inline(always)]
    pub fn reserve(&mut self, additional: usize) {
        self.buffer
            .reserve((self.length + additional).saturating_add(7) / 8 - self.buffer.len())
    }

    /// Returns the capacity of [`MutableBitmap`] in number of bits.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.buffer.capacity() * 8
    }

    /// Pushes a new bit to the [`MutableBitmap`]
    ///
    /// # Safety
    /// The caller must ensure that the [`MutableBitmap`] has sufficient capacity.
    #[inline]
    pub unsafe fn push_unchecked(&mut self, value: bool) {
        if self.length % 8 == 0 {
            self.buffer.push(0);
        }
        let byte = self.buffer.as_mut_slice().last_mut().unwrap();
        *byte = set(*byte, self.length % 8, value);
        self.length += 1;
    }

    /// Returns the number of unset bits on this [`MutableBitmap`].
    ///
    /// Guaranteed to be `<= self.len()`.
    /// # Implementation
    /// This function is `O(N)`
    pub fn unset_bits(&self) -> usize {
        count_zeros(&self.buffer, 0, self.length)
    }

    /// Returns the number of set bits on this [`MutableBitmap`].
    ///
    /// Guaranteed to be `<= self.len()`.
    /// # Implementation
    /// This function is `O(N)`
    pub fn set_bits(&self) -> usize {
        self.length - self.unset_bits()
    }

    /// Returns the length of the [`MutableBitmap`].
    #[inline]
    pub fn len(&self) -> usize {
        self.length
    }

    /// Returns whether [`MutableBitmap`] is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// # Safety
    /// The caller must ensure that the [`MutableBitmap`] was properly initialized up to `len`.
    #[inline]
    pub(crate) unsafe fn set_len(&mut self, len: usize) {
        self.buffer.set_len(len.saturating_add(7) / 8);
        self.length = len;
    }

    fn extend_set(&mut self, mut additional: usize) {
        let offset = self.length % 8;
        let added = if offset != 0 {
            // offset != 0 => at least one byte in the buffer
            let last_index = self.buffer.len() - 1;
            let last = &mut self.buffer[last_index];

            let remaining = 0b11111111u8;
            let remaining = remaining >> 8usize.saturating_sub(additional);
            let remaining = remaining << offset;
            *last |= remaining;
            std::cmp::min(additional, 8 - offset)
        } else {
            0
        };
        self.length += added;
        additional = additional.saturating_sub(added);
        if additional > 0 {
            debug_assert_eq!(self.length % 8, 0);
            let existing = self.length.saturating_add(7) / 8;
            let required = (self.length + additional).saturating_add(7) / 8;
            // add remaining as full bytes
            self.buffer
                .extend(std::iter::repeat(0b11111111u8).take(required - existing));
            self.length += additional;
        }
    }

    fn extend_unset(&mut self, mut additional: usize) {
        let offset = self.length % 8;
        let added = if offset != 0 {
            // offset != 0 => at least one byte in the buffer
            let last_index = self.buffer.len() - 1;
            let last = &mut self.buffer[last_index];
            *last &= 0b11111111u8 >> (8 - offset); // unset them
            std::cmp::min(additional, 8 - offset)
        } else {
            0
        };
        self.length += added;
        additional = additional.saturating_sub(added);
        if additional > 0 {
            debug_assert_eq!(self.length % 8, 0);
            self.buffer
                .resize((self.length + additional).saturating_add(7) / 8, 0);
            self.length += additional;
        }
    }

    /// Sets the position `index` to `value`
    ///
    /// # Safety
    /// Caller must ensure that `index < self.len()`
    #[inline]
    pub unsafe fn set_unchecked(&mut self, index: usize, value: bool) {
        set_bit_unchecked(self.buffer.as_mut_slice(), index, value)
    }

    /// Shrinks the capacity of the [`MutableBitmap`] to fit its current length.
    pub fn shrink_to_fit(&mut self) {
        self.buffer.shrink_to_fit();
    }

    /// Returns an iterator over bits in bit chunks [`BitChunk`].
    ///
    /// This iterator is useful to operate over multiple bits via e.g. bitwise.
    pub fn chunks<T: BitChunk>(&self) -> BitChunks<T> {
        BitChunks::new(&self.buffer, 0, self.length)
    }

    /// Returns an iterator over mutable slices, [`BitChunksExactMut`]
    pub(crate) fn bitchunks_exact_mut<T: BitChunk>(&mut self) -> BitChunksExactMut<T> {
        BitChunksExactMut::new(&mut self.buffer, self.length)
    }

    pub fn intersects_with(&self, other: &Self) -> bool {
        intersects_with_mut(self, other)
    }

    pub fn freeze(self) -> Bitmap {
        self.into()
    }
}

impl From<MutableBitmap> for Bitmap {
    #[inline]
    fn from(buffer: MutableBitmap) -> Self {
        Bitmap::try_new(buffer.buffer, buffer.length).unwrap()
    }
}

impl From<MutableBitmap> for Option<Bitmap> {
    #[inline]
    fn from(buffer: MutableBitmap) -> Self {
        let unset_bits = buffer.unset_bits();
        if unset_bits > 0 {
            // SAFETY: invariants of the `MutableBitmap` equal that of `Bitmap`.
            let bitmap = unsafe {
                Bitmap::from_inner_unchecked(
                    Arc::new(buffer.buffer.into()),
                    0,
                    buffer.length,
                    Some(unset_bits),
                )
            };
            Some(bitmap)
        } else {
            None
        }
    }
}

impl<P: AsRef<[bool]>> From<P> for MutableBitmap {
    #[inline]
    fn from(slice: P) -> Self {
        MutableBitmap::from_trusted_len_iter(slice.as_ref().iter().copied())
    }
}

impl FromIterator<bool> for MutableBitmap {
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = bool>,
    {
        let mut iterator = iter.into_iter();
        let mut buffer = {
            let byte_capacity: usize = iterator.size_hint().0.saturating_add(7) / 8;
            Vec::with_capacity(byte_capacity)
        };

        let mut length = 0;

        loop {
            let mut exhausted = false;
            let mut byte_accum: u8 = 0;
            let mut mask: u8 = 1;

            //collect (up to) 8 bits into a byte
            while mask != 0 {
                if let Some(value) = iterator.next() {
                    length += 1;
                    byte_accum |= match value {
                        true => mask,
                        false => 0,
                    };
                    mask <<= 1;
                } else {
                    exhausted = true;
                    break;
                }
            }

            // break if the iterator was exhausted before it provided a bool for this byte
            if exhausted && mask == 1 {
                break;
            }

            //ensure we have capacity to write the byte
            if buffer.len() == buffer.capacity() {
                //no capacity for new byte, allocate 1 byte more (plus however many more the iterator advertises)
                let additional_byte_capacity = 1usize.saturating_add(
                    iterator.size_hint().0.saturating_add(7) / 8, //convert bit count to byte count, rounding up
                );
                buffer.reserve(additional_byte_capacity)
            }

            // Soundness: capacity was allocated above
            buffer.push(byte_accum);
            if exhausted {
                break;
            }
        }
        Self { buffer, length }
    }
}

// [7, 6, 5, 4, 3, 2, 1, 0], [15, 14, 13, 12, 11, 10, 9, 8]
// [00000001_00000000_00000000_00000000_...] // u64
/// # Safety
/// The iterator must be trustedLen and its len must be least `len`.
#[inline]
unsafe fn get_chunk_unchecked(iterator: &mut impl Iterator<Item = bool>) -> u64 {
    let mut byte = 0u64;
    let mut mask;
    for i in 0..8 {
        mask = 1u64 << (8 * i);
        for _ in 0..8 {
            let value = match iterator.next() {
                Some(value) => value,
                None => unsafe { unreachable_unchecked() },
            };

            byte |= match value {
                true => mask,
                false => 0,
            };
            mask <<= 1;
        }
    }
    byte
}

/// # Safety
/// The iterator must be trustedLen and its len must be least `len`.
#[inline]
unsafe fn get_byte_unchecked(len: usize, iterator: &mut impl Iterator<Item = bool>) -> u8 {
    let mut byte_accum: u8 = 0;
    let mut mask: u8 = 1;
    for _ in 0..len {
        let value = match iterator.next() {
            Some(value) => value,
            None => unsafe { unreachable_unchecked() },
        };

        byte_accum |= match value {
            true => mask,
            false => 0,
        };
        mask <<= 1;
    }
    byte_accum
}

/// Extends the [`Vec<u8>`] from `iterator`
/// # Safety
/// The iterator MUST be [`TrustedLen`].
#[inline]
unsafe fn extend_aligned_trusted_iter_unchecked(
    buffer: &mut Vec<u8>,
    mut iterator: impl Iterator<Item = bool>,
) -> usize {
    let additional_bits = iterator.size_hint().1.unwrap();
    let chunks = additional_bits / 64;
    let remainder = additional_bits % 64;

    let additional = (additional_bits + 7) / 8;
    assert_eq!(
        additional,
        // a hint of how the following calculation will be done
        chunks * 8 + remainder / 8 + (remainder % 8 > 0) as usize
    );
    buffer.reserve(additional);

    // chunks of 64 bits
    for _ in 0..chunks {
        let chunk = get_chunk_unchecked(&mut iterator);
        buffer.extend_from_slice(&chunk.to_le_bytes());
    }

    // remaining complete bytes
    for _ in 0..(remainder / 8) {
        let byte = unsafe { get_byte_unchecked(8, &mut iterator) };
        buffer.push(byte)
    }

    // remaining bits
    let remainder = remainder % 8;
    if remainder > 0 {
        let byte = unsafe { get_byte_unchecked(remainder, &mut iterator) };
        buffer.push(byte)
    }
    additional_bits
}

impl MutableBitmap {
    /// Extends `self` from a [`TrustedLen`] iterator.
    #[inline]
    pub fn extend_from_trusted_len_iter<I: TrustedLen<Item = bool>>(&mut self, iterator: I) {
        // SAFETY: I: TrustedLen
        unsafe { self.extend_from_trusted_len_iter_unchecked(iterator) }
    }

    /// Extends `self` from an iterator of trusted len.
    ///
    /// # Safety
    /// The caller must guarantee that the iterator has a trusted len.
    #[inline]
    pub unsafe fn extend_from_trusted_len_iter_unchecked<I: Iterator<Item = bool>>(
        &mut self,
        mut iterator: I,
    ) {
        // the length of the iterator throughout this function.
        let mut length = iterator.size_hint().1.unwrap();

        let bit_offset = self.length % 8;

        if length < 8 - bit_offset {
            if bit_offset == 0 {
                self.buffer.push(0);
            }
            // the iterator will not fill the last byte
            let byte = self.buffer.as_mut_slice().last_mut().unwrap();
            let mut i = bit_offset;
            for value in iterator {
                *byte = set(*byte, i, value);
                i += 1;
            }
            self.length += length;
            return;
        }

        // at this point we know that length will hit a byte boundary and thus
        // increase the buffer.

        if bit_offset != 0 {
            // we are in the middle of a byte; lets finish it
            let byte = self.buffer.as_mut_slice().last_mut().unwrap();
            (bit_offset..8).for_each(|i| {
                *byte = set(*byte, i, iterator.next().unwrap());
            });
            self.length += 8 - bit_offset;
            length -= 8 - bit_offset;
        }

        // everything is aligned; proceed with the bulk operation
        debug_assert_eq!(self.length % 8, 0);

        unsafe { extend_aligned_trusted_iter_unchecked(&mut self.buffer, iterator) };
        self.length += length;
    }

    /// Creates a new [`MutableBitmap`] from an iterator of booleans.
    ///
    /// # Safety
    /// The iterator must report an accurate length.
    #[inline]
    pub unsafe fn from_trusted_len_iter_unchecked<I>(iterator: I) -> Self
    where
        I: Iterator<Item = bool>,
    {
        let mut buffer = Vec::<u8>::new();

        let length = extend_aligned_trusted_iter_unchecked(&mut buffer, iterator);

        Self { buffer, length }
    }

    /// Creates a new [`MutableBitmap`] from an iterator of booleans.
    #[inline]
    pub fn from_trusted_len_iter<I>(iterator: I) -> Self
    where
        I: TrustedLen<Item = bool>,
    {
        // SAFETY: Iterator is `TrustedLen`
        unsafe { Self::from_trusted_len_iter_unchecked(iterator) }
    }

    /// Creates a new [`MutableBitmap`] from an iterator of booleans.
    pub fn try_from_trusted_len_iter<E, I>(iterator: I) -> std::result::Result<Self, E>
    where
        I: TrustedLen<Item = std::result::Result<bool, E>>,
    {
        unsafe { Self::try_from_trusted_len_iter_unchecked(iterator) }
    }

    /// Creates a new [`MutableBitmap`] from an falible iterator of booleans.
    ///
    /// # Safety
    /// The caller must guarantee that the iterator is `TrustedLen`.
    pub unsafe fn try_from_trusted_len_iter_unchecked<E, I>(
        mut iterator: I,
    ) -> std::result::Result<Self, E>
    where
        I: Iterator<Item = std::result::Result<bool, E>>,
    {
        let length = iterator.size_hint().1.unwrap();

        let mut buffer = vec![0u8; (length + 7) / 8];

        let chunks = length / 8;
        let reminder = length % 8;

        let data = buffer.as_mut_slice();
        data[..chunks].iter_mut().try_for_each(|byte| {
            (0..8).try_for_each(|i| {
                *byte = set(*byte, i, iterator.next().unwrap()?);
                Ok(())
            })
        })?;

        if reminder != 0 {
            let last = &mut data[chunks];
            iterator.enumerate().try_for_each(|(i, value)| {
                *last = set(*last, i, value?);
                Ok(())
            })?;
        }

        Ok(Self { buffer, length })
    }

    fn extend_unaligned(&mut self, slice: &[u8], offset: usize, length: usize) {
        // e.g.
        // [a, b, --101010]     <- to be extended
        // [00111111, 11010101] <- to extend
        // [a, b, 11101010, --001111] expected result

        let aligned_offset = offset / 8;
        let own_offset = self.length % 8;
        debug_assert_eq!(offset % 8, 0); // assumed invariant
        debug_assert!(own_offset != 0); // assumed invariant

        let bytes_len = length.saturating_add(7) / 8;
        let items = &slice[aligned_offset..aligned_offset + bytes_len];
        // self has some offset => we need to shift all `items`, and merge the first
        let buffer = self.buffer.as_mut_slice();
        let last = &mut buffer[buffer.len() - 1];

        // --101010 | 00111111 << 6 = 11101010
        // erase previous
        *last &= 0b11111111u8 >> (8 - own_offset); // unset before setting
        *last |= items[0] << own_offset;

        if length + own_offset <= 8 {
            // no new bytes needed
            self.length += length;
            return;
        }
        let additional = length - (8 - own_offset);

        let remaining = [items[items.len() - 1], 0];
        let bytes = items
            .windows(2)
            .chain(std::iter::once(remaining.as_ref()))
            .map(|w| merge_reversed(w[0], w[1], 8 - own_offset))
            .take(additional.saturating_add(7) / 8);
        self.buffer.extend(bytes);

        self.length += length;
    }

    fn extend_aligned(&mut self, slice: &[u8], offset: usize, length: usize) {
        let aligned_offset = offset / 8;
        let bytes_len = length.saturating_add(7) / 8;
        let items = &slice[aligned_offset..aligned_offset + bytes_len];
        self.buffer.extend_from_slice(items);
        self.length += length;
    }

    /// Extends the [`MutableBitmap`] from a slice of bytes with optional offset.
    /// This is the fastest way to extend a [`MutableBitmap`].
    /// # Implementation
    /// When both [`MutableBitmap`]'s length and `offset` are both multiples of 8,
    /// this function performs a memcopy. Else, it first aligns bit by bit and then performs a memcopy.
    ///
    /// # Safety
    /// Caller must ensure `offset + length <= slice.len() * 8`
    #[inline]
    pub unsafe fn extend_from_slice_unchecked(
        &mut self,
        slice: &[u8],
        offset: usize,
        length: usize,
    ) {
        if length == 0 {
            return;
        };
        let is_aligned = self.length % 8 == 0;
        let other_is_aligned = offset % 8 == 0;
        match (is_aligned, other_is_aligned) {
            (true, true) => self.extend_aligned(slice, offset, length),
            (false, true) => self.extend_unaligned(slice, offset, length),
            // todo: further optimize the other branches.
            _ => self.extend_from_trusted_len_iter(BitmapIter::new(slice, offset, length)),
        }
        // internal invariant:
        debug_assert_eq!(self.length.saturating_add(7) / 8, self.buffer.len());
    }

    /// Extends the [`MutableBitmap`] from a slice of bytes with optional offset.
    /// This is the fastest way to extend a [`MutableBitmap`].
    /// # Implementation
    /// When both [`MutableBitmap`]'s length and `offset` are both multiples of 8,
    /// this function performs a memcopy. Else, it first aligns bit by bit and then performs a memcopy.
    #[inline]
    pub fn extend_from_slice(&mut self, slice: &[u8], offset: usize, length: usize) {
        assert!(offset + length <= slice.len() * 8);
        // SAFETY: invariant is asserted
        unsafe { self.extend_from_slice_unchecked(slice, offset, length) }
    }

    /// Extends the [`MutableBitmap`] from a [`Bitmap`].
    #[inline]
    pub fn extend_from_bitmap(&mut self, bitmap: &Bitmap) {
        let (slice, offset, length) = bitmap.as_slice();
        // SAFETY: bitmap.as_slice adheres to the invariant
        unsafe {
            self.extend_from_slice_unchecked(slice, offset, length);
        }
    }

    /// Returns the slice of bytes of this [`MutableBitmap`].
    /// Note that the last byte may not be fully used.
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        let len = (self.length).saturating_add(7) / 8;
        &self.buffer[..len]
    }

    /// Returns the slice of bytes of this [`MutableBitmap`].
    /// Note that the last byte may not be fully used.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        let len = (self.length).saturating_add(7) / 8;
        &mut self.buffer[..len]
    }
}

impl Default for MutableBitmap {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> IntoIterator for &'a MutableBitmap {
    type Item = bool;
    type IntoIter = BitmapIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        BitmapIter::<'a>::new(&self.buffer, 0, self.length)
    }
}
