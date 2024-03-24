use arrow::bitmap::{Bitmap, MutableBitmap};

#[test]
fn from_slice() {
    let slice = &[true, false, true];
    let a = MutableBitmap::from(slice);
    assert_eq!(a.iter().collect::<Vec<_>>(), slice);
}

#[test]
fn from_len_zeroed() {
    let a = MutableBitmap::from_len_zeroed(10);
    assert_eq!(a.len(), 10);
    assert_eq!(a.unset_bits(), 10);
}

#[test]
fn from_len_set() {
    let a = MutableBitmap::from_len_set(10);
    assert_eq!(a.len(), 10);
    assert_eq!(a.unset_bits(), 0);
}

#[test]
fn try_new_invalid() {
    assert!(MutableBitmap::try_new(vec![], 2).is_err());
}

#[test]
fn clear() {
    let mut a = MutableBitmap::from_len_zeroed(10);
    a.clear();
    assert_eq!(a.len(), 0);
}

#[test]
fn trusted_len() {
    let data = vec![true; 65];
    let bitmap = MutableBitmap::from_trusted_len_iter(data.into_iter());
    let bitmap: Bitmap = bitmap.into();
    assert_eq!(bitmap.len(), 65);

    assert_eq!(bitmap.as_slice().0[8], 0b00000001);
}

#[test]
fn trusted_len_small() {
    let data = vec![true; 7];
    let bitmap = MutableBitmap::from_trusted_len_iter(data.into_iter());
    let bitmap: Bitmap = bitmap.into();
    assert_eq!(bitmap.len(), 7);

    assert_eq!(bitmap.as_slice().0[0], 0b01111111);
}

#[test]
fn push() {
    let mut bitmap = MutableBitmap::new();
    bitmap.push(true);
    bitmap.push(false);
    bitmap.push(false);
    for _ in 0..7 {
        bitmap.push(true)
    }
    let bitmap: Bitmap = bitmap.into();
    assert_eq!(bitmap.len(), 10);

    assert_eq!(bitmap.as_slice().0, &[0b11111001, 0b00000011]);
}

#[test]
fn push_small() {
    let mut bitmap = MutableBitmap::new();
    bitmap.push(true);
    bitmap.push(true);
    bitmap.push(false);
    let bitmap: Option<Bitmap> = bitmap.into();
    let bitmap = bitmap.unwrap();
    assert_eq!(bitmap.len(), 3);
    assert_eq!(bitmap.as_slice().0[0], 0b00000011);
}

#[test]
fn push_exact_zeros() {
    let mut bitmap = MutableBitmap::new();
    for _ in 0..8 {
        bitmap.push(false)
    }
    let bitmap: Option<Bitmap> = bitmap.into();
    let bitmap = bitmap.unwrap();
    assert_eq!(bitmap.len(), 8);
    assert_eq!(bitmap.as_slice().0.len(), 1);
}

#[test]
fn push_exact_ones() {
    let mut bitmap = MutableBitmap::new();
    for _ in 0..8 {
        bitmap.push(true)
    }
    let bitmap: Option<Bitmap> = bitmap.into();
    assert!(bitmap.is_none());
}

#[test]
fn pop() {
    let mut bitmap = MutableBitmap::new();
    bitmap.push(false);
    bitmap.push(true);
    bitmap.push(false);
    bitmap.push(true);

    assert_eq!(bitmap.pop(), Some(true));
    assert_eq!(bitmap.len(), 3);

    assert_eq!(bitmap.pop(), Some(false));
    assert_eq!(bitmap.len(), 2);

    let bitmap: Bitmap = bitmap.into();
    assert_eq!(bitmap.len(), 2);
    assert_eq!(bitmap.as_slice().0[0], 0b00001010);
}

#[test]
fn pop_large() {
    let mut bitmap = MutableBitmap::new();
    for _ in 0..8 {
        bitmap.push(true);
    }

    bitmap.push(false);
    bitmap.push(true);
    bitmap.push(false);

    assert_eq!(bitmap.pop(), Some(false));
    assert_eq!(bitmap.len(), 10);

    assert_eq!(bitmap.pop(), Some(true));
    assert_eq!(bitmap.len(), 9);

    assert_eq!(bitmap.pop(), Some(false));
    assert_eq!(bitmap.len(), 8);

    let bitmap: Bitmap = bitmap.into();
    assert_eq!(bitmap.len(), 8);
    assert_eq!(bitmap.as_slice().0, &[0b11111111]);
}

#[test]
fn pop_all() {
    let mut bitmap = MutableBitmap::new();
    bitmap.push(false);
    bitmap.push(true);
    bitmap.push(true);
    bitmap.push(true);

    assert_eq!(bitmap.pop(), Some(true));
    assert_eq!(bitmap.len(), 3);
    assert_eq!(bitmap.pop(), Some(true));
    assert_eq!(bitmap.len(), 2);
    assert_eq!(bitmap.pop(), Some(true));
    assert_eq!(bitmap.len(), 1);
    assert_eq!(bitmap.pop(), Some(false));
    assert_eq!(bitmap.len(), 0);
    assert_eq!(bitmap.pop(), None);
    assert_eq!(bitmap.len(), 0);
}

#[test]
fn capacity() {
    let b = MutableBitmap::with_capacity(10);
    assert!(b.capacity() >= 10);
}

#[test]
fn capacity_push() {
    let mut b = MutableBitmap::with_capacity(512);
    (0..512).for_each(|_| b.push(true));
    assert_eq!(b.capacity(), 512);
    b.reserve(8);
    assert_eq!(b.capacity(), 1024);
}

#[test]
fn extend() {
    let mut b = MutableBitmap::new();

    let iter = (0..512).map(|i| i % 6 == 0);
    unsafe { b.extend_from_trusted_len_iter_unchecked(iter) };
    let b: Bitmap = b.into();
    for (i, v) in b.iter().enumerate() {
        assert_eq!(i % 6 == 0, v);
    }
}

#[test]
fn extend_offset() {
    let mut b = MutableBitmap::new();
    b.push(true);

    let iter = (0..512).map(|i| i % 6 == 0);
    unsafe { b.extend_from_trusted_len_iter_unchecked(iter) };
    let b: Bitmap = b.into();
    let mut iter = b.iter().enumerate();
    assert!(iter.next().unwrap().1);
    for (i, v) in iter {
        assert_eq!((i - 1) % 6 == 0, v);
    }
}

#[test]
fn set() {
    let mut bitmap = MutableBitmap::from_len_zeroed(12);
    bitmap.set(0, true);
    assert!(bitmap.get(0));
    bitmap.set(0, false);
    assert!(!bitmap.get(0));

    bitmap.set(11, true);
    assert!(bitmap.get(11));
    bitmap.set(11, false);
    assert!(!bitmap.get(11));
    bitmap.set(11, true);

    let bitmap: Option<Bitmap> = bitmap.into();
    let bitmap = bitmap.unwrap();
    assert_eq!(bitmap.len(), 12);
    assert_eq!(bitmap.as_slice().0[0], 0b00000000);
}

#[test]
fn extend_from_bitmap() {
    let other = Bitmap::from(&[true, false, true]);
    let mut bitmap = MutableBitmap::new();

    // call is optimized to perform a memcopy
    bitmap.extend_from_bitmap(&other);

    assert_eq!(bitmap.len(), 3);
    assert_eq!(bitmap.as_slice()[0], 0b00000101);

    // this call iterates over all bits
    bitmap.extend_from_bitmap(&other);

    assert_eq!(bitmap.len(), 6);
    assert_eq!(bitmap.as_slice()[0], 0b00101101);
}

#[test]
fn extend_from_bitmap_offset() {
    let other = Bitmap::from_u8_slice([0b00111111], 8);
    let mut bitmap = MutableBitmap::from_vec(vec![1, 0, 0b00101010], 22);

    // call is optimized to perform a memcopy
    bitmap.extend_from_bitmap(&other);

    assert_eq!(bitmap.len(), 22 + 8);
    assert_eq!(bitmap.as_slice(), &[1, 0, 0b11101010, 0b00001111]);

    // more than one byte
    let other = Bitmap::from_u8_slice([0b00111111, 0b00001111, 0b0001100], 20);
    let mut bitmap = MutableBitmap::from_vec(vec![1, 0, 0b00101010], 22);

    // call is optimized to perform a memcopy
    bitmap.extend_from_bitmap(&other);

    assert_eq!(bitmap.len(), 22 + 20);
    assert_eq!(
        bitmap.as_slice(),
        &[1, 0, 0b11101010, 0b11001111, 0b0000011, 0b0000011]
    );
}

#[test]
fn debug() {
    let mut b = MutableBitmap::new();
    assert_eq!(format!("{b:?}"), "Bitmap { len: 0, offset: 0, bytes: [] }");
    b.push(true);
    b.push(false);
    assert_eq!(
        format!("{b:?}"),
        "Bitmap { len: 2, offset: 0, bytes: [0b______01] }"
    );
    b.push(false);
    b.push(false);
    b.push(false);
    b.push(false);
    b.push(true);
    b.push(true);
    assert_eq!(
        format!("{b:?}"),
        "Bitmap { len: 8, offset: 0, bytes: [0b11000001] }"
    );
    b.push(true);
    assert_eq!(
        format!("{b:?}"),
        "Bitmap { len: 9, offset: 0, bytes: [0b11000001, 0b_______1] }"
    );
}

#[test]
fn extend_set() {
    let mut b = MutableBitmap::new();
    b.extend_constant(6, true);
    assert_eq!(b.as_slice(), &[0b11111111]);
    assert_eq!(b.len(), 6);

    let mut b = MutableBitmap::from(&[false]);
    b.extend_constant(6, true);
    assert_eq!(b.as_slice(), &[0b01111110]);
    assert_eq!(b.len(), 1 + 6);

    let mut b = MutableBitmap::from(&[false]);
    b.extend_constant(9, true);
    assert_eq!(b.as_slice(), &[0b11111110, 0b11111111]);
    assert_eq!(b.len(), 1 + 9);

    let mut b = MutableBitmap::from(&[false, false, false, false]);
    b.extend_constant(2, true);
    assert_eq!(b.as_slice(), &[0b00110000]);
    assert_eq!(b.len(), 4 + 2);

    let mut b = MutableBitmap::from(&[false, false, false, false]);
    b.extend_constant(8, true);
    assert_eq!(b.as_slice(), &[0b11110000, 0b11111111]);
    assert_eq!(b.len(), 4 + 8);

    let mut b = MutableBitmap::from(&[true, true]);
    b.extend_constant(3, true);
    assert_eq!(b.as_slice(), &[0b00011111]);
    assert_eq!(b.len(), 2 + 3);
}

#[test]
fn extend_unset() {
    let mut b = MutableBitmap::new();
    b.extend_constant(6, false);
    assert_eq!(b.as_slice(), &[0b0000000]);
    assert_eq!(b.len(), 6);

    let mut b = MutableBitmap::from(&[true]);
    b.extend_constant(6, false);
    assert_eq!(b.as_slice(), &[0b00000001]);
    assert_eq!(b.len(), 1 + 6);

    let mut b = MutableBitmap::from(&[true]);
    b.extend_constant(9, false);
    assert_eq!(b.as_slice(), &[0b0000001, 0b00000000]);
    assert_eq!(b.len(), 1 + 9);

    let mut b = MutableBitmap::from(&[true, true, true, true]);
    b.extend_constant(2, false);
    assert_eq!(b.as_slice(), &[0b00001111]);
    assert_eq!(b.len(), 4 + 2);
}

#[test]
fn extend_bitmap() {
    let mut b = MutableBitmap::from(&[true]);
    b.extend_from_slice(&[0b00011001], 0, 6);
    assert_eq!(b.as_slice(), &[0b00110011]);
    assert_eq!(b.len(), 1 + 6);

    let mut b = MutableBitmap::from(&[true]);
    b.extend_from_slice(&[0b00011001, 0b00011001], 0, 9);
    assert_eq!(b.as_slice(), &[0b00110011, 0b00110010]);
    assert_eq!(b.len(), 1 + 9);

    let mut b = MutableBitmap::from(&[true, true, true, true]);
    b.extend_from_slice(&[0b00011001, 0b00011001], 0, 9);
    assert_eq!(b.as_slice(), &[0b10011111, 0b10010001]);
    assert_eq!(b.len(), 4 + 9);

    let mut b = MutableBitmap::from(&[true, true, true, true, true]);
    b.extend_from_slice(&[0b00001011], 0, 4);
    assert_eq!(b.as_slice(), &[0b01111111, 0b00000001]);
    assert_eq!(b.len(), 5 + 4);
}

// TODO! undo miri ignore once issue is fixed in miri
// this test was a memory hog and lead to OOM in CI
// given enough memory it was able to pass successfully on a local
#[test]
#[cfg_attr(miri, ignore)]
fn extend_constant1() {
    use std::iter::FromIterator;
    for i in 0..64 {
        for j in 0..64 {
            let mut b = MutableBitmap::new();
            b.extend_constant(i, false);
            b.extend_constant(j, true);
            assert_eq!(
                b,
                MutableBitmap::from_iter(
                    std::iter::repeat(false)
                        .take(i)
                        .chain(std::iter::repeat(true).take(j))
                )
            );

            let mut b = MutableBitmap::new();
            b.extend_constant(i, true);
            b.extend_constant(j, false);
            assert_eq!(
                b,
                MutableBitmap::from_iter(
                    std::iter::repeat(true)
                        .take(i)
                        .chain(std::iter::repeat(false).take(j))
                )
            );
        }
    }
}

#[test]
fn extend_bitmap_one() {
    for offset in 0..7 {
        let mut b = MutableBitmap::new();
        for _ in 0..4 {
            b.extend_from_slice(&[!0], offset, 1);
            b.extend_from_slice(&[!0], offset, 1);
        }
        assert_eq!(b.as_slice(), &[0b11111111]);
    }
}

#[test]
fn extend_bitmap_other() {
    let mut a = MutableBitmap::from([true, true, true, false, true, true, true, false, true, true]);
    a.extend_from_slice(&[0b01111110u8, 0b10111111, 0b11011111, 0b00000111], 20, 2);
    assert_eq!(
        a,
        MutableBitmap::from([
            true, true, true, false, true, true, true, false, true, true, true, false
        ])
    );
}

#[test]
fn shrink_to_fit() {
    let mut a = MutableBitmap::with_capacity(1025);
    a.push(false);
    a.shrink_to_fit();
    assert!(a.capacity() < 1025);
}
