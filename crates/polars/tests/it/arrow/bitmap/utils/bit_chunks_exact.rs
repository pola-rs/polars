use arrow::bitmap::utils::BitChunksExact;

#[test]
fn basics() {
    let mut iter = BitChunksExact::<u8>::new(&[0b11111111u8, 0b00000001u8], 9);
    assert_eq!(iter.next().unwrap(), 0b11111111u8);
    assert_eq!(iter.remainder(), 0b00000001u8);
}

#[test]
fn basics_u16_small() {
    let mut iter = BitChunksExact::<u16>::new(&[0b11111111u8], 7);
    assert_eq!(iter.next(), None);
    assert_eq!(iter.remainder(), 0b0000_0000_1111_1111u16);
}

#[test]
fn basics_u16() {
    let mut iter = BitChunksExact::<u16>::new(&[0b11111111u8, 0b00000001u8], 9);
    assert_eq!(iter.next(), None);
    assert_eq!(iter.remainder(), 0b0000_0001_1111_1111u16);
}

#[test]
fn remainder_u16() {
    let mut iter = BitChunksExact::<u16>::new(
        &[0b11111111u8, 0b00000001u8, 0b00000001u8, 0b11011011u8],
        23,
    );
    assert_eq!(iter.next(), Some(511));
    assert_eq!(iter.next(), None);
    assert_eq!(iter.remainder(), 1u16);
}
