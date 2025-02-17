use arrow::bitmap::utils::BitChunks;
use arrow::types::BitChunkIter;

#[test]
fn basics() {
    let mut iter = BitChunks::<u16>::new(&[0b00000001u8, 0b00000010u8], 0, 16);
    assert_eq!(iter.next().unwrap(), 0b0000_0010_0000_0001u16);
    assert_eq!(iter.remainder(), 0);
}

#[test]
fn remainder() {
    let a = BitChunks::<u16>::new(&[0b00000001u8, 0b00000010u8, 0b00000100u8], 0, 18);
    assert_eq!(a.remainder(), 0b00000100u16);
}

#[test]
fn remainder_saturating() {
    let a = BitChunks::<u16>::new(&[0b00000001u8, 0b00000010u8, 0b00000010u8], 0, 18);
    assert_eq!(a.remainder(), 0b0000_0000_0000_0010u16);
}

#[test]
fn basics_offset() {
    let mut iter = BitChunks::<u16>::new(&[0b00000001u8, 0b00000011u8, 0b00000001u8], 1, 16);
    assert_eq!(iter.remainder(), 0);
    assert_eq!(iter.next().unwrap(), 0b1000_0001_1000_0000u16);
    assert_eq!(iter.next(), None);
}

#[test]
fn basics_offset_remainder() {
    let mut a = BitChunks::<u16>::new(&[0b00000001u8, 0b00000011u8, 0b10000001u8], 1, 15);
    assert_eq!(a.next(), None);
    assert_eq!(a.remainder(), 0b1000_0001_1000_0000u16);
    assert_eq!(a.remainder_len(), 15);
}

#[test]
fn offset_remainder_saturating() {
    let a = BitChunks::<u16>::new(&[0b00000001u8, 0b00000011u8, 0b00000011u8], 1, 17);
    assert_eq!(a.remainder(), 0b0000_0000_0000_0001u16);
}

#[test]
fn offset_remainder_saturating2() {
    let a = BitChunks::<u64>::new(&[0b01001001u8, 0b00000001], 1, 8);
    assert_eq!(a.remainder(), 0b1010_0100u64);
}

#[test]
fn offset_remainder_saturating3() {
    let input: &[u8] = &[0b01000000, 0b01000001];
    let a = BitChunks::<u64>::new(input, 8, 2);
    assert_eq!(a.remainder(), 0b0100_0001u64);
}

#[test]
fn basics_multiple() {
    let mut iter = BitChunks::<u16>::new(
        &[0b00000001u8, 0b00000010u8, 0b00000100u8, 0b00001000u8],
        0,
        4 * 8,
    );
    assert_eq!(iter.next().unwrap(), 0b0000_0010_0000_0001u16);
    assert_eq!(iter.next().unwrap(), 0b0000_1000_0000_0100u16);
    assert_eq!(iter.remainder(), 0);
}

#[test]
fn basics_multiple_offset() {
    let mut iter = BitChunks::<u16>::new(
        &[
            0b00000001u8,
            0b00000010u8,
            0b00000100u8,
            0b00001000u8,
            0b00000001u8,
        ],
        1,
        4 * 8,
    );
    assert_eq!(iter.next().unwrap(), 0b0000_0001_0000_0000u16);
    assert_eq!(iter.next().unwrap(), 0b1000_0100_0000_0010u16);
    assert_eq!(iter.remainder(), 0);
}

#[test]
fn remainder_large() {
    let input: &[u8] = &[
        0b00100100, 0b01001001, 0b10010010, 0b00100100, 0b01001001, 0b10010010, 0b00100100,
        0b01001001, 0b10010010, 0b00100100, 0b01001001, 0b10010010, 0b00000100,
    ];
    let mut iter = BitChunks::<u8>::new(input, 0, 8 * 12 + 4);
    assert_eq!(iter.remainder_len(), 100 - 96);

    for j in 0..12 {
        let mut a = BitChunkIter::new(iter.next().unwrap(), 8);
        for i in 0..8 {
            assert_eq!(a.next().unwrap(), (j * 8 + i + 1) % 3 == 0);
        }
    }
    assert_eq!(None, iter.next());

    let expected_remainder = 0b00000100u8;
    assert_eq!(iter.remainder(), expected_remainder);

    let mut a = BitChunkIter::new(expected_remainder, 8);
    for i in 0..4 {
        assert_eq!(a.next().unwrap(), (i + 1) % 3 == 0);
    }
}

#[test]
fn basics_1() {
    let mut iter = BitChunks::<u16>::new(
        &[0b00000001u8, 0b00000010u8, 0b00000100u8, 0b00001000u8],
        8,
        3 * 8,
    );
    assert_eq!(iter.next().unwrap(), 0b0000_0100_0000_0010u16);
    assert_eq!(iter.next(), None);
    assert_eq!(iter.remainder(), 0b0000_0000_0000_1000u16);
    assert_eq!(iter.remainder_len(), 8);
}

#[test]
fn basics_2() {
    let mut iter = BitChunks::<u16>::new(
        &[0b00000001u8, 0b00000010u8, 0b00000100u8, 0b00001000u8],
        7,
        3 * 8,
    );
    assert_eq!(iter.remainder(), 0b0000_0000_0001_0000u16);
    assert_eq!(iter.next().unwrap(), 0b0000_1000_0000_0100u16);
    assert_eq!(iter.next(), None);
}

#[test]
fn remainder_1() {
    let mut iter = BitChunks::<u64>::new(&[0b11111111u8, 0b00000001u8], 0, 9);
    assert_eq!(iter.next(), None);
    assert_eq!(iter.remainder(), 0b1_1111_1111u64);
}

#[test]
fn remainder_2() {
    // (i % 3 == 0) in bitmap
    let input: &[u8] = &[
        0b01001001, 0b10010010, 0b00100100, 0b01001001, 0b10010010, 0b00100100, 0b01001001,
        0b10010010, 0b00100100, 0b01001001, /* 73 */
        0b10010010, /* 146 */
        0b00100100, 0b00001001,
    ];
    let offset = 10; // 8 + 2
    let length = 90;

    let mut iter = BitChunks::<u64>::new(input, offset, length);
    let first: u64 = 0b0100100100100100100100100100100100100100100100100100100100100100;
    assert_eq!(first, iter.next().unwrap());
    assert_eq!(iter.next(), None);
    assert_eq!(iter.remainder(), 0b10010010010010010010010010u64);
}
