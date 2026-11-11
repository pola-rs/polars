use polars_arrow::array::*;
use polars_arrow::datatypes::ArrowDataType;
use polars_buffer::Buffer;

fn array() -> BinaryViewArrayGeneric<str> {
    let datatype = ArrowDataType::Utf8View;

    let hello = View::new_from_bytes(b"hello", 0, 0);
    let there = View::new_from_bytes(b"there", 0, 6);
    let bye = View::new_from_bytes(b"bye", 1, 0);
    let excl = View::new_from_bytes(b"!!!", 1, 3);
    let hello_there = View::new_from_bytes(b"hello there", 1, 0);

    let views = Buffer::from(vec![hello, there, bye, excl, hello_there]);
    let buffers = Buffer::from(vec![
        Buffer::from(b"hello there".to_vec()),
        Buffer::from(b"bye!!!".to_vec()),
    ]);
    let validity = None;

    BinaryViewArrayGeneric::try_new(datatype, views, buffers, validity).unwrap()
}

#[test]
fn inline_view_from_block_matches_inline_view_of_its_prefix() {
    // The view of the first `len` bytes of a block holds them, and only them, in memory order.
    let block: [u8; 16] = std::array::from_fn(|i| i as u8 + 1);
    for len in 0..=View::MAX_INLINE_SIZE as usize {
        let view = View::new_inline_from_block(block, len);
        assert_eq!(view, View::new_inline(&block[..len]));
        assert_eq!(view.length as usize, len);
        assert_eq!(view.get_inlined_slice(), Some(&block[..len]));
    }
}

#[test]
fn split_at() {
    let (lhs, rhs) = array().split_at(2);

    assert_eq!(lhs.value(0), "hello");
    assert_eq!(lhs.value(1), "there");
    assert_eq!(rhs.value(0), "bye");
    assert_eq!(rhs.value(1), "!!!");
    assert_eq!(rhs.value(2), "hello there");
}
