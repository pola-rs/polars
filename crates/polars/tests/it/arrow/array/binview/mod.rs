use std::sync::Arc;

use arrow::array::*;
use arrow::buffer::Buffer;
use arrow::datatypes::ArrowDataType;

fn array() -> BinaryViewArrayGeneric<str> {
    let datatype = ArrowDataType::Utf8View;

    let hello = View::new_from_bytes(b"hello", 0, 0);
    let there = View::new_from_bytes(b"there", 0, 6);
    let bye = View::new_from_bytes(b"bye", 1, 0);
    let excl = View::new_from_bytes(b"!!!", 1, 3);
    let hello_there = View::new_from_bytes(b"hello there", 1, 0);

    let views = Buffer::from(vec![hello, there, bye, excl, hello_there]);
    let buffers = Arc::new([
        Buffer::from(b"hello there".to_vec()),
        Buffer::from(b"bye!!!".to_vec()),
    ]);
    let validity = None;

    BinaryViewArrayGeneric::try_new(datatype, views, buffers, validity).unwrap()
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
