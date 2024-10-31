use arrow::buffer::Buffer;

#[test]
fn new() {
    let buffer = Buffer::<i32>::new();
    assert_eq!(buffer.len(), 0);
    assert!(buffer.is_empty());
}

#[test]
fn from_slice() {
    let buffer = Buffer::<i32>::from(vec![0, 1, 2]);
    assert_eq!(buffer.len(), 3);
    assert_eq!(buffer.as_slice(), &[0, 1, 2]);
}

#[test]
fn slice() {
    let buffer = Buffer::<i32>::from(vec![0, 1, 2, 3]);
    let buffer = buffer.sliced(1, 2);
    assert_eq!(buffer.len(), 2);
    assert_eq!(buffer.as_slice(), &[1, 2]);
}

#[test]
fn from_iter() {
    let buffer = (0..3).collect::<Buffer<i32>>();
    assert_eq!(buffer.len(), 3);
    assert_eq!(buffer.as_slice(), &[0, 1, 2]);
}

#[test]
fn debug() {
    let buffer = Buffer::<i32>::from(vec![0, 1, 2, 3]);
    let buffer = buffer.sliced(1, 2);
    let a = format!("{buffer:?}");
    assert_eq!(a, "[1, 2]")
}

#[test]
fn from_vec() {
    let buffer = Buffer::<i32>::from(vec![0, 1, 2]);
    assert_eq!(buffer.len(), 3);
    assert_eq!(buffer.as_slice(), &[0, 1, 2]);
}
