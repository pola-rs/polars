mod binary;
mod boolean;
mod fixed_size_binary;
mod fixed_size_list;
mod list;
mod map;
mod null;
mod primitive;
mod struct_;
mod utf8;

// check that `PartialEq` can be derived
#[allow(dead_code)]
#[derive(PartialEq)]
struct A {
    array: Box<dyn arrow::scalar::Scalar>,
}
