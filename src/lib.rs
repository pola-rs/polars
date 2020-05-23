#![allow(dead_code)]
mod error;
mod series {
    pub(crate) mod arithmetic;
    pub mod chunked_array;
    pub(crate) mod iterator;
    pub(crate) mod series;
}
mod datatypes;
mod frame;
