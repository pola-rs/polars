#![allow(dead_code)]
#![feature(specialization)]
mod error;
mod series {
    pub(crate) mod chunked_array;
    pub(crate) mod iterator;
    pub(crate) mod series;
}
mod datatypes;
mod frame;
