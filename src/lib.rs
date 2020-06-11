#![allow(dead_code)]
#![feature(min_specialization)]
mod error;
mod series {
    pub(crate) mod aggregate;
    pub(crate) mod arithmetic;
    pub mod chunked_array;
    pub(crate) mod comparison;
    pub(crate) mod series;
}
mod datatypes;
mod frame;
mod prelude;
