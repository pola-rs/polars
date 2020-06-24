#![allow(dead_code)]
#![feature(iterator_fold_self)]
pub mod error;
pub mod series {
    pub(crate) mod aggregate;
    pub(crate) mod arithmetic;
    pub mod chunked_array;
    mod comparison;
    pub(crate) mod iterator;
    pub mod series;
}
pub mod datatypes;
mod fmt;
pub mod frame;
pub mod prelude;
pub mod testing;
