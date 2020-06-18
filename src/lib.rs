#![allow(dead_code)]
#![feature(iterator_fold_self)]
pub mod error;
pub mod series {
    pub mod aggregate;
    pub mod arithmetic;
    pub mod chunked_array;
    pub mod comparison;
    pub mod series;
}
pub mod datatypes;
pub mod fmt;
pub mod frame;
pub mod prelude;
pub mod testing;
