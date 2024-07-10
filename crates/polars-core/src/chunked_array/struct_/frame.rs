use crate::frame::DataFrame;
use crate::prelude::StructChunked2;

impl DataFrame {
    pub fn into_struct(self, name: &str) -> StructChunked2 {
        StructChunked2::from_series(name, &self.columns).expect("same invariants")
    }

}