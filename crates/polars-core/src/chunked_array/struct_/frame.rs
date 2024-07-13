use crate::frame::DataFrame;
use crate::prelude::StructChunked;

impl DataFrame {
    pub fn into_struct(self, name: &str) -> StructChunked {
        StructChunked::from_series(name, &self.columns).expect("same invariants")
    }
}
