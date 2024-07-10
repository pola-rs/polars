use crate::prelude::*;

impl DataFrame {
    pub fn into_struct(self, name: &str) -> StructChunked {
        StructChunked::new(name, &self.columns).unwrap()
    }
}
