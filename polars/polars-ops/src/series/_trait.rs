use std::ops::Deref;

use super::*;

pub trait SeriesOps {
    fn dtype(&self) -> &DataType;
}

impl SeriesOps for Series {
    fn dtype(&self) -> &DataType {
        self.deref().dtype()
    }
}
