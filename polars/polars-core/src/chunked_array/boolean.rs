use crate::prelude::*;
use crate::utils::NoNull;

impl BooleanChunked {
    pub fn arg_true(&self) -> IdxCa {
        let ca: NoNull<IdxCa> = (0..self.len() as IdxSize).collect_trusted();
        ca.into_inner().filter(self).unwrap()
    }
}
