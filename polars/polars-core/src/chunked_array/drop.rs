use crate::chunked_array::object::extension::drop::drop_list;
use crate::prelude::*;

impl<T> Drop for ChunkedArray<T> {
    fn drop(&mut self) {
        if matches!(self.dtype(), DataType::List(_)) {
            // guarded by the type system
            unsafe { drop_list(std::mem::transmute(self)) }
        }
    }
}
