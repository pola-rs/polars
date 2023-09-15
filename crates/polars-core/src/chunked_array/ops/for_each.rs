use crate::prelude::*;

impl<T> ChunkedArray<T>
where
    T: PolarsDataType,
{
    pub fn for_each<'a, F>(&'a self, mut op: F)
    where
        F: FnMut(Option<T::Physical<'a>>),
    {
        self.downcast_iter().for_each(|arr| {
            arr.iter().for_each(&mut op);
        })
    }
}
