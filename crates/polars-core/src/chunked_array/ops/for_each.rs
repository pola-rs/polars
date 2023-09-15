use crate::prelude::*;

impl<T> ChunkedArray<T>
where
    T: PolarsDataType,
{
    pub fn for_each_generic<'a, F>(&'a self, mut op: F)
    where
        F: FnMut(Option<T::Physical<'a>>),
    {
        if self.null_count() == 0 {
            self.downcast_iter().for_each(|arr| {
                arr.values_iter().for_each(|x| op(Some(x)));
            })
        } else {
            self.downcast_iter().for_each(|arr| {
                arr.iter().for_each(|x| op(x));
            })
        }
    }
}
