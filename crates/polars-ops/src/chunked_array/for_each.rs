use polars_core::prelude::*;

pub fn for_each<'a, T, F>(ca: &'a ChunkedArray<T>, mut op: F)
where
    T: PolarsDataType,
    F: FnMut(Option<T::Physical<'a>>),
{
    ca.downcast_iter().for_each(|arr| {
        arr.iter().for_each(&mut op);
    })
}
