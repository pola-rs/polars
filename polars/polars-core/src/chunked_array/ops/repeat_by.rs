use arrow::array::ListArray;
use polars_arrow::array::ListFromIter;

use super::RepeatBy;
use crate::prelude::*;

use std::iter::repeat;
type LargeListArray = ListArray<i64>;

impl<T> RepeatBy for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn repeat_by(&self, by: &IdxCa) -> ListChunked {
        let iter = self
            .into_iter()
            .zip(by.into_iter())
            .map(|(opt_v, opt_by)| opt_by.map(|by| std::iter::repeat(opt_v).take(by as usize)));

        // Safety:
        // Length of iter is trusted
        unsafe {
            ListChunked::from_chunks(
                self.name(),
                vec![Box::new(LargeListArray::from_iter_primitive_trusted_len::<
                    T::Native,
                    _,
                    _,
                >(iter, T::get_dtype().to_arrow()))],
            )
        }
    }
}
impl RepeatBy for BooleanChunked {
    fn repeat_by(&self, by: &IdxCa) -> ListChunked {
        let iter = self
            .into_iter()
            .zip(by.into_iter())
            .map(|(opt_v, opt_by)| opt_by.map(|by| std::iter::repeat(opt_v).take(by as usize)));

        // Safety:
        // Length of iter is trusted
        unsafe {
            ListChunked::from_chunks(
                self.name(),
                vec![Box::new(LargeListArray::from_iter_bool_trusted_len(iter))],
            )
        }
    }
}
impl RepeatBy for Utf8Chunked {
    fn repeat_by(&self, by: &IdxCa) -> ListChunked {
        // TODO! dispatch via binary.
        if self.len() != by.len() {
            return self.repeat_by(&IdxCa::new("", std::iter::repeat(by.get(0).unwrap()).take(self.len()).collect::<Vec<IdxSize>>()));
        }

        let iter = self
            .into_iter()
            .zip(by.into_iter())
            .map(|(opt_v, opt_by)| {let k = opt_by.map(|by| std::iter::repeat(opt_v).take(by as usize));
        eprintln!("this is k {:?}, opt_v {:?}, opt_by {:?}", k, opt_v, opt_by); k});

        // eprintln!("This is the len of self {} and the len of by {}", self.len(), by.len());
        // let it = match self.len() == by.len() {
        //     true => by,
        //     false => {
        //         let t = by.get(0);
        //         IdxCa::new("literal", std::iter::repeat(by.get(0) as IdxSize).take(self.len()).collect::<Vec<IdxCa>>()).as_ref(),
        //     }
        // };
        // let t: IdxCa = by.into_iter().chain(repeat(by.get(by.len() - 1)).take(self.len().saturating_sub(by.len()))).collect();
        // let iter = self
        //     .into_iter()
        //     .zip(it)
        //     .map(|(opt_v, opt_by)| {let k = opt_by.map(|by| std::iter::repeat(opt_v).take(by as usize));
        // eprintln!("this is k {:?}, opt_v {:?}, opt_by {:?}", k, opt_v, opt_by); k});
       //  .iter()
       //  .chain(repeat(&a[a.len() - 1]).take(b.len().saturating_sub(a.len())))
       //  .zip(
       //      b
       //  )
        // Safety:
        // Length of iter is trusted
        unsafe {
            let d = Box::new(LargeListArray::from_iter_utf8_trusted_len(
                        iter,
                        self.len(),
                    ));
            ListChunked::from_chunks(
                self.name(),
                vec![d],
            )
        }
    }
}
impl RepeatBy for BinaryChunked {
    fn repeat_by(&self, by: &IdxCa) -> ListChunked {
        let iter = self
            .into_iter()
            .zip(by.into_iter())
            .map(|(opt_v, opt_by)| opt_by.map(|by| std::iter::repeat(opt_v).take(by as usize)));

        // Safety:
        // Length of iter is trusted
        unsafe {
            ListChunked::from_chunks(
                self.name(),
                vec![Box::new(LargeListArray::from_iter_binary_trusted_len(
                    iter,
                    self.len(),
                ))],
            )
        }
    }
}
