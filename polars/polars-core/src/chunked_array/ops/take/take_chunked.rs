use polars_utils::slice::GetSaferUnchecked;

use super::*;
use crate::series::IsSorted;

pub trait TakeChunked {
    unsafe fn take_chunked_unchecked(&self, by: &[ChunkId], sorted: IsSorted) -> Self;

    unsafe fn take_opt_chunked_unchecked(&self, by: &[Option<ChunkId>]) -> Self;
}

impl<T> TakeChunked for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    unsafe fn take_chunked_unchecked(&self, by: &[ChunkId], sorted: IsSorted) -> Self {
        let mut ca = if self.null_count() == 0 {
            let arrs = self
                .downcast_iter()
                .map(|arr| arr.values().as_slice())
                .collect::<Vec<_>>();

            let ca: NoNull<Self> = by
                .iter()
                .map(|[chunk_idx, array_idx]| {
                    let arr = arrs.get_unchecked_release(*chunk_idx as usize);
                    *arr.get_unchecked_release(*array_idx as usize)
                })
                .collect_trusted();

            ca.into_inner()
        } else {
            let arrs = self.downcast_iter().collect::<Vec<_>>();
            by.iter()
                .map(|[chunk_idx, array_idx]| {
                    let arr = arrs.get_unchecked(*chunk_idx as usize);
                    arr.get_unchecked(*array_idx as usize)
                })
                .collect_trusted()
        };
        ca.rename(self.name());
        ca.set_sorted_flag(sorted);
        ca
    }

    unsafe fn take_opt_chunked_unchecked(&self, by: &[Option<ChunkId>]) -> Self {
        let arrs = self.downcast_iter().collect::<Vec<_>>();
        let mut ca: Self = by
            .iter()
            .map(|opt_idx| {
                opt_idx.and_then(|[chunk_idx, array_idx]| {
                    let arr = arrs.get_unchecked_release(chunk_idx as usize);
                    arr.get_unchecked(array_idx as usize)
                })
            })
            .collect_trusted();

        ca.rename(self.name());
        ca
    }
}

impl TakeChunked for Utf8Chunked {
    unsafe fn take_chunked_unchecked(&self, by: &[ChunkId], sorted: IsSorted) -> Self {
        let arrs = self.downcast_iter().collect::<Vec<_>>();
        let mut ca: Self = by
            .iter()
            .map(|[chunk_idx, array_idx]| {
                let arr = arrs.get_unchecked(*chunk_idx as usize);
                arr.get_unchecked(*array_idx as usize)
            })
            .collect_trusted();
        ca.rename(self.name());
        ca.set_sorted_flag(sorted);
        ca
    }

    unsafe fn take_opt_chunked_unchecked(&self, by: &[Option<ChunkId>]) -> Self {
        let arrs = self.downcast_iter().collect::<Vec<_>>();
        let mut ca: Self = by
            .iter()
            .map(|opt_idx| {
                opt_idx.and_then(|[chunk_idx, array_idx]| {
                    let arr = arrs.get_unchecked(chunk_idx as usize);
                    arr.get_unchecked(array_idx as usize)
                })
            })
            .collect_trusted();

        ca.rename(self.name());
        ca
    }
}

#[cfg(feature = "dtype-binary")]
impl TakeChunked for BinaryChunked {
    unsafe fn take_chunked_unchecked(&self, by: &[ChunkId], sorted: IsSorted) -> Self {
        let arrs = self.downcast_iter().collect::<Vec<_>>();
        let mut ca: Self = by
            .iter()
            .map(|[chunk_idx, array_idx]| {
                let arr = arrs.get_unchecked(*chunk_idx as usize);
                arr.get_unchecked(*array_idx as usize)
            })
            .collect_trusted();
        ca.rename(self.name());
        ca.set_sorted_flag(sorted);
        ca
    }

    unsafe fn take_opt_chunked_unchecked(&self, by: &[Option<ChunkId>]) -> Self {
        let arrs = self.downcast_iter().collect::<Vec<_>>();
        let mut ca: Self = by
            .iter()
            .map(|opt_idx| {
                opt_idx.and_then(|[chunk_idx, array_idx]| {
                    let arr = arrs.get_unchecked(chunk_idx as usize);
                    arr.get_unchecked(array_idx as usize)
                })
            })
            .collect_trusted();

        ca.rename(self.name());
        ca
    }
}

impl TakeChunked for BooleanChunked {
    unsafe fn take_chunked_unchecked(&self, by: &[ChunkId], sorted: IsSorted) -> Self {
        let arrs = self.downcast_iter().collect::<Vec<_>>();
        let mut ca: Self = by
            .iter()
            .map(|[chunk_idx, array_idx]| {
                let arr = arrs.get_unchecked(*chunk_idx as usize);
                arr.get_unchecked(*array_idx as usize)
            })
            .collect_trusted();
        ca.rename(self.name());
        ca.set_sorted_flag(sorted);
        ca
    }

    unsafe fn take_opt_chunked_unchecked(&self, by: &[Option<ChunkId>]) -> Self {
        let arrs = self.downcast_iter().collect::<Vec<_>>();
        let mut ca: Self = by
            .iter()
            .map(|opt_idx| {
                opt_idx.and_then(|[chunk_idx, array_idx]| {
                    let arr = arrs.get_unchecked(chunk_idx as usize);
                    arr.get_unchecked(array_idx as usize)
                })
            })
            .collect_trusted();

        ca.rename(self.name());
        ca
    }
}

impl TakeChunked for ListChunked {
    unsafe fn take_chunked_unchecked(&self, by: &[ChunkId], sorted: IsSorted) -> Self {
        let arrs = self.downcast_iter().collect::<Vec<_>>();
        let mut ca: Self = by
            .iter()
            .map(|[chunk_idx, array_idx]| {
                let arr = arrs.get_unchecked(*chunk_idx as usize);
                arr.get_unchecked(*array_idx as usize)
            })
            .collect();
        ca.rename(self.name());
        ca.set_sorted_flag(sorted);
        ca
    }

    unsafe fn take_opt_chunked_unchecked(&self, by: &[Option<ChunkId>]) -> Self {
        let arrs = self.downcast_iter().collect::<Vec<_>>();
        let mut ca: Self = by
            .iter()
            .map(|opt_idx| {
                opt_idx.and_then(|[chunk_idx, array_idx]| {
                    let arr = arrs.get_unchecked(chunk_idx as usize);
                    arr.get_unchecked(array_idx as usize)
                })
            })
            .collect();

        ca.rename(self.name());
        ca
    }
}
#[cfg(feature = "object")]
impl<T: PolarsObject> TakeChunked for ObjectChunked<T> {
    unsafe fn take_chunked_unchecked(&self, by: &[ChunkId], sorted: IsSorted) -> Self {
        let arrs = self.downcast_iter().collect::<Vec<_>>();

        let mut ca: Self = by
            .iter()
            .map(|[chunk_idx, array_idx]| {
                let arr = arrs.get_unchecked(*chunk_idx as usize);
                arr.get_unchecked(*array_idx as usize).cloned()
            })
            .collect();

        ca.rename(self.name());
        ca.set_sorted_flag(sorted);
        ca
    }

    unsafe fn take_opt_chunked_unchecked(&self, by: &[Option<ChunkId>]) -> Self {
        let arrs = self.downcast_iter().collect::<Vec<_>>();
        let mut ca: Self = by
            .iter()
            .map(|opt_idx| {
                opt_idx.and_then(|[chunk_idx, array_idx]| {
                    let arr = arrs.get_unchecked(chunk_idx as usize);
                    arr.get_unchecked(array_idx as usize).cloned()
                })
            })
            .collect();

        ca.rename(self.name());
        ca
    }
}
