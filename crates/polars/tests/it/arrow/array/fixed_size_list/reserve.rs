use std::any::Any;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow::array::builder::{ShareStrategy, StaticArrayBuilder};
use arrow::array::{
    Array, FixedSizeListArrayBuilder, MutableArray, MutableFixedSizeListArray, NullArray,
};
use arrow::bitmap::MutableBitmap;
use arrow::datatypes::{ArrowDataType, Field};
use polars_utils::IdxSize;

#[derive(Debug)]
struct ReserveTracker(Arc<AtomicUsize>);

impl ReserveTracker {
    fn new(reserved: Arc<AtomicUsize>) -> Self {
        Self(reserved)
    }

    fn record(&self, additional: usize) {
        self.0.store(additional, Ordering::Relaxed);
    }
}

impl StaticArrayBuilder for ReserveTracker {
    type Array = NullArray;

    fn dtype(&self) -> &ArrowDataType {
        &ArrowDataType::Null
    }

    fn reserve(&mut self, additional: usize) {
        self.record(additional);
    }

    fn freeze(self) -> Self::Array {
        NullArray::new(ArrowDataType::Null, 0)
    }

    fn freeze_reset(&mut self) -> Self::Array {
        NullArray::new(ArrowDataType::Null, 0)
    }

    fn len(&self) -> usize {
        0
    }

    fn extend_nulls(&mut self, _length: usize) {
        unreachable!()
    }

    fn subslice_extend(
        &mut self,
        _other: &Self::Array,
        _start: usize,
        _length: usize,
        _share: ShareStrategy,
    ) {
        unreachable!()
    }

    fn subslice_extend_each_repeated(
        &mut self,
        _other: &Self::Array,
        _start: usize,
        _length: usize,
        _repeats: usize,
        _share: ShareStrategy,
    ) {
        unreachable!()
    }

    unsafe fn gather_extend(
        &mut self,
        _other: &Self::Array,
        _idxs: &[IdxSize],
        _share: ShareStrategy,
    ) {
        unreachable!()
    }

    fn opt_gather_extend(
        &mut self,
        _other: &Self::Array,
        _idxs: &[IdxSize],
        _share: ShareStrategy,
    ) {
        unreachable!()
    }
}

impl MutableArray for ReserveTracker {
    fn dtype(&self) -> &ArrowDataType {
        &ArrowDataType::Null
    }

    fn len(&self) -> usize {
        0
    }

    fn validity(&self) -> Option<&MutableBitmap> {
        None
    }

    fn as_box(&mut self) -> Box<dyn Array> {
        Box::new(NullArray::new(ArrowDataType::Null, 0))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_mut_any(&mut self) -> &mut dyn Any {
        self
    }

    fn push_null(&mut self) {
        unreachable!()
    }

    fn reserve(&mut self, additional: usize) {
        self.record(additional);
    }

    fn shrink_to_fit(&mut self) {}
}

fn fixed_size_list_dtype(width: usize) -> ArrowDataType {
    ArrowDataType::FixedSizeList(
        Box::new(Field::new("item".into(), ArrowDataType::Null, true)),
        width,
    )
}

#[test]
fn builder_reserves_child_values() {
    let reserved = Arc::new(AtomicUsize::new(0));
    let mut builder = FixedSizeListArrayBuilder::new(
        fixed_size_list_dtype(7),
        ReserveTracker::new(Arc::clone(&reserved)),
    );

    StaticArrayBuilder::reserve(&mut builder, 3);

    assert_eq!(reserved.load(Ordering::Relaxed), 21);
}

#[test]
fn mutable_reserves_child_values() {
    let reserved = Arc::new(AtomicUsize::new(0));
    let mut array = MutableFixedSizeListArray::new(ReserveTracker::new(Arc::clone(&reserved)), 7);

    array.reserve(3);

    assert_eq!(reserved.load(Ordering::Relaxed), 21);
}
