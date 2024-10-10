use arrow::array::{Array, MutableArray, StructArray};
use arrow::datatypes::ArrowDataType;
use polars_error::PolarsResult;

use super::make_mutable;

#[derive(Debug)]
pub struct DynMutableStructArray {
    dtype: ArrowDataType,
    inner: Vec<Box<dyn MutableArray>>,
}

impl DynMutableStructArray {
    pub fn try_with_capacity(dtype: ArrowDataType, capacity: usize) -> PolarsResult<Self> {
        let inners = match dtype.to_logical_type() {
            ArrowDataType::Struct(inner) => inner,
            _ => unreachable!(),
        };

        assert!(!inners.is_empty());

        let inner = inners
            .iter()
            .map(|f| make_mutable(f.dtype(), capacity))
            .collect::<PolarsResult<Vec<_>>>()?;

        Ok(Self { dtype, inner })
    }

    pub fn inner_mut(&mut self) -> &mut [Box<dyn MutableArray>] {
        &mut self.inner
    }
}

impl MutableArray for DynMutableStructArray {
    fn dtype(&self) -> &ArrowDataType {
        &self.dtype
    }

    fn len(&self) -> usize {
        self.inner[0].len()
    }

    fn validity(&self) -> Option<&arrow::bitmap::MutableBitmap> {
        None
    }

    fn as_box(&mut self) -> Box<dyn Array> {
        let len = self.len();
        let inner = self.inner.iter_mut().map(|x| x.as_box()).collect();
        Box::new(StructArray::new(self.dtype.clone(), len, inner, None))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_mut_any(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn push_null(&mut self) {
        todo!()
    }

    fn reserve(&mut self, _: usize) {
        todo!();
    }

    fn shrink_to_fit(&mut self) {
        todo!()
    }
}
