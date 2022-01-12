use arrow::array::{Array, ListArray};
use arrow::bitmap::MutableBitmap;
use arrow::compute::concatenate;
use arrow::error::Result;

pub struct AnonymousBuilder<'a> {
    arrays: Vec<&'a dyn Array>,
    offsets: Vec<i64>,
    validity: Option<MutableBitmap>,
    size: i64,
}

impl<'a> AnonymousBuilder<'a> {
    pub fn new(size: usize) -> Self {
        let mut offsets = Vec::with_capacity(size + 1);
        offsets.push(0i64);
        Self {
            arrays: Vec::with_capacity(size),
            offsets,
            validity: None,
            size: 0,
        }
    }
    #[inline]
    fn last_offset(&self) -> i64 {
        *self.offsets.last().unwrap()
    }

    pub fn push(&mut self, arr: &'a dyn Array) {
        self.size += arr.len() as i64;
        self.offsets.push(self.size);
        self.arrays.push(arr);

        if let Some(validity) = &mut self.validity {
            validity.push(true)
        }
    }
    pub fn push_null(&mut self) {
        self.offsets.push(self.last_offset());
        match &mut self.validity {
            Some(validity) => validity.push(false),
            None => self.init_validity(),
        }
    }

    fn init_validity(&mut self) {
        let len = self.offsets.len() - 1;

        let mut validity = MutableBitmap::with_capacity(self.offsets.capacity());
        validity.extend_constant(len, true);
        validity.set(len - 1, false);
        self.validity = Some(validity)
    }

    pub fn finish(self) -> Result<ListArray<i64>> {
        let inner_dtype = self.arrays[0].data_type();
        let values = concatenate::concatenate(&self.arrays)?;

        let dtype = ListArray::<i64>::default_datatype(inner_dtype.clone());
        Ok(ListArray::<i64>::from_data(
            dtype,
            self.offsets.into(),
            values.into(),
            self.validity.map(|validity| validity.into()),
        ))
    }
}
