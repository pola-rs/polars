use polars_array::builder::StaticArrayBuilder;

use crate::chunked_array::object::builder::ObjectChunkedBuilder;
use crate::prelude::*;

impl<T: PolarsObject> ObjectChunked<T> {
    pub(crate) fn get_list_builder(
        name: PlSmallStr,
        values_capacity: usize,
        list_capacity: usize,
    ) -> Box<dyn ListBuilderTrait> {
        Box::new(ExtensionListBuilder::<T>::new(
            name,
            values_capacity,
            list_capacity,
        ))
    }
}

pub(crate) struct ExtensionListBuilder<T: PolarsObject> {
    values_builder: ObjectChunkedBuilder<T>,
    offsets: Vec<u64>,
    fast_explode: bool,
}

impl<T: PolarsObject> ExtensionListBuilder<T> {
    pub(crate) fn new(name: PlSmallStr, values_capacity: usize, list_capacity: usize) -> Self {
        let mut offsets = Vec::with_capacity(list_capacity + 1);
        offsets.push(0);
        Self {
            values_builder: ObjectChunkedBuilder::new(name, values_capacity),
            offsets,
            fast_explode: true,
        }
    }
}

impl<T: PolarsObject> ListBuilderTrait for ExtensionListBuilder<T> {
    fn append_series(&mut self, s: &Series) -> PolarsResult<()> {
        let arr = s.as_any().downcast_ref::<ObjectChunked<T>>().unwrap();

        for v in arr.iter() {
            self.values_builder.append_option(v.cloned())
        }
        if arr.is_empty() {
            self.fast_explode = false;
        }
        let len_so_far = self.offsets[self.offsets.len() - 1];
        self.offsets.push(len_so_far + arr.len() as u64);
        Ok(())
    }

    fn append_null(&mut self) {
        self.values_builder.append_null();
        let len_so_far = self.offsets[self.offsets.len() - 1];
        self.offsets.push(len_so_far + 1);
    }

    fn finish(&mut self) -> ListChunked {
        let mut values_builder = std::mem::take(&mut self.values_builder);
        let offsets = std::mem::take(&mut self.offsets);
        let name = values_builder.field().name().clone();

        // The values of a list of objects are the object array itself, which holds the values and
        // drops them with it — there is no packing into bytes for an in-memory column.
        let length = offsets.len() - 1;
        let values = values_builder.freeze_reset();
        // SAFETY: the offsets were built by appending the length of every element.
        let arr =
            unsafe { PlListArray::new_unchecked(Box::new(values), offsets.into(), length, None) };

        let mut listarr = unsafe {
            ListChunked::from_chunks_and_dtype_unchecked(
                name,
                vec![Box::new(arr)],
                DataType::List(Box::new(DataType::Object(T::type_name()))),
            )
        };
        if self.fast_explode {
            listarr.set_fast_explode()
        }
        listarr
    }
}
