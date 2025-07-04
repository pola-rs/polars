use arrow::bitmap::BitmapBuilder;

use crate::prelude::*;

pub struct CategoricalChunkedBuilder<T: PolarsCategoricalType> {
    name: PlSmallStr,
    dtype: DataType,
    mapping: Arc<CategoricalMapping>,
    is_enum: bool,
    cats: Vec<T::Native>,
    validity: BitmapBuilder,
}

impl<T: PolarsCategoricalType> CategoricalChunkedBuilder<T> {
    pub fn new(name: PlSmallStr, dtype: DataType) -> Self {
        let (DataType::Categorical(_, mapping) | DataType::Enum(_, mapping)) = &dtype else {
            panic!("non-Categorical/Enum dtype in CategoricalChunkedbuilder")
        };
        Self {
            name,
            mapping: mapping.clone(),
            is_enum: matches!(dtype, DataType::Enum(_, _)),
            dtype,
            cats: Vec::new(),
            validity: BitmapBuilder::new(),
        }
    }

    pub fn dtype(&self) -> &DataType {
        &self.dtype
    }

    pub fn reserve(&mut self, len: usize) {
        self.cats.reserve(len);
        self.validity.reserve(len);
    }

    pub fn append_cat(
        &mut self,
        cat: CatSize,
        mapping: &Arc<CategoricalMapping>,
    ) -> PolarsResult<()> {
        if Arc::ptr_eq(&self.mapping, mapping) {
            self.cats.push(T::Native::from_cat(cat));
            self.validity.push(true);
        } else if let Some(s) = mapping.cat_to_str(cat) {
            self.append_str(s)?;
        } else {
            self.append_null();
        }
        Ok(())
    }

    pub fn append_str(&mut self, val: &str) -> PolarsResult<()> {
        let cat = if self.is_enum {
            self.mapping.get_cat(val).ok_or_else(|| {
                polars_err!(ComputeError: "attempted to insert '{val}' into Enum which does not contain this string")
            })?
        } else {
            self.mapping.insert_cat(val)?
        };
        self.cats.push(T::Native::from_cat(cat));
        self.validity.push(true);
        Ok(())
    }

    pub fn append_null(&mut self) {
        self.cats.push(T::Native::default());
        self.validity.push(false);
    }

    pub fn finish(self) -> CategoricalChunked<T> {
        unsafe {
            let phys = ChunkedArray::from_vec_validity(
                self.name,
                self.cats,
                self.validity.into_opt_validity(),
            );
            CategoricalChunked::from_cats_and_dtype_unchecked(phys, self.dtype)
        }
    }
}
