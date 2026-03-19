use crate::chunked_array::cast::CastOptions;
use crate::prelude::*;

#[derive(Clone)]
pub struct ExtensionChunked {
    dtype: DataType,
    storage: Series,
}

impl ExtensionChunked {
    pub fn from_storage(typ: ExtensionTypeInstance, storage: Series) -> Self {
        assert!(
            !matches!(storage.dtype(), DataType::Extension(_, _)),
            "can't nest extension types"
        );
        let dtype = DataType::Extension(typ, Box::new(storage.dtype().clone()));
        Self { dtype, storage }
    }

    pub fn name(&self) -> &PlSmallStr {
        self.storage.name()
    }

    pub fn rename(&mut self, name: PlSmallStr) {
        self.storage.rename(name);
    }

    pub fn field(&self) -> Field {
        Field::new(self.storage.name().clone(), self.dtype.clone())
    }

    pub fn dtype(&self) -> &DataType {
        &self.dtype
    }

    pub fn extension_type(&self) -> &ExtensionTypeInstance {
        match &self.dtype {
            DataType::Extension(typ, _) => typ,
            _ => unreachable!("ExtensionChunked must have DataType::Extension"),
        }
    }

    pub fn storage(&self) -> &Series {
        &self.storage
    }

    pub fn storage_mut(&mut self) -> &mut Series {
        &mut self.storage
    }

    pub fn into_storage(self) -> Series {
        self.storage
    }

    pub fn len(&self) -> usize {
        self.storage.len()
    }

    pub fn is_empty(&self) -> bool {
        self.storage.is_empty()
    }

    pub fn get_any_value(&self, i: usize) -> PolarsResult<AnyValue<'_>> {
        self.storage().get(i)
    }

    pub fn cast_with_options(
        &self,
        dtype: &DataType,
        _options: CastOptions,
    ) -> PolarsResult<Series> {
        polars_bail!(ComputeError: "cannot cast extension types to {dtype:?}")
    }
}
