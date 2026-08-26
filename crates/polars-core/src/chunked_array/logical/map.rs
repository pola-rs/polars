use crate::chunked_array::cast::CastOptions;
use crate::prelude::*;

/// A `Map` backed by a `List(Struct {key, value})` [`Series`].
///
/// Keys are unique and non-null within each row. Equality is entry-order-sensitive.
#[derive(Clone)]
pub struct MapChunked {
    dtype: DataType,
    storage: Series,
}

impl MapChunked {
    /// # Safety
    /// `dtype` must be a [`DataType::Map`] matching `storage`, with unique,
    /// non-null keys in every row.
    pub unsafe fn from_storage_unchecked(dtype: DataType, storage: Series) -> Self {
        debug_assert_eq!(
            dtype.map_entries_list_dtype().as_ref(),
            Some(storage.dtype())
        );
        Self { dtype, storage }
    }

    /// Validate and canonicalize a map's storage.
    pub fn try_from_storage(dtype: DataType, storage: Series) -> PolarsResult<Self> {
        let DataType::Map(key, _) = &dtype else {
            polars_bail!(InvalidOperation: "`{dtype}` is not a Map dtype");
        };
        key.ensure_valid_map_key()?;

        let storage_dtype = dtype.map_entries_list_dtype().unwrap();
        polars_ensure!(
            storage.dtype() == &storage_dtype,
            InvalidOperation: "expected `{storage_dtype}` storage for `{dtype}`, got `{}`",
            storage.dtype()
        );

        let storage = canonicalize_map_storage(storage)?;
        Ok(Self { dtype, storage })
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

    pub fn key_dtype(&self) -> &DataType {
        match &self.dtype {
            DataType::Map(key, _) => key,
            _ => unreachable!("MapChunked must have DataType::Map"),
        }
    }

    pub fn value_dtype(&self) -> &DataType {
        match &self.dtype {
            DataType::Map(_, value) => value,
            _ => unreachable!("MapChunked must have DataType::Map"),
        }
    }

    pub fn entries_dtype(&self) -> DataType {
        DataType::map_entries(self.key_dtype().clone(), self.value_dtype().clone())
    }

    pub fn storage(&self) -> &Series {
        &self.storage
    }

    /// Mutable access is crate-private to protect the key invariants.
    pub(crate) fn storage_mut(&mut self) -> &mut Series {
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

    pub fn get_any_value(&self, _i: usize) -> PolarsResult<AnyValue<'_>> {
        todo!("AnyValue::Map")
    }

    pub fn cast_with_options(
        &self,
        _dtype: &DataType,
        _options: CastOptions,
    ) -> PolarsResult<Series> {
        todo!("MapChunked::cast_with_options")
    }
}

/// Canonicalize duplicates using first-key-position/last-value semantics:
/// `[("a", 1), ("b", 2), ("a", 3)] -> [("a", 3), ("b", 2)]`.
pub(crate) fn canonicalize_map_storage(_storage: Series) -> PolarsResult<Series> {
    todo!("map key canonicalization")
}
