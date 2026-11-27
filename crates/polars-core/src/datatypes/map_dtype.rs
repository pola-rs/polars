use polars_error::{PolarsResult, polars_ensure};
use polars_utils::pl_str::PlSmallStr;

use super::{DataType, Field, StructFields};
use crate::prelude::{MAP_KEY_NAME, MAP_VALUE_NAME};

/// The key and value dtypes of a `Map` dtype, which satisfy the Map key and value rules.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MapDtype {
    key: Box<DataType>,
    value: Box<DataType>,
}

impl MapDtype {
    pub fn try_new(key: DataType, value: DataType) -> PolarsResult<Self> {
        ensure_valid_key(&key)?;
        ensure_valid_value(&value)?;
        Ok(Self {
            key: Box::new(key),
            value: Box::new(value),
        })
    }

    /// The `Map` whose entries are `entries`, matching the two fields by position.
    ///
    /// Positional, should only be called by Arrow/Parquet readers.
    pub(crate) fn from_positional_entries(entries: &DataType) -> Option<Self> {
        let DataType::Struct(fields) = entries else {
            return None;
        };
        let [key, value] = fields.as_slice() else {
            return None;
        };
        Self::try_new(key.dtype().clone(), value.dtype().clone()).ok()
    }

    /// The `Map` whose entries are `entries`, matching the `key` and `value` fields by name.
    pub fn try_from_named_entries(entries: &DataType) -> PolarsResult<Self> {
        crate::chunked_array::logical::ensure_map_entries_dtype(entries)?;
        let DataType::Struct(fields) = entries else {
            unreachable!("map entries are a struct")
        };
        let dtype_of = |name: &PlSmallStr| {
            fields
                .iter()
                .find(|f| f.name() == name)
                .expect("entry fields checked above")
                .dtype()
                .clone()
        };
        Self::try_new(dtype_of(&MAP_KEY_NAME), dtype_of(&MAP_VALUE_NAME))
    }

    pub fn key(&self) -> &DataType {
        &self.key
    }

    pub fn value(&self) -> &DataType {
        &self.value
    }

    pub fn into_parts(self) -> (DataType, DataType) {
        (*self.key, *self.value)
    }

    /// The `Struct {key, value}` dtype of an entry.
    pub fn entries_dtype(&self) -> DataType {
        DataType::Struct(
            StructFields::try_new(vec![
                Field::new(MAP_KEY_NAME, self.key().clone()),
                Field::new(MAP_VALUE_NAME, self.value().clone()),
            ])
            .unwrap(),
        )
    }

    /// The `List(Struct {key, value})` dtype of the [`Series`](crate::series::Series) that
    /// stores this `Map`.
    ///
    /// This is intentionally separate from [`DataType::to_storage`]. Adding `Map` there would
    /// imply also unwrapping it in `Series::to_storage`. Python `map_elements` calls the
    /// latter, so each map row would reach the UDF as a `List(Struct)` value instead of a
    /// `dict`.
    pub fn storage_dtype(&self) -> DataType {
        DataType::List(Box::new(self.entries_dtype()))
    }
}

fn ensure_valid_key(dtype: &DataType) -> PolarsResult<()> {
    // A top-level Null cannot supply a non-null key; nested nulls are fine.
    polars_ensure!(
        !matches!(dtype, DataType::Null),
        InvalidOperation: "`{dtype}` cannot be used as a Map key dtype"
    );
    // Neither is supported by the row encoder.
    polars_ensure!(
        !dtype.contains_objects(),
        InvalidOperation: "`{dtype}` cannot be used as a Map key dtype: objects cannot be row-encoded"
    );
    polars_ensure!(
        !dtype.contains_unknown(),
        InvalidOperation: "`{dtype}` cannot be used as a Map key dtype: it is not materialized"
    );
    Ok(())
}

fn ensure_valid_value(dtype: &DataType) -> PolarsResult<()> {
    // Map values are a `Struct` field, and a struct cannot hold objects.
    polars_ensure!(
        !dtype.contains_objects(),
        InvalidOperation: "`{dtype}` cannot be used as a Map value dtype: objects cannot be nested"
    );
    Ok(())
}

#[cfg(all(test, feature = "serde"))]
mod test {
    use super::*;

    #[test]
    fn deserialize_rejects_invalid_map_dtype() {
        let valid = DataType::Map(MapDtype::try_new(DataType::String, DataType::Int64).unwrap());
        let json = serde_json::to_string(&valid).unwrap();
        assert_eq!(serde_json::from_str::<DataType>(&json).unwrap(), valid);

        let null_key = json.replace("String", "Null");
        assert_ne!(null_key, json);
        let err = serde_json::from_str::<DataType>(&null_key).unwrap_err();
        assert!(
            err.to_string()
                .contains("cannot be used as a Map key dtype")
        );
    }
}
