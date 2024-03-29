mod av_buffer;
mod dataframe;
mod transpose;

use std::borrow::Borrow;
use std::fmt::Debug;
#[cfg(feature = "object")]
use std::hash::{Hash, Hasher};
use std::hint::unreachable_unchecked;

use arrow::bitmap::Bitmap;
pub use av_buffer::*;
#[cfg(feature = "object")]
use polars_utils::total_ord::TotalHash;
use rayon::prelude::*;

use crate::prelude::*;
use crate::utils::{dtypes_to_schema, dtypes_to_supertype, try_get_supertype};
use crate::POOL;

#[cfg(feature = "object")]
pub(crate) struct AnyValueRows<'a> {
    vals: Vec<AnyValue<'a>>,
    width: usize,
}

#[cfg(feature = "object")]
pub(crate) struct AnyValueRow<'a>(&'a [AnyValue<'a>]);

#[cfg(feature = "object")]
impl<'a> AnyValueRows<'a> {
    pub(crate) fn get(&'a self, i: usize) -> AnyValueRow<'a> {
        let start = i * self.width;
        let end = (i + 1) * self.width;
        AnyValueRow(&self.vals[start..end])
    }
}

#[cfg(feature = "object")]
impl TotalEq for AnyValueRow<'_> {
    fn tot_eq(&self, other: &Self) -> bool {
        let lhs = self.0;
        let rhs = other.0;

        // Should only be used in that context.
        debug_assert_eq!(lhs.len(), rhs.len());
        lhs.iter().zip(rhs.iter()).all(|(l, r)| l == r)
    }
}

#[cfg(feature = "object")]
impl TotalHash for AnyValueRow<'_> {
    fn tot_hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.0.iter().for_each(|av| av.hash(state))
    }
}

impl DataFrame {
    #[cfg(feature = "object")]
    #[allow(clippy::wrong_self_convention)]
    // Create indexable rows in a single allocation.
    pub(crate) fn to_av_rows(&mut self) -> AnyValueRows<'_> {
        self.as_single_chunk_par();
        let width = self.width();
        let size = width * self.height();
        let mut buf = vec![AnyValue::Null; size];
        for (col_i, s) in self.columns.iter().enumerate() {
            match s.dtype() {
                #[cfg(feature = "object")]
                DataType::Object(_, _) => {
                    for row_i in 0..s.len() {
                        let av = s.get(row_i).unwrap();
                        buf[row_i * width + col_i] = av
                    }
                },
                _ => {
                    for (row_i, av) in s.iter().enumerate() {
                        buf[row_i * width + col_i] = av
                    }
                },
            }
        }
        AnyValueRows { vals: buf, width }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Row<'a>(pub Vec<AnyValue<'a>>);

impl<'a> Row<'a> {
    pub fn new(values: Vec<AnyValue<'a>>) -> Self {
        Row(values)
    }
}

type Tracker = PlIndexMap<String, PlHashSet<DataType>>;

pub fn infer_schema(
    iter: impl Iterator<Item = Vec<(String, impl Into<DataType>)>>,
    infer_schema_length: usize,
) -> Schema {
    let mut values: Tracker = Tracker::default();
    let len = iter.size_hint().1.unwrap_or(infer_schema_length);

    let max_infer = std::cmp::min(len, infer_schema_length);
    for inner in iter.take(max_infer) {
        for (key, value) in inner {
            add_or_insert(&mut values, &key, value.into());
        }
    }
    Schema::from_iter(resolve_fields(values))
}

fn add_or_insert(values: &mut Tracker, key: &str, data_type: DataType) {
    if data_type == DataType::Null {
        return;
    }

    if values.contains_key(key) {
        let x = values.get_mut(key).unwrap();
        x.insert(data_type);
    } else {
        // create hashset and add value type
        let mut hs = PlHashSet::new();
        hs.insert(data_type);
        values.insert(key.to_string(), hs);
    }
}

fn resolve_fields(spec: Tracker) -> Vec<Field> {
    spec.iter()
        .map(|(k, hs)| {
            let v: Vec<&DataType> = hs.iter().collect();
            Field::new(k, coerce_data_type(&v))
        })
        .collect()
}

/// Coerces a slice of datatypes into a single supertype.
pub fn coerce_data_type<A: Borrow<DataType>>(datatypes: &[A]) -> DataType {
    use DataType::*;

    let are_all_equal = datatypes.windows(2).all(|w| w[0].borrow() == w[1].borrow());

    if are_all_equal {
        return datatypes[0].borrow().clone();
    }
    if datatypes.len() > 2 {
        return String;
    }

    let (lhs, rhs) = (datatypes[0].borrow(), datatypes[1].borrow());
    try_get_supertype(lhs, rhs).unwrap_or(String)
}

/// Infer the schema of rows by determining the supertype of the values.
///
/// Field names are set as `column_0`, `column_1`, and so on.
pub fn rows_to_schema_supertypes(
    rows: &[Row],
    infer_schema_length: Option<usize>,
) -> PolarsResult<Schema> {
    let dtypes = rows_to_supertypes(rows, infer_schema_length)?;
    let schema = dtypes_to_schema(dtypes);
    Ok(schema)
}

/// Infer the schema data types of rows by determining the supertype of the values.
pub fn rows_to_supertypes(
    rows: &[Row],
    infer_schema_length: Option<usize>,
) -> PolarsResult<Vec<DataType>> {
    polars_ensure!(!rows.is_empty(), NoData: "no rows, cannot infer schema");

    let max_infer = infer_schema_length.unwrap_or(rows.len());

    let mut dtypes: Vec<PlIndexSet<DataType>> = vec![PlIndexSet::new(); rows[0].0.len()];
    for row in rows.iter().take(max_infer) {
        for (val, dtypes_set) in row.0.iter().zip(dtypes.iter_mut()) {
            dtypes_set.insert(val.into());
        }
    }

    dtypes
        .into_iter()
        .map(|dtypes_set| dtypes_to_supertype(&dtypes_set))
        .collect()
}

/// Infer schema from rows and set the first no null type as column data type.
pub fn rows_to_schema_first_non_null(
    rows: &[Row],
    infer_schema_length: Option<usize>,
) -> PolarsResult<Schema> {
    polars_ensure!(!rows.is_empty(), NoData: "no rows, cannot infer schema");

    let max_infer = infer_schema_length.unwrap_or(rows.len());
    let mut schema: Schema = (&rows[0]).into();

    // the first row that has no nulls will be used to infer the schema.
    // if there is a null, we check the next row and see if we can update the schema

    for row in rows.iter().take(max_infer).skip(1) {
        // for i in 1..max_infer {
        let nulls: Vec<_> = schema
            .iter_dtypes()
            .enumerate()
            .filter_map(|(i, dtype)| {
                // double check struct and list types types
                // nested null values can be wrongly inferred by front ends
                match dtype {
                    DataType::Null | DataType::List(_) => Some(i),
                    #[cfg(feature = "dtype-struct")]
                    DataType::Struct(_) => Some(i),
                    _ => None,
                }
            })
            .collect();
        if nulls.is_empty() {
            break;
        } else {
            for i in nulls {
                let val = &row.0[i];

                if !val.is_nested_null() {
                    let dtype = val.into();
                    schema.set_dtype_at_index(i, dtype).unwrap();
                }
            }
        }
    }
    Ok(schema)
}

impl<'a> From<&AnyValue<'a>> for Field {
    fn from(val: &AnyValue<'a>) -> Self {
        Field::new("", val.into())
    }
}

impl From<&Row<'_>> for Schema {
    fn from(row: &Row) -> Self {
        row.0
            .iter()
            .enumerate()
            .map(|(i, av)| {
                let dtype = av.into();
                Field::new(format!("column_{i}").as_ref(), dtype)
            })
            .collect()
    }
}
