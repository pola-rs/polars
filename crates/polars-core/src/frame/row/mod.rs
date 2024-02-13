mod av_buffer;
mod dataframe;
mod transpose;

use std::borrow::Borrow;
use std::fmt::Debug;
use std::hint::unreachable_unchecked;

use arrow::bitmap::Bitmap;
pub use av_buffer::*;
use rayon::prelude::*;

use crate::prelude::*;
use crate::utils::try_get_supertype;
use crate::POOL;

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

pub fn any_values_to_dtype(column: &[AnyValue]) -> PolarsResult<(DataType, usize)> {
    // we need an index-map as the order of dtypes influences how the
    // struct fields are constructed.
    let mut types_set = PlIndexSet::new();
    for val in column.iter() {
        types_set.insert(val.into());
    }
    let n_types = types_set.len();
    Ok((types_set_to_dtype(types_set)?, n_types))
}

fn types_set_to_dtype(types_set: PlIndexSet<DataType>) -> PolarsResult<DataType> {
    types_set
        .into_iter()
        .map(Ok)
        .reduce(|a, b| try_get_supertype(&a?, &b?))
        .unwrap()
}

/// Infer schema from rows and set the supertypes of the columns as column data type.
pub fn rows_to_schema_supertypes(
    rows: &[Row],
    infer_schema_length: Option<usize>,
) -> PolarsResult<Schema> {
    // no of rows to use to infer dtype
    let max_infer = infer_schema_length.unwrap_or(rows.len());
    polars_ensure!(!rows.is_empty(), NoData: "no rows, cannot infer schema");
    let mut dtypes: Vec<PlIndexSet<DataType>> = vec![PlIndexSet::new(); rows[0].0.len()];

    for row in rows.iter().take(max_infer) {
        for (val, types_set) in row.0.iter().zip(dtypes.iter_mut()) {
            types_set.insert(val.into());
        }
    }

    dtypes
        .into_iter()
        .enumerate()
        .map(|(i, types_set)| {
            let dtype = if types_set.is_empty() {
                DataType::Unknown
            } else {
                types_set_to_dtype(types_set)?
            };
            Ok(Field::new(format!("column_{i}").as_ref(), dtype))
        })
        .collect::<PolarsResult<_>>()
}

/// Infer schema from rows and set the first no null type as column data type.
pub fn rows_to_schema_first_non_null(rows: &[Row], infer_schema_length: Option<usize>) -> Schema {
    // no of rows to use to infer dtype
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
    schema
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
