use polars_core::utils::slice_offsets;

use super::*;
use crate::{map, map_as_slice};

#[derive(Clone, Eq, PartialEq, Hash, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum StructFunction {
    FieldByIndex(i64),
    FieldByName(Arc<str>),
    RenameFields(Arc<[String]>),
    PrefixFields(Arc<str>),
    SuffixFields(Arc<str>),
    #[cfg(feature = "json")]
    JsonEncode,
    WithFields,
    MultipleFields(Arc<[ColumnName]>),
}

impl StructFunction {
    pub(super) fn get_field(&self, mapper: FieldsMapper) -> PolarsResult<Field> {
        use StructFunction::*;

        match self {
            FieldByIndex(index) => mapper.try_map_field(|field| {
                let (index, _) = slice_offsets(*index, 0, mapper.get_fields_lens());
                if let DataType::Struct(ref fields) = field.dtype {
                    fields.get(index).cloned().ok_or_else(
                        || polars_err!(ComputeError: "index out of bounds in `struct.field`"),
                    )
                } else {
                    polars_bail!(
                        ComputeError: "expected struct dtype, got: `{}`", &field.dtype
                    )
                }
            }),
            FieldByName(name) => mapper.try_map_field(|field| {
                if let DataType::Struct(ref fields) = field.dtype {
                    let fld = fields
                        .iter()
                        .find(|fld| fld.name() == name.as_ref())
                        .ok_or_else(|| polars_err!(StructFieldNotFound: "{}", name.as_ref()))?;
                    Ok(fld.clone())
                } else {
                    polars_bail!(StructFieldNotFound: "{}", name.as_ref());
                }
            }),
            RenameFields(names) => mapper.map_dtype(|dt| match dt {
                DataType::Struct(fields) => {
                    let fields = fields
                        .iter()
                        .zip(names.as_ref())
                        .map(|(fld, name)| Field::new(name, fld.data_type().clone()))
                        .collect();
                    DataType::Struct(fields)
                },
                // The types will be incorrect, but its better than nothing
                // we can get an incorrect type with python lambdas, because we only know return type when running
                // the query
                dt => DataType::Struct(
                    names
                        .iter()
                        .map(|name| Field::new(name, dt.clone()))
                        .collect(),
                ),
            }),
            PrefixFields(prefix) => mapper.try_map_dtype(|dt| match dt {
                DataType::Struct(fields) => {
                    let fields = fields
                        .iter()
                        .map(|fld| {
                            let name = fld.name();
                            Field::new(&format!("{prefix}{name}"), fld.data_type().clone())
                        })
                        .collect();
                    Ok(DataType::Struct(fields))
                },
                _ => polars_bail!(op = "prefix_fields", got = dt, expected = "Struct"),
            }),
            SuffixFields(suffix) => mapper.try_map_dtype(|dt| match dt {
                DataType::Struct(fields) => {
                    let fields = fields
                        .iter()
                        .map(|fld| {
                            let name = fld.name();
                            Field::new(&format!("{name}{suffix}"), fld.data_type().clone())
                        })
                        .collect();
                    Ok(DataType::Struct(fields))
                },
                _ => polars_bail!(op = "suffix_fields", got = dt, expected = "Struct"),
            }),
            #[cfg(feature = "json")]
            JsonEncode => mapper.with_dtype(DataType::String),
            WithFields => {
                let args = mapper.args();
                let struct_ = &args[0];

                if let DataType::Struct(fields) = struct_.data_type() {
                    let mut name_2_dtype = PlIndexMap::with_capacity(fields.len() * 2);

                    for field in fields {
                        name_2_dtype.insert(field.name(), field.data_type());
                    }
                    for arg in &args[1..] {
                        name_2_dtype.insert(arg.name(), arg.data_type());
                    }
                    let dtype = DataType::Struct(
                        name_2_dtype
                            .iter()
                            .map(|(name, dtype)| Field::new(name, (*dtype).clone()))
                            .collect(),
                    );
                    let mut out = struct_.clone();
                    out.coerce(dtype);
                    Ok(out)
                } else {
                    let dt = struct_.data_type();
                    polars_bail!(op = "with_fields", got = dt, expected = "Struct")
                }
            },
            MultipleFields(_) => panic!("should be expanded"),
        }
    }
}

impl Display for StructFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use StructFunction::*;
        match self {
            FieldByIndex(index) => write!(f, "struct.field_by_index({index})"),
            FieldByName(name) => write!(f, "struct.field_by_name({name})"),
            RenameFields(names) => write!(f, "struct.rename_fields({:?})", names),
            PrefixFields(_) => write!(f, "name.prefix_fields"),
            SuffixFields(_) => write!(f, "name.suffixFields"),
            #[cfg(feature = "json")]
            JsonEncode => write!(f, "struct.to_json"),
            WithFields => write!(f, "with_fields"),
            MultipleFields(_) => write!(f, "multiple_fields"),
        }
    }
}

impl From<StructFunction> for SpecialEq<Arc<dyn SeriesUdf>> {
    fn from(func: StructFunction) -> Self {
        use StructFunction::*;
        match func {
            FieldByIndex(_) => panic!("should be replaced"),
            FieldByName(name) => map!(get_by_name, name.clone()),
            RenameFields(names) => map!(rename_fields, names.clone()),
            PrefixFields(prefix) => map!(prefix_fields, prefix.clone()),
            SuffixFields(suffix) => map!(suffix_fields, suffix.clone()),
            #[cfg(feature = "json")]
            JsonEncode => map!(to_json),
            WithFields => map_as_slice!(with_fields),
            MultipleFields(_) => unimplemented!(),
        }
    }
}

pub(super) fn get_by_name(s: &Series, name: Arc<str>) -> PolarsResult<Series> {
    let ca = s.struct_()?;
    ca.field_by_name(name.as_ref())
}

pub(super) fn rename_fields(s: &Series, names: Arc<[String]>) -> PolarsResult<Series> {
    let ca = s.struct_()?;
    let fields = ca
        .fields_as_series()
        .iter()
        .zip(names.as_ref())
        .map(|(s, name)| {
            let mut s = s.clone();
            s.rename(name);
            s
        })
        .collect::<Vec<_>>();
    let mut out = StructChunked::from_series(ca.name(), &fields)?;
    out.zip_outer_validity(ca);
    Ok(out.into_series())
}

pub(super) fn prefix_fields(s: &Series, prefix: Arc<str>) -> PolarsResult<Series> {
    let ca = s.struct_()?;
    let fields = ca
        .fields_as_series()
        .iter()
        .map(|s| {
            let mut s = s.clone();
            let name = s.name();
            s.rename(&format!("{prefix}{name}"));
            s
        })
        .collect::<Vec<_>>();
    let mut out = StructChunked::from_series(ca.name(), &fields)?;
    out.zip_outer_validity(ca);
    Ok(out.into_series())
}

pub(super) fn suffix_fields(s: &Series, suffix: Arc<str>) -> PolarsResult<Series> {
    let ca = s.struct_()?;
    let fields = ca
        .fields_as_series()
        .iter()
        .map(|s| {
            let mut s = s.clone();
            let name = s.name();
            s.rename(&format!("{name}{suffix}"));
            s
        })
        .collect::<Vec<_>>();
    let mut out = StructChunked::from_series(ca.name(), &fields)?;
    out.zip_outer_validity(ca);
    Ok(out.into_series())
}

#[cfg(feature = "json")]
pub(super) fn to_json(s: &Series) -> PolarsResult<Series> {
    let ca = s.struct_()?;
    let dtype = ca.dtype().to_arrow(CompatLevel::newest());

    let iter = ca.chunks().iter().map(|arr| {
        let arr = arrow::compute::cast::cast_unchecked(arr.as_ref(), &dtype).unwrap();
        polars_json::json::write::serialize_to_utf8(arr.as_ref())
    });

    Ok(StringChunked::from_chunk_iter(ca.name(), iter).into_series())
}

pub(super) fn with_fields(args: &[Series]) -> PolarsResult<Series> {
    let s = &args[0];

    let ca = s.struct_()?;
    let current = ca.fields_as_series();

    let mut fields = PlIndexMap::with_capacity(current.len() + s.len() - 1);

    for field in current.iter() {
        fields.insert(field.name(), field);
    }

    for field in &args[1..] {
        fields.insert(field.name(), field);
    }

    let new_fields = fields.into_values().cloned().collect::<Vec<_>>();
    let mut out = StructChunked::from_series(ca.name(), &new_fields)?;
    out.zip_outer_validity(ca);
    Ok(out.into_series())
}
