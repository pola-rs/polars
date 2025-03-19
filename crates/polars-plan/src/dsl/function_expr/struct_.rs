use polars_core::utils::slice_offsets;
use polars_utils::format_pl_smallstr;

use super::*;
use crate::{map, map_as_slice};

#[derive(Clone, Eq, PartialEq, Hash, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum StructFunction {
    FieldByIndex(i64),
    FieldByName(PlSmallStr),
    RenameFields(Arc<[PlSmallStr]>),
    PrefixFields(PlSmallStr),
    SuffixFields(PlSmallStr),
    #[cfg(feature = "json")]
    JsonEncode,
    WithFields,
    MultipleFields(Arc<[PlSmallStr]>),
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
                        .find(|fld| fld.name() == name)
                        .ok_or_else(|| polars_err!(StructFieldNotFound: "{}", name))?;
                    Ok(fld.clone())
                } else {
                    polars_bail!(StructFieldNotFound: "{}", name);
                }
            }),
            RenameFields(names) => mapper.map_dtype(|dt| match dt {
                DataType::Struct(fields) => {
                    let fields = fields
                        .iter()
                        .zip(names.as_ref())
                        .map(|(fld, name)| Field::new(name.clone(), fld.dtype().clone()))
                        .collect();
                    DataType::Struct(fields)
                },
                // The types will be incorrect, but its better than nothing
                // we can get an incorrect type with python lambdas, because we only know return type when running
                // the query
                dt => DataType::Struct(
                    names
                        .iter()
                        .map(|name| Field::new(name.clone(), dt.clone()))
                        .collect(),
                ),
            }),
            PrefixFields(prefix) => mapper.try_map_dtype(|dt| match dt {
                DataType::Struct(fields) => {
                    let fields = fields
                        .iter()
                        .map(|fld| {
                            let name = fld.name();
                            Field::new(format_pl_smallstr!("{prefix}{name}"), fld.dtype().clone())
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
                            Field::new(format_pl_smallstr!("{name}{suffix}"), fld.dtype().clone())
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

                if let DataType::Struct(fields) = struct_.dtype() {
                    let mut name_2_dtype = PlIndexMap::with_capacity(fields.len() * 2);

                    for field in fields {
                        name_2_dtype.insert(field.name(), field.dtype());
                    }
                    for arg in &args[1..] {
                        name_2_dtype.insert(arg.name(), arg.dtype());
                    }
                    let dtype = DataType::Struct(
                        name_2_dtype
                            .iter()
                            .map(|(&name, &dtype)| Field::new(name.clone(), dtype.clone()))
                            .collect(),
                    );
                    let mut out = struct_.clone();
                    out.coerce(dtype);
                    Ok(out)
                } else {
                    let dt = struct_.dtype();
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

impl From<StructFunction> for SpecialEq<Arc<dyn ColumnsUdf>> {
    fn from(func: StructFunction) -> Self {
        use StructFunction::*;
        match func {
            FieldByIndex(_) => panic!("should be replaced"),
            FieldByName(name) => map!(get_by_name, &name),
            RenameFields(names) => map!(rename_fields, names.clone()),
            PrefixFields(prefix) => map!(prefix_fields, prefix.as_str()),
            SuffixFields(suffix) => map!(suffix_fields, suffix.as_str()),
            #[cfg(feature = "json")]
            JsonEncode => map!(to_json),
            WithFields => map_as_slice!(with_fields),
            MultipleFields(_) => unimplemented!(),
        }
    }
}

pub(super) fn get_by_name(s: &Column, name: &str) -> PolarsResult<Column> {
    let ca = s.struct_()?;
    ca.field_by_name(name).map(Column::from)
}

pub(super) fn rename_fields(s: &Column, names: Arc<[PlSmallStr]>) -> PolarsResult<Column> {
    let ca = s.struct_()?;
    let fields = ca
        .fields_as_series()
        .iter()
        .zip(names.as_ref())
        .map(|(s, name)| {
            let mut s = s.clone();
            s.rename(name.clone());
            s
        })
        .collect::<Vec<_>>();
    let mut out = StructChunked::from_series(ca.name().clone(), ca.len(), fields.iter())?;
    out.zip_outer_validity(ca);
    Ok(out.into_column())
}

pub(super) fn prefix_fields(s: &Column, prefix: &str) -> PolarsResult<Column> {
    let ca = s.struct_()?;
    let fields = ca
        .fields_as_series()
        .iter()
        .map(|s| {
            let mut s = s.clone();
            let name = s.name();
            s.rename(format_pl_smallstr!("{prefix}{name}"));
            s
        })
        .collect::<Vec<_>>();
    let mut out = StructChunked::from_series(ca.name().clone(), ca.len(), fields.iter())?;
    out.zip_outer_validity(ca);
    Ok(out.into_column())
}

pub(super) fn suffix_fields(s: &Column, suffix: &str) -> PolarsResult<Column> {
    let ca = s.struct_()?;
    let fields = ca
        .fields_as_series()
        .iter()
        .map(|s| {
            let mut s = s.clone();
            let name = s.name();
            s.rename(format_pl_smallstr!("{name}{suffix}"));
            s
        })
        .collect::<Vec<_>>();
    let mut out = StructChunked::from_series(ca.name().clone(), ca.len(), fields.iter())?;
    out.zip_outer_validity(ca);
    Ok(out.into_column())
}

#[cfg(feature = "json")]
pub(super) fn to_json(s: &Column) -> PolarsResult<Column> {
    let ca = s.struct_()?;
    let dtype = ca.dtype().to_arrow(CompatLevel::newest());

    let iter = ca.chunks().iter().map(|arr| {
        let arr = polars_compute::cast::cast_unchecked(arr.as_ref(), &dtype).unwrap();
        polars_json::json::write::serialize_to_utf8(arr.as_ref())
    });

    Ok(StringChunked::from_chunk_iter(ca.name().clone(), iter).into_column())
}

pub(super) fn with_fields(args: &[Column]) -> PolarsResult<Column> {
    let s = &args[0];

    let ca = s.struct_()?;
    let current = ca.fields_as_series();

    let mut fields = PlIndexMap::with_capacity(current.len() + s.len() - 1);

    for field in current.iter() {
        fields.insert(field.name(), field);
    }

    for field in &args[1..] {
        fields.insert(field.name(), field.as_materialized_series());
    }

    let new_fields = fields.into_values().cloned().collect::<Vec<_>>();
    let mut out = StructChunked::from_series(ca.name().clone(), ca.len(), new_fields.iter())?;
    out.zip_outer_validity(ca);
    Ok(out.into_column())
}
