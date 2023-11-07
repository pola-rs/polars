use polars_core::utils::slice_offsets;
use polars_utils::format_smartstring;

use super::*;
use crate::map;

#[derive(Clone, Eq, PartialEq, Hash, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum StructFunction {
    FieldByIndex(i64),
    FieldByName(Arc<str>),
    RenameFields(Arc<Vec<String>>),
    Prefix(Arc<str>),
    Suffix(Arc<str>),
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
            Prefix(prefix) => mapper.map_dtype(|dt| match dt {
                DataType::Struct(fields) => {
                    let fields = fields
                        .iter()
                        .map(|fld| {
                            let name = &format_smartstring!("{}{}", prefix, fld.name());
                            Field::new(name, fld.data_type().clone())
                        })
                        .collect();
                    DataType::Struct(fields)
                },
                // The types will be incorrect, but its better than nothing
                // we can get an incorrect type with python lambdas, because we only know return type when running
                // the query
                //
                //
                _ => dt.clone(),
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
            Suffix(suffix) => mapper.map_dtype(|dt| match dt {
                DataType::Struct(fields) => {
                    let fields = fields
                        .iter()
                        .map(|fld| {
                            let name = &format_smartstring!("{}{}", fld.name(), suffix);
                            Field::new(name, fld.data_type().clone())
                        })
                        .collect();
                    DataType::Struct(fields)
                },
                // The types will be incorrect, but its better than nothing
                // we can get an incorrect type with python lambdas, because we only know return type when running
                // the query
                _ => dt.clone(),
            }),
        }
    }
}

impl Display for StructFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use StructFunction::*;
        match self {
            FieldByIndex(index) => write!(f, "struct.field_by_index({index})"),
            FieldByName(name) => write!(f, "struct.field_by_name({name})"),
            Prefix(prefix) => write!(f, "struct.prefix({:?})", prefix),
            RenameFields(names) => write!(f, "struct.rename_fields({:?})", names),
            Suffix(suffix) => write!(f, "struct.suffix({:?})", suffix),
        }
    }
}

impl From<StructFunction> for SpecialEq<Arc<dyn SeriesUdf>> {
    fn from(func: StructFunction) -> Self {
        use StructFunction::*;
        match func {
            FieldByIndex(index) => map!(struct_::get_by_index, index),
            FieldByName(name) => map!(struct_::get_by_name, name.clone()),
            Prefix(prefix) => map!(struct_::prefix, prefix.clone()),
            RenameFields(names) => map!(struct_::rename_fields, names.clone()),
            Suffix(suffix) => map!(struct_::suffix, suffix.clone()),
        }
    }
}

pub(super) fn get_by_index(s: &Series, index: i64) -> PolarsResult<Series> {
    let s = s.struct_()?;
    let (index, _) = slice_offsets(index, 0, s.fields().len());
    s.fields()
        .get(index)
        .cloned()
        .ok_or_else(|| polars_err!(ComputeError: "struct field index out of bounds"))
}
pub(super) fn get_by_name(s: &Series, name: Arc<str>) -> PolarsResult<Series> {
    let ca = s.struct_()?;
    ca.field_by_name(name.as_ref())
}

pub(super) fn prefix(s: &Series, prefix: Arc<str>) -> PolarsResult<Series> {
    let ca = s.struct_()?;
    let fields = ca
        .fields()
        .iter()
        .map(|fld| {
            let mut fld = fld.clone();
            let name = &format_smartstring!("{}{}", prefix, fld.name());
            fld.rename(name);
            fld
        })
        .collect::<Vec<_>>();
    StructChunked::new(ca.name(), &fields).map(|ca| ca.into_series())
}

pub(super) fn rename_fields(s: &Series, names: Arc<Vec<String>>) -> PolarsResult<Series> {
    let ca = s.struct_()?;
    let fields = ca
        .fields()
        .iter()
        .zip(names.as_ref())
        .map(|(fld, name)| {
            let mut fld = fld.clone();
            fld.rename(name);
            fld
        })
        .collect::<Vec<_>>();
    StructChunked::new(ca.name(), &fields).map(|ca| ca.into_series())
}

pub(super) fn suffix(s: &Series, suffix: Arc<str>) -> PolarsResult<Series> {
    let ca = s.struct_()?;
    let fields = ca
        .fields()
        .iter()
        .map(|fld| {
            let mut fld = fld.clone();
            let name = &format_smartstring!("{}{}", fld.name(), suffix);
            fld.rename(name);
            fld
        })
        .collect::<Vec<_>>();
    StructChunked::new(ca.name(), &fields).map(|ca| ca.into_series())
}
