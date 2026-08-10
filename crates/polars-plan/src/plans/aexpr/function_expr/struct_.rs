use polars_utils::format_pl_smallstr;

use super::*;

#[derive(Clone, Eq, PartialEq, Hash, Debug)]
#[cfg_attr(feature = "ir_serde", derive(serde::Serialize, serde::Deserialize))]
pub enum IRStructFunction {
    FieldByName(PlSmallStr),
    RenameFields(Arc<[PlSmallStr]>),
    DropFields(Arc<[PlSmallStr]>, bool),
    PrefixFields(PlSmallStr),
    SuffixFields(PlSmallStr),
    #[cfg(feature = "json")]
    JsonEncode,
    MapFieldNames(PlanCallback<PlSmallStr, PlSmallStr>),
}

impl IRStructFunction {
    pub(super) fn get_field(&self, mapper: FieldsMapper) -> PolarsResult<Field> {
        use IRStructFunction::*;

        match self {
            FieldByName(name) => mapper.try_map_field(|field| {
                if let DataType::Struct(ref fields) = field.dtype {
                    let fld = fields
                        .iter()
                        .find(|fld| fld.name() == name)
                        .ok_or_else(|| polars_err!(StructFieldNotFound: "{name}"))?;
                    Ok(fld.clone())
                } else {
                    polars_bail!(StructFieldNotFound: "{name}");
                }
            }),
            RenameFields(names) => mapper.map_dtype(|dt| match dt {
                DataType::Struct(fields) => {
                    if names.len() != fields.len() {
                        let dropped_str = match fields.len() as isize - names.len() as isize {
                            1 => String::from("field of the struct"),
                            -1 => String::from("name of the argument"),
                            n if n > 0 => format!("{n} fields of the struct"),
                            n  => format!("{n} names of the argument"),
                        };
                        polars_warn!(
                            Deprecation,
                            "struct.rename_fields() argument has a different number of fields than the struct it operates on ({} vs {}).\n\
                            This silently drops the last {dropped_str}, and it will become an error in Polars 2.0.\n\
                            To replicate the old behavior and suppress this warning, use struct.drop() to drop the trailing struct fields first (if any) and then call struct.rename_fields() normally.",
                            names.len(), fields.len(),
                        )
                    }
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
            DropFields(names, strict) => mapper.try_map_dtype(|dt| match dt{
                DataType::Struct(fields)=> {
                    // Use PlIndexSets to prevent quadratic behavior
                    if *strict {
                        let fields_set = PlIndexSet::from_iter(fields.iter().map(|fld| fld.name()));
                        for name in names.iter() {
                            polars_ensure!(fields_set.contains(name), StructFieldNotFound: "{name}");
                        }
                    }
                    let names_set = PlIndexSet::from_iter(names.iter());
                    let fields = fields.iter().filter(|fld| !names_set.contains(fld.name())).cloned().collect();
                    Ok(DataType::Struct(fields))
                },
                _ => polars_bail!(op = "struct.drop", got = dt, expected = "Struct"),
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
            MapFieldNames(function) => mapper.try_map_dtype(|dt| match dt {
                DataType::Struct(fields) => {
                    let fields = fields
                        .iter()
                        .map(|fld| {
                            let name = fld.name();
                            let new_name = function.call(name.clone()).map_err(|e| polars_err!(ComputeError: "'name.map_fields' produced an error: {e}."))?;
                            Ok(Field::new(new_name, fld.dtype().clone()))
                        })
                        .collect::<PolarsResult<_>>()?;
                    Ok(DataType::Struct(fields))
                },
                _ => polars_bail!(op = "prefix_fields", got = dt, expected = "Struct"),
            }),
        }
    }

    pub fn function_options(&self) -> FunctionOptions {
        use IRStructFunction as S;
        match self {
            S::FieldByName(_) => {
                FunctionOptions::elementwise().with_flags(|f| f | FunctionFlags::ALLOW_RENAME)
            },
            S::RenameFields(_) | S::DropFields(..) | S::PrefixFields(_) | S::SuffixFields(_) => {
                FunctionOptions::elementwise()
            },
            #[cfg(feature = "json")]
            S::JsonEncode => FunctionOptions::elementwise(),
            S::MapFieldNames(_) => FunctionOptions::elementwise(),
        }
    }
}

impl Display for IRStructFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use IRStructFunction::*;
        match self {
            FieldByName(name) => write!(f, "struct.field_by_name({name})"),
            RenameFields(names) => write!(f, "struct.rename_fields({names:?})"),
            DropFields(names, false) => write!(f, "struct.drop({names:?}, strict=False)"),
            DropFields(names, true) => write!(f, "struct.drop({names:?})"),
            PrefixFields(_) => write!(f, "name.prefix_fields"),
            SuffixFields(_) => write!(f, "name.suffixFields"),
            #[cfg(feature = "json")]
            JsonEncode => write!(f, "struct.to_json"),
            MapFieldNames(_) => write!(f, "map_field_names"),
        }
    }
}
