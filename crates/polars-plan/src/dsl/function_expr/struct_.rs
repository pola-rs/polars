use super::*;

#[derive(Clone, Eq, PartialEq, Hash, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum StructFunction {
    FieldByName(PlSmallStr),
    RenameFields(Arc<[PlSmallStr]>),
    PrefixFields(PlSmallStr),
    SuffixFields(PlSmallStr),
    #[cfg(feature = "json")]
    JsonEncode,
    WithFields,
    SelectFields(Selector),
    #[cfg(feature = "python")]
    MapFieldNames(SpecialEq<Arc<polars_utils::python_function::PythonObject>>),
}

impl Display for StructFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use StructFunction::*;
        match self {
            FieldByName(name) => write!(f, "struct.field_by_name({name})"),
            RenameFields(names) => write!(f, "struct.rename_fields({names:?})"),
            PrefixFields(_) => write!(f, "name.prefix_fields"),
            SuffixFields(_) => write!(f, "name.suffixFields"),
            #[cfg(feature = "json")]
            JsonEncode => write!(f, "struct.to_json"),
            WithFields => write!(f, "struct.with_fields"),
            SelectFields(_) => write!(f, "struct.field"),
            #[cfg(feature = "python")]
            MapFieldNames(_) => write!(f, "map_field_names"),
        }
    }
}
