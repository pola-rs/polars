use polars_core::utils::slice_offsets;

use super::*;

#[derive(Clone, Eq, PartialEq, Hash, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum StructFunction {
    FieldByIndex(i64),
    FieldByName(Arc<str>),
}

impl Display for StructFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use self::*;
        match self {
            StructFunction::FieldByIndex(_) => write!(f, "struct.field_by_name"),
            StructFunction::FieldByName(_) => write!(f, "struct.field_by_index"),
        }
    }
}

pub(super) fn get_by_index(s: &Series, index: i64) -> PolarsResult<Series> {
    let s = s.struct_()?;
    let (index, _) = slice_offsets(index, 0, s.fields().len());
    s.fields()
        .get(index)
        .cloned()
        .ok_or_else(|| PolarsError::ComputeError("index out of bounds in 'struct.field'".into()))
}
pub(super) fn get_by_name(s: &Series, name: Arc<str>) -> PolarsResult<Series> {
    let ca = s.struct_()?;
    ca.field_by_name(name.as_ref())
}
