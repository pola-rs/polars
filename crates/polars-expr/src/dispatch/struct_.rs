use std::sync::Arc;

use polars_core::error::{PolarsResult, polars_err};
use polars_core::prelude::{
    Column, InitHashMaps, IntoColumn, PlIndexMap, StringChunked, StructChunked,
};
use polars_plan::dsl::{ColumnsUdf, SpecialEq};
use polars_plan::plans::IRStructFunction;
use polars_plan::prelude::PlanCallback;
use polars_utils::format_pl_smallstr;
use polars_utils::pl_str::PlSmallStr;

pub fn function_expr_to_udf(func: IRStructFunction) -> SpecialEq<Arc<dyn ColumnsUdf>> {
    use IRStructFunction::*;
    match func {
        FieldByName(name) => map!(get_by_name, &name),
        RenameFields(names) => map!(rename_fields, names.clone()),
        PrefixFields(prefix) => map!(prefix_fields, prefix.as_str()),
        SuffixFields(suffix) => map!(suffix_fields, suffix.as_str()),
        #[cfg(feature = "json")]
        JsonEncode => map!(to_json),
        WithFields => map_as_slice!(with_fields),
        MapFieldNames(function) => map!(map_field_names, &function),
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
    use polars_core::prelude::CompatLevel;

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

pub(super) fn map_field_names(
    s: &Column,
    function: &PlanCallback<PlSmallStr, PlSmallStr>,
) -> PolarsResult<Column> {
    let ca = s.struct_()?;
    let fields = ca
        .fields_as_series()
        .iter()
        .map(|s| {
            let mut s = s.clone();
            let name = s.name();
            let new_name = function.call(name.clone()).map_err(
                |e| polars_err!(ComputeError: "'name.map_fields' produced an error: {e}."),
            )?;
            s.rename(new_name);
            Ok(s)
        })
        .collect::<PolarsResult<Vec<_>>>()?;
    let mut out = StructChunked::from_series(ca.name().clone(), ca.len(), fields.iter())?;
    out.zip_outer_validity(ca);
    Ok(out.into_column())
}
