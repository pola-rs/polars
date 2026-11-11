//! Parquet has no union type, so a union is stored as a struct with one nullable member per
//! variant: a row holds its value in the member of its variant and nulls in the others. The
//! Arrow schema in the file metadata keeps the union, which tells the reader to convert back.
//!
//! Writers convert the arrays (`Series::unions_to_structs` in polars-core); this module maps the
//! dtypes and converts back on read.
use polars_arrow::array::{StructArray, UnionArray};
use polars_arrow::datatypes::{ArrowDataType, Field, UnionMode, UnionType};
use polars_error::{PolarsResult, polars_ensure};

/// The struct a union of `u` is stored as.
pub fn union_storage_dtype(u: &UnionType) -> ArrowDataType {
    let fields = u
        .fields
        .iter()
        .map(|f| Field {
            is_nullable: true,
            ..f.clone()
        })
        .collect();
    ArrowDataType::Struct(fields)
}

/// Replaces every union nested in `dtype` by its storage struct.
pub fn dtype_unions_to_structs(dtype: &ArrowDataType) -> ArrowDataType {
    use ArrowDataType as D;
    let field = |f: &Field| f.clone().with_dtype(dtype_unions_to_structs(&f.dtype));
    match dtype {
        D::Union(u) => match union_storage_dtype(u) {
            D::Struct(fields) => D::Struct(fields.iter().map(field).collect()),
            _ => unreachable!(),
        },
        D::Struct(fields) => D::Struct(fields.iter().map(field).collect()),
        D::List(f) => D::List(Box::new(field(f))),
        D::LargeList(f) => D::LargeList(Box::new(field(f))),
        D::FixedSizeList(f, width) => D::FixedSizeList(Box::new(field(f)), *width),
        D::Map(f, sorted) => D::Map(Box::new(field(f)), *sorted),
        dt => dt.clone(),
    }
}

/// Whether `dtype` holds a union, looking through structs, lists and maps.
pub fn contains_union(dtype: &ArrowDataType) -> bool {
    use ArrowDataType as D;
    match dtype {
        D::Union(_) => true,
        D::Struct(fields) => fields.iter().any(|f| contains_union(&f.dtype)),
        D::List(f) | D::LargeList(f) | D::FixedSizeList(f, _) | D::Map(f, _) => {
            contains_union(&f.dtype)
        },
        _ => false,
    }
}

/// Converts the storage struct of a union back to a union of `u`, which is sparse.
pub fn struct_to_union(arr: &StructArray, u: &UnionType) -> PolarsResult<UnionArray> {
    let members = arr.values();
    let mut types = Vec::with_capacity(arr.len());
    for i in 0..arr.len() {
        let mut valid = members.iter().enumerate().filter(|(_, m)| m.is_valid(i));
        // A null row has no value; its slot in the first member is null.
        let t = valid.next().map_or(0, |(t, _)| t);
        polars_ensure!(
            valid.next().is_none(),
            ComputeError: "row {i} of a union stored in Parquet has values for more than one variant"
        );
        types.push(t as i8);
    }
    debug_assert!(u.ids.is_none() && u.mode == UnionMode::Sparse);
    let dtype = ArrowDataType::Union(Box::new(u.clone()));
    UnionArray::try_new(dtype, types.into(), members.to_vec(), None)
}
