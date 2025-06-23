use core::fmt;
use std::fmt::Write;

use polars_core::error::{PolarsResult, polars_bail, polars_ensure};
use polars_core::prelude::{DataType, Field, InitHashMaps, PlHashSet};
use polars_core::schema::Schema;
use polars_utils::arena::Arena;
use polars_utils::pl_str::PlSmallStr;

use super::{
    ArrayDataTypeFunction, DataTypeFunction, DataTypeKind, EnumDataTypeFunction, Expr,
    StructDataTypeFunction,
};
use crate::plans::to_expr_ir;

#[derive(Clone, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum DataTypeExpr {
    Literal(DataType),
    OfExpr(Box<Expr>),

    InnerDataType {
        input: Box<DataTypeExpr>,
        validation: Option<SequenceKind>,
    },

    Int(Box<DataTypeExpr>, IntDataTypeExpr),
    Struct(Box<DataTypeExpr>, StructDataTypeExpr),

    // Constructors for nested types
    WrapInList(Box<DataTypeExpr>),
    WrapInArray(Box<DataTypeExpr>, usize),
    StructWithFields(Vec<(PlSmallStr, DataTypeExpr)>),
}

#[derive(Clone, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum SequenceKind {
    List,
    Array,
}

#[derive(PartialEq, Clone, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum IntDataTypeExpr {
    ToUnsigned,
    ToSigned,
}

#[derive(PartialEq, Clone, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum StructDataTypeExpr {
    FieldDataTypeByIndex(i64),
    FieldDataTypeByName(PlSmallStr),
}

#[recursive::recursive]
fn into_datatype_impl(dt_expr: DataTypeExpr, schema: &Schema) -> PolarsResult<DataType> {
    use DataTypeExpr as D;
    let dtype = match dt_expr {
        D::Literal(dt) => dt,
        D::OfExpr(expr) => {
            let mut arena = Arena::new();
            let e = to_expr_ir(*expr, &mut arena, schema)?;
            let dtype = arena
                .get(e.node())
                .to_dtype(schema, Default::default(), &arena)?;
            polars_ensure!(!dtype.contains_unknown(),InvalidOperation:"DataType expression is not allowed to instantiate to `unknown`");
            dtype
        },
        D::InnerDataType { input, validation } => {
            let dt = input.into_datatype(schema)?;
            let Some(validation) = validation else {
                return dt.try_into_inner_dtype();
            };

            match (dt, validation) {
                (DataType::Array(inner, _), SequenceKind::Array) => *inner,
                (DataType::List(inner), SequenceKind::List) => *inner,
                (dt, SequenceKind::Array) => {
                    polars_bail!(SchemaMismatch: "expected `arr` type but got `{dt}`")
                },
                (dt, SequenceKind::List) => {
                    polars_bail!(SchemaMismatch: "expected `list` type but got `{dt}`")
                },
            }
        },
        D::Int(dt_expr, f) => {
            use DataType as DT;
            let dt = dt_expr.into_datatype(schema)?;
            polars_ensure!(dt.is_integer(), InvalidOperation: "`{dt}` is not an integer type");
            match f {
                IntDataTypeExpr::ToUnsigned => match dt {
                    DT::UInt8 | DT::Int8 => DT::UInt8,
                    DT::UInt16 | DT::Int16 => DT::UInt16,
                    DT::UInt32 | DT::Int32 => DT::UInt32,
                    DT::UInt64 | DT::Int64 => DT::UInt64,
                    DT::Int128 => {
                        polars_bail!(InvalidOperation: "`int128` has no unsigned equivalent")
                    },
                    _ => unreachable!(),
                },
                IntDataTypeExpr::ToSigned => {
                    use DataType as DT;
                    match dt {
                        DT::UInt8 | DT::Int8 => DT::Int8,
                        DT::UInt16 | DT::Int16 => DT::Int16,
                        DT::UInt32 | DT::Int32 => DT::Int32,
                        DT::UInt64 | DT::Int64 => DT::Int64,
                        DT::Int128 => DT::Int128,
                        _ => unreachable!(),
                    }
                },
            }
        },
        D::Struct(dt_expr, f) => {
            let fields = match dt_expr.into_datatype(schema)? {
                DataType::Struct(fields) => fields,
                dt => polars_bail!(InvalidOperation: "`{dt}` is not a `struct`"),
            };
            match f {
                StructDataTypeExpr::FieldDataTypeByIndex(idx) => {
                    let offset = if idx < 0 {
                        let offset = usize::try_from(idx.abs_diff(0)).unwrap();
                        polars_ensure!(
                            offset <= fields.len(),
                            InvalidOperation: "`struct` has {} fields, but field {idx} was requested",
                            fields.len()
                        );
                        fields.len() - offset
                    } else {
                        let offset = usize::try_from(idx).unwrap();
                        polars_ensure!(
                            offset < fields.len(),
                            InvalidOperation: "`struct` has {} fields, but field {idx} was requested",
                            fields.len()
                        );
                        offset
                    };

                    fields.into_iter().nth(offset).unwrap().dtype
                },
                StructDataTypeExpr::FieldDataTypeByName(name) => {
                    let Some(field) = fields.into_iter().find(|f| f.name() == &name) else {
                        polars_bail!(
                            InvalidOperation: "`struct` does not have field '{name}'",
                        );
                    };
                    field.dtype
                },
            }
        },
        D::WrapInList(dt_expr) => DataType::List(Box::new(dt_expr.into_datatype(schema)?)),
        D::WrapInArray(dt_expr, width) => {
            DataType::Array(Box::new(dt_expr.into_datatype(schema)?), width)
        },
        D::StructWithFields(field_exprs) => {
            let mut seen = PlHashSet::with_capacity(field_exprs.len());
            let mut fields = Vec::with_capacity(field_exprs.len());
            for (name, dt_expr) in field_exprs {
                let dt = dt_expr.into_datatype(schema)?;
                if !seen.insert(name.clone()) {
                    polars_bail!(
                        InvalidOperation:
                        "`struct` cannot have duplicate field name `{name}`"
                    );
                }
                fields.push(Field::new(name, dt));
            }
            DataType::Struct(fields)
        },
    };

    Ok(dtype)
}

impl DataTypeExpr {
    pub fn into_datatype(self, schema: &Schema) -> PolarsResult<DataType> {
        into_datatype_impl(self, schema)
    }

    pub fn as_literal(&self) -> Option<&DataType> {
        match self {
            Self::Literal(dt) => Some(dt),
            _ => None,
        }
    }

    pub fn into_literal(self) -> Option<DataType> {
        match self {
            Self::Literal(dt) => Some(dt),
            _ => None,
        }
    }

    pub fn inner_dtype(self) -> Self {
        Self::InnerDataType {
            input: Box::new(self),
            validation: None,
        }
    }

    pub fn equals(self, other: Self) -> Expr {
        Expr::DataTypeFunction(DataTypeFunction::Eq(self, other))
    }

    pub fn to_string(self) -> Expr {
        Expr::DataTypeFunction(DataTypeFunction::ToString(self))
    }

    pub fn element_bitsize(self) -> Expr {
        Expr::DataTypeFunction(DataTypeFunction::ElementBitSize(self))
    }

    pub fn is_kind(self, kind: DataTypeKind) -> Expr {
        Expr::DataTypeFunction(DataTypeFunction::IsKind(self, kind))
    }

    pub fn is_numeric(self) -> Expr {
        self.is_kind(DataTypeKind::Numeric)
    }

    pub fn is_integer(self) -> Expr {
        self.is_kind(DataTypeKind::Integer)
    }

    pub fn is_float(self) -> Expr {
        self.is_kind(DataTypeKind::Float)
    }

    pub fn is_decimal(self) -> Expr {
        self.is_kind(DataTypeKind::Decimal)
    }

    pub fn is_categorical(self) -> Expr {
        self.is_kind(DataTypeKind::Categorical)
    }

    pub fn is_enum(self) -> Expr {
        self.is_kind(DataTypeKind::Enum)
    }

    pub fn is_nested(self) -> Expr {
        self.is_kind(DataTypeKind::Nested)
    }

    pub fn is_list(self) -> Expr {
        self.is_kind(DataTypeKind::List)
    }

    pub fn is_array(self) -> Expr {
        self.is_kind(DataTypeKind::Array(None))
    }

    pub fn is_struct(self) -> Expr {
        self.is_kind(DataTypeKind::Struct)
    }

    pub fn is_temporal(self) -> Expr {
        self.is_kind(DataTypeKind::Temporal)
    }

    pub fn is_datetime(self) -> Expr {
        self.is_kind(DataTypeKind::Datetime)
    }

    pub fn is_duration(self) -> Expr {
        self.is_kind(DataTypeKind::Duration)
    }

    pub fn is_object(self) -> Expr {
        self.is_kind(DataTypeKind::Object)
    }

    pub fn wrap_in_list(self) -> Self {
        Self::WrapInList(Box::new(self))
    }

    pub fn wrap_in_array(self, width: usize) -> Self {
        Self::WrapInArray(Box::new(self), width)
    }

    pub fn int(self) -> DataTypeExprIntNameSpace {
        DataTypeExprIntNameSpace(self)
    }

    pub fn enum_(self) -> DataTypeExprEnumNameSpace {
        DataTypeExprEnumNameSpace(self)
    }

    pub fn list(self) -> DataTypeExprListNameSpace {
        DataTypeExprListNameSpace(self)
    }

    pub fn arr(self) -> DataTypeExprArrNameSpace {
        DataTypeExprArrNameSpace(self)
    }

    pub fn struct_(self) -> DataTypeExprStructNameSpace {
        DataTypeExprStructNameSpace(self)
    }
}

impl fmt::Debug for DataTypeExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Literal(data_type) => data_type.fmt(f),
            Self::OfExpr(expr) => write!(f, "dtype_of({expr:?})"),
            Self::InnerDataType { input, validation } => {
                fmt::Debug::fmt(input.as_ref(), f)?;
                match validation {
                    None => {},
                    Some(SequenceKind::List) => f.write_str(".list")?,
                    Some(SequenceKind::Array) => f.write_str(".arr")?,
                }
                f.write_str(".inner()")
            },
            Self::Int(dt_expr, t) => {
                fmt::Debug::fmt(dt_expr.as_ref(), f)?;
                f.write_str(".int")?;
                match t {
                    IntDataTypeExpr::ToUnsigned => f.write_str(".to_unsigned()"),
                    IntDataTypeExpr::ToSigned => f.write_str(".to_signed()"),
                }
            },
            Self::Struct(dt_expr, t) => {
                fmt::Debug::fmt(dt_expr.as_ref(), f)?;
                f.write_str(".struct")?;
                match t {
                    StructDataTypeExpr::FieldDataTypeByIndex(i) => {
                        write!(f, "[{i}]")
                    },
                    StructDataTypeExpr::FieldDataTypeByName(name) => {
                        write!(f, "[{name}]")
                    },
                }
            },
            Self::WrapInList(dt_expr) => {
                write!(f, "{dt_expr:?}.wrap_in_list()")
            },
            Self::WrapInArray(dt_expr, width) => {
                write!(f, "{dt_expr:?}.wrap_in_array(width={width})")
            },
            Self::StructWithFields(field_exprs) => {
                f.write_str("struct_with_fields({")?;
                if let Some((field_name, field_expr)) = field_exprs.first() {
                    write!(f, " {field_name}: {field_expr:?}")?;
                    for (field_name, field_expr) in &field_exprs[1..] {
                        write!(f, ", {field_name}: {field_expr:?}")?;
                    }
                    f.write_char(' ')?;
                }
                f.write_str("})")
            },
        }
    }
}

impl From<DataType> for DataTypeExpr {
    fn from(value: DataType) -> Self {
        Self::Literal(value)
    }
}

pub struct DataTypeExprIntNameSpace(DataTypeExpr);
pub struct DataTypeExprEnumNameSpace(DataTypeExpr);
pub struct DataTypeExprListNameSpace(DataTypeExpr);
pub struct DataTypeExprArrNameSpace(DataTypeExpr);
pub struct DataTypeExprStructNameSpace(DataTypeExpr);

impl DataTypeExprIntNameSpace {
    #[expect(clippy::wrong_self_convention)]
    pub fn to_unsigned(self) -> DataTypeExpr {
        DataTypeExpr::Int(Box::new(self.0), IntDataTypeExpr::ToUnsigned)
    }

    #[expect(clippy::wrong_self_convention)]
    pub fn to_signed(self) -> DataTypeExpr {
        DataTypeExpr::Int(Box::new(self.0), IntDataTypeExpr::ToSigned)
    }

    #[expect(clippy::wrong_self_convention)]
    pub fn is_unsigned(self) -> Expr {
        self.0.is_kind(DataTypeKind::UnsignedInteger)
    }

    #[expect(clippy::wrong_self_convention)]
    pub fn is_signed(self) -> Expr {
        self.0.is_kind(DataTypeKind::SignedInteger)
    }
}

impl DataTypeExprEnumNameSpace {
    pub fn num_categories(self) -> Expr {
        Expr::DataTypeFunction(DataTypeFunction::Enum(
            self.0,
            EnumDataTypeFunction::NumCategories,
        ))
    }

    pub fn categories(self) -> Expr {
        Expr::DataTypeFunction(DataTypeFunction::Enum(
            self.0,
            EnumDataTypeFunction::Categories,
        ))
    }

    pub fn get_category(self, index: i64, raise: bool) -> Expr {
        Expr::DataTypeFunction(DataTypeFunction::Enum(
            self.0,
            EnumDataTypeFunction::GetCategory {
                idx: index,
                raise_on_oob: raise,
            },
        ))
    }

    pub fn index_of_category(self, value: &str, raise: bool) -> Expr {
        Expr::DataTypeFunction(DataTypeFunction::Enum(
            self.0,
            EnumDataTypeFunction::IndexOfCategory {
                cat: value.into(),
                raise_on_missing: raise,
            },
        ))
    }
}

impl DataTypeExprListNameSpace {
    pub fn inner_dtype(self) -> DataTypeExpr {
        DataTypeExpr::InnerDataType {
            input: Box::new(self.0),
            validation: Some(SequenceKind::List),
        }
    }
}

impl DataTypeExprArrNameSpace {
    pub fn inner_dtype(self) -> DataTypeExpr {
        DataTypeExpr::InnerDataType {
            input: Box::new(self.0),
            validation: Some(SequenceKind::Array),
        }
    }

    pub fn has_width(self, width: usize) -> Expr {
        self.0.is_kind(DataTypeKind::Array(Some(width)))
    }

    pub fn width(self) -> Expr {
        Expr::DataTypeFunction(DataTypeFunction::Array(
            self.0,
            ArrayDataTypeFunction::Width,
        ))
    }

    pub fn dimensions(self) -> Expr {
        Expr::DataTypeFunction(DataTypeFunction::Array(
            self.0,
            ArrayDataTypeFunction::Dimensions,
        ))
    }
}

impl DataTypeExprStructNameSpace {
    pub fn field_dtype_by_index(self, index: i64) -> DataTypeExpr {
        DataTypeExpr::Struct(
            Box::new(self.0),
            StructDataTypeExpr::FieldDataTypeByIndex(index),
        )
    }

    pub fn field_dtype_by_name(self, name: &str) -> DataTypeExpr {
        DataTypeExpr::Struct(
            Box::new(self.0),
            StructDataTypeExpr::FieldDataTypeByName(name.into()),
        )
    }

    pub fn num_fields(self) -> Expr {
        Expr::DataTypeFunction(DataTypeFunction::Struct(
            self.0,
            StructDataTypeFunction::NumFields,
        ))
    }

    pub fn field_names(self) -> Expr {
        Expr::DataTypeFunction(DataTypeFunction::Struct(
            self.0,
            StructDataTypeFunction::FieldNames,
        ))
    }

    pub fn field_name(self, index: i64, raise_on_oob: bool) -> Expr {
        Expr::DataTypeFunction(DataTypeFunction::Struct(
            self.0,
            StructDataTypeFunction::FieldName {
                idx: index,
                raise_on_oob,
            },
        ))
    }

    pub fn field_index(self, name: &str, raise_on_missing: bool) -> Expr {
        Expr::DataTypeFunction(DataTypeFunction::Struct(
            self.0,
            StructDataTypeFunction::FieldIndex {
                name: name.into(),
                raise_on_missing,
            },
        ))
    }
}
