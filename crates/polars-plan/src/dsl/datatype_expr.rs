use core::fmt;
use std::fmt::Write;

use polars_core::error::{PolarsResult, feature_gated, polars_bail, polars_ensure};
use polars_core::prelude::{DataType, Field};
use polars_core::schema::Schema;
use polars_utils::arena::Arena;
use polars_utils::pl_str::PlSmallStr;

use super::{
    ArrayDataTypeFunction, DataTypeFunction, DataTypeSelector, Expr, StructDataTypeFunction,
};
use crate::frame::OptFlags;
use crate::plans::{ExprToIRContext, ToFieldContext, expand_expression, to_expr_ir};

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

    /// Invariant, must be directly materialized in `map_elements/map_batches`
    /// After materialization it becomes `OfExpr<self>`
    SelfDtype,
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
fn into_datatype_impl(
    dt_expr: DataTypeExpr,
    schema: &Schema,
    self_dtype: Option<&DataType>,
) -> PolarsResult<DataType> {
    use DataTypeExpr as D;
    let dtype = match dt_expr {
        D::Literal(dt) => dt,
        D::OfExpr(expr) => {
            let mut out = Vec::with_capacity(1);
            expand_expression(
                expr.as_ref(),
                &Default::default(),
                schema,
                &mut out,
                &mut OptFlags::default(),
            )?;
            polars_ensure!(
                out.len() == 1,
                InvalidOperation: "DataType expression are not allowed to expand to more than 1 expression"
            );

            let expr = out.pop().unwrap();
            let mut arena = Arena::new();
            let mut ctx = ExprToIRContext::new(&mut arena, schema);
            let e = to_expr_ir(expr, &mut ctx)?;
            let dtype = arena
                .get(e.node())
                .to_dtype(&ToFieldContext::new(&arena, schema))?;
            let dtype = dtype.materialize_unknown(true)?;
            polars_ensure!(!dtype.contains_unknown(),InvalidOperation:"DataType expression is not allowed to instantiate to `unknown`");
            dtype
        },
        D::SelfDtype => match self_dtype {
            None => polars_bail!(
                InvalidOperation: "'self_dtype' cannot be used in this context",
            ),
            Some(self_dtype) => self_dtype.clone(),
        },
        D::InnerDataType { input, validation } => {
            let dt = into_datatype_impl(*input, schema, self_dtype)?;
            let Some(validation) = validation else {
                return dt.try_into_inner_dtype();
            };

            match (dt, validation) {
                #[cfg(feature = "dtype-array")]
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
            let dt = into_datatype_impl(*dt_expr, schema, self_dtype)?;
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
            let fields: Vec<Field> = match into_datatype_impl(*dt_expr, schema, self_dtype)? {
                #[cfg(feature = "dtype-struct")]
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
        D::WrapInList(dt_expr) => {
            DataType::List(Box::new(into_datatype_impl(*dt_expr, schema, self_dtype)?))
        },
        D::WrapInArray(dt_expr, width) => feature_gated!("dtype-array", {
            DataType::Array(
                Box::new(into_datatype_impl(*dt_expr, schema, self_dtype)?),
                width,
            )
        }),
        D::StructWithFields(field_exprs) => feature_gated!("dtype-struct", {
            use polars_core::prelude::{Field, InitHashMaps, PlHashSet};
            let mut seen = PlHashSet::with_capacity(field_exprs.len());
            let mut fields = Vec::with_capacity(field_exprs.len());
            for (name, dt_expr) in field_exprs {
                let dt = into_datatype_impl(dt_expr, schema, self_dtype)?;
                if !seen.insert(name.clone()) {
                    polars_bail!(
                        InvalidOperation:
                        "`struct` cannot have duplicate field name `{name}`"
                    );
                }
                fields.push(Field::new(name, dt));
            }
            DataType::Struct(fields)
        }),
    };

    Ok(dtype)
}

impl DataTypeExpr {
    pub fn into_datatype(self, schema: &Schema) -> PolarsResult<DataType> {
        self.into_datatype_with_opt_self(schema, None)
    }

    pub fn into_datatype_with_self(
        self,
        schema: &Schema,
        self_dtype: &DataType,
    ) -> PolarsResult<DataType> {
        self.into_datatype_with_opt_self(schema, Some(self_dtype))
    }

    pub fn into_datatype_with_opt_self(
        self,
        schema: &Schema,
        self_dtype: Option<&DataType>,
    ) -> PolarsResult<DataType> {
        into_datatype_impl(self, schema, self_dtype)
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

    pub fn display(self) -> Expr {
        Expr::DataTypeFunction(DataTypeFunction::Display(self))
    }

    pub fn matches(self, selector: DataTypeSelector) -> Expr {
        Expr::DataTypeFunction(DataTypeFunction::Matches(self, selector))
    }

    pub fn wrap_in_list(self) -> Self {
        Self::WrapInList(Box::new(self))
    }

    pub fn wrap_in_array(self, width: usize) -> Self {
        Self::WrapInArray(Box::new(self), width)
    }

    pub fn default_value(self, n: usize, numeric_to_one: bool, num_list_values: usize) -> Expr {
        Expr::DataTypeFunction(DataTypeFunction::DefaultValue {
            dt_expr: self,
            n,
            numeric_to_one,
            num_list_values,
        })
    }

    pub fn int(self) -> DataTypeExprIntNameSpace {
        DataTypeExprIntNameSpace(self)
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
            Self::SelfDtype => write!(f, "self_dtype()"),
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
                match t {
                    IntDataTypeExpr::ToUnsigned => f.write_str(".to_unsigned_integer()"),
                    IntDataTypeExpr::ToSigned => f.write_str(".to_signed_integer()"),
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

    pub fn width(self) -> Expr {
        Expr::DataTypeFunction(DataTypeFunction::Array(
            self.0,
            ArrayDataTypeFunction::Width,
        ))
    }

    pub fn shape(self) -> Expr {
        Expr::DataTypeFunction(DataTypeFunction::Array(
            self.0,
            ArrayDataTypeFunction::Shape,
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

    pub fn field_names(self) -> Expr {
        Expr::DataTypeFunction(DataTypeFunction::Struct(
            self.0,
            StructDataTypeFunction::FieldNames,
        ))
    }
}
