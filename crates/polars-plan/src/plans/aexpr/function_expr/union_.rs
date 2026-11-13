use super::*;

#[cfg_attr(feature = "ir_serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Eq, PartialEq, Debug, Hash)]
pub enum IRUnionFunction {
    Tag,
    Field(PlSmallStr),
    /// Every row holds the input as the variant `variant` of the union `dtype`.
    Value {
        dtype: DataType,
        variant: PlSmallStr,
    },
    /// The inputs are the union, an arm per name in `names`, and the default if `has_default`;
    /// the output has dtype `dtype`.
    ///
    /// An arm reads its variant's value as `input.union.field(name)`, so it is valid in the outer
    /// frame. The kernel evaluates each arm only on the rows of its variant.
    Match {
        names: Vec<PlSmallStr>,
        has_default: bool,
        dtype: DataType,
    },
}

impl IRUnionFunction {
    pub(super) fn get_field(&self, mapper: FieldsMapper) -> PolarsResult<Field> {
        if let Self::Match { dtype, .. } = self {
            return Ok(Field::new(mapper.args()[0].name().clone(), dtype.clone()));
        }
        if let Self::Value { dtype, variant } = self {
            return mapper.try_map_dtype(|input| {
                let DataType::Union(fields) = dtype else {
                    unreachable!()
                };
                let expected = fields[fields.try_index_of(variant)?].dtype();
                // Type coercion casts these to the variant's dtype.
                polars_ensure!(
                    input == expected || input.is_null() || literal_fits(input, expected),
                    SchemaMismatch: "variant `{variant}` of `{dtype}` has dtype `{expected}`, \
                    got a value of dtype `{input}`; cast the value first"
                );
                Ok(dtype.clone())
            });
        }
        mapper.try_map_dtype(|dtype| {
            let DataType::Union(fields) = dtype else {
                polars_bail!(InvalidOperation: "`{self}` requires a Union dtype, got `{dtype}`");
            };
            Ok(match self {
                Self::Tag => fields.tag_dtype(),
                Self::Field(name) => fields[fields.try_index_of(name)?].dtype().clone(),
                Self::Value { .. } | Self::Match { .. } => unreachable!(),
            })
        })
    }

    pub fn function_options(&self) -> FunctionOptions {
        FunctionOptions::elementwise()
    }
}

impl Display for IRUnionFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Tag => write!(f, "union.tag"),
            Self::Field(name) => write!(f, "union.field({name})"),
            Self::Value { variant, .. } => write!(f, "union.value({variant})"),
            Self::Match {
                names, has_default, ..
            } => {
                write!(f, "union.match(")?;
                for (i, name) in names.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{name}")?;
                }
                if *has_default {
                    write!(f, "{}default", if names.is_empty() { "" } else { ", " })?;
                }
                write!(f, ")")
            },
        }
    }
}

impl From<IRUnionFunction> for IRFunctionExpr {
    fn from(value: IRUnionFunction) -> Self {
        Self::UnionExpr(value)
    }
}

/// Whether a dynamic literal of dtype `input` can take the dtype `to` without changing kind.
pub(crate) fn literal_fits(input: &DataType, to: &DataType) -> bool {
    match input {
        DataType::Unknown(UnknownKind::Int(_)) => to.is_primitive_numeric() || to.is_decimal(),
        DataType::Unknown(UnknownKind::Float) => to.is_float() || to.is_decimal(),
        DataType::Unknown(UnknownKind::Str) => {
            to.is_string() || to.is_categorical() || to.is_enum()
        },
        _ => false,
    }
}
