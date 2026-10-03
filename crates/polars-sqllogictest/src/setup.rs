use polars_core::prelude::*;
use polars_lazy::prelude::*;
use polars_sql::SQLContext;
use sqlparser::ast::{
    BinaryOperator, ColumnDef, DataType as SqlDataType, Delete, Expr, FromTable, Insert,
    ObjectType, SetExpr, Statement, TableFactor, TableObject, UnaryOperator, Value,
};

use crate::engine::EngineError;

pub fn is_setup_statement(stmt: &Statement) -> bool {
    matches!(
        stmt,
        Statement::CreateTable(_)
            | Statement::Insert(_)
            | Statement::CreateIndex(_)
            | Statement::CreateView(_)
            | Statement::Drop {
                object_type: ObjectType::Table | ObjectType::Index | ObjectType::View,
                ..
            }
    )
}

pub fn run_setup_statement(
    tables: &mut PlHashMap<String, DataFrame>,
    views: &mut PlHashSet<String>,
    ctx: &mut SQLContext,
    stmt: &Statement,
) -> Result<u64, EngineError> {
    match stmt {
        Statement::CreateTable(create) => {
            let name = object_name_to_string(&create.name);
            let df = empty_frame(&create.columns)?;
            ctx.register(&name, df.clone().lazy());
            tables.insert(name, df);
            Ok(0)
        },
        Statement::Insert(insert) => run_insert(tables, ctx, insert),
        Statement::CreateIndex(_) => Ok(0),
        Statement::CreateView(create_view) => {
            let name = object_name_to_string(&create_view.name);
            if !create_view.or_replace && (tables.contains_key(&name) || views.contains(&name)) {
                return Err(EngineError::new(format!("view already exists: {name}")));
            }
            let lf = ctx
                .execute(&create_view.query.to_string())
                .map_err(EngineError::from)?;
            ctx.register(&name, lf);
            views.insert(name);
            Ok(0)
        },
        Statement::Drop {
            object_type: ObjectType::Table,
            names,
            ..
        } => {
            for name in names {
                let name = object_name_to_string(name);
                ctx.unregister(&name);
                tables.remove(&name);
            }
            Ok(0)
        },
        Statement::Drop {
            object_type: ObjectType::Index,
            ..
        } => Ok(0),
        Statement::Drop {
            object_type: ObjectType::View,
            names,
            ..
        } => {
            for name in names {
                let name = object_name_to_string(name);
                if !views.remove(&name) {
                    return Err(EngineError::new(format!("no such view: {name}")));
                }
                ctx.unregister(&name);
            }
            Ok(0)
        },
        _ => Err(EngineError::new(format!(
            "unsupported setup statement: {stmt}"
        ))),
    }
}

pub fn delete_target_table(stmt: &Statement) -> Option<String> {
    let Statement::Delete(Delete { from, .. }) = stmt else {
        return None;
    };
    let from_tables = match from {
        FromTable::WithFromKeyword(f) | FromTable::WithoutKeyword(f) => f,
    };
    match &from_tables.first()?.relation {
        TableFactor::Table { name, .. } => Some(object_name_to_string(name)),
        _ => None,
    }
}

fn object_name_to_string(name: &sqlparser::ast::ObjectName) -> String {
    name.0
        .last()
        .map(|part| part.to_string())
        .unwrap_or_default()
}

fn map_dtype(dtype: &SqlDataType) -> Result<DataType, EngineError> {
    let dt = match dtype {
        SqlDataType::TinyInt(_)
        | SqlDataType::SmallInt(_)
        | SqlDataType::Int(_)
        | SqlDataType::Integer(_)
        | SqlDataType::BigInt(_) => DataType::Int64,
        SqlDataType::Real
        | SqlDataType::Float(_)
        | SqlDataType::Double(_)
        | SqlDataType::DoublePrecision => DataType::Float64,
        SqlDataType::Char(_)
        | SqlDataType::Varchar(_)
        | SqlDataType::Text
        | SqlDataType::String(_) => DataType::String,
        SqlDataType::Bool | SqlDataType::Boolean => DataType::Boolean,
        SqlDataType::Unspecified => DataType::Int64,
        other => {
            return Err(EngineError::new(format!(
                "unsupported column type in shim: {other}"
            )));
        },
    };
    Ok(dt)
}

fn empty_frame(columns: &[ColumnDef]) -> Result<DataFrame, EngineError> {
    let mut series = Vec::with_capacity(columns.len());
    for col in columns {
        let dtype = map_dtype(&col.data_type)?;
        series.push(Column::new_empty(col.name.value.as_str().into(), &dtype));
    }
    DataFrame::new_infer_height(series).map_err(EngineError::from)
}

fn run_insert(
    tables: &mut PlHashMap<String, DataFrame>,
    ctx: &mut SQLContext,
    insert: &Insert,
) -> Result<u64, EngineError> {
    let name = match &insert.table {
        TableObject::TableName(name) => object_name_to_string(name),
        other => {
            return Err(EngineError::new(format!(
                "unsupported INSERT target: {other}"
            )));
        },
    };

    let base = tables
        .get(&name)
        .ok_or_else(|| EngineError::new(format!("INSERT into unknown table: {name}")))?
        .clone();

    let schema = base.schema();
    let all_names: Vec<String> = schema.iter_names().map(|n| n.to_string()).collect();

    let target_names: Vec<String> = if insert.columns.is_empty() {
        all_names.clone()
    } else {
        insert.columns.iter().map(object_name_to_string).collect()
    };

    let query = insert
        .source
        .as_deref()
        .ok_or_else(|| EngineError::new("INSERT requires a source"))?;

    let new_rows = match query.body.as_ref() {
        SetExpr::Values(values) => {
            let rows = &values.rows;
            let mut columns: PlHashMap<String, Vec<AnyValue<'static>>> = PlHashMap::new();
            for target in &target_names {
                columns.insert(target.clone(), Vec::with_capacity(rows.len()));
            }

            for row in rows {
                let exprs = &row.content;
                if exprs.len() != target_names.len() {
                    return Err(EngineError::new(format!(
                        "INSERT row has {} values but {} columns were targeted",
                        exprs.len(),
                        target_names.len()
                    )));
                }
                for (target, expr) in target_names.iter().zip(exprs) {
                    let value = expr_to_any_value(expr)?;
                    columns.get_mut(target).unwrap().push(value);
                }
            }

            let n_rows = rows.len();
            let mut series = Vec::with_capacity(all_names.len());
            for (field_name, dtype) in schema.iter() {
                let name_str = field_name.to_string();
                let values = match columns.get(&name_str) {
                    Some(values) => values.clone(),
                    None => vec![AnyValue::Null; n_rows],
                };
                let s =
                    Series::from_any_values_and_dtype(field_name.clone(), &values, dtype, false)
                        .map_err(EngineError::from)?;
                series.push(s.into_column());
            }
            DataFrame::new_infer_height(series).map_err(EngineError::from)?
        },
        _ => {
            let source = ctx
                .execute(&query.to_string())
                .and_then(|lf| lf.collect())
                .map_err(EngineError::from)?;
            if source.width() != target_names.len() {
                return Err(EngineError::new(format!(
                    "INSERT source has {} columns but {} columns were targeted",
                    source.width(),
                    target_names.len()
                )));
            }

            let height = source.height();
            let mut series = Vec::with_capacity(all_names.len());
            for (field_name, dtype) in schema.iter() {
                let name_str = field_name.to_string();
                let pos = target_names.iter().position(|t| t == &name_str);
                let s = match pos {
                    Some(i) => source
                        .select_at_idx(i)
                        .unwrap()
                        .cast(dtype)
                        .map_err(EngineError::from)?
                        .with_name(field_name.clone()),
                    None => Column::full_null(field_name.clone(), height, dtype),
                };
                series.push(s);
            }
            DataFrame::new_infer_height(series).map_err(EngineError::from)?
        },
    };

    let n_rows = new_rows.height();
    let stacked = base.vstack(&new_rows).map_err(EngineError::from)?;
    ctx.register(&name, stacked.clone().lazy());
    tables.insert(name, stacked);
    Ok(n_rows as u64)
}

fn expr_to_any_value(expr: &Expr) -> Result<AnyValue<'static>, EngineError> {
    match expr {
        Expr::Value(value) => value_to_any_value(&value.value, false),
        Expr::UnaryOp {
            op: UnaryOperator::Minus,
            expr,
        } => match expr.as_ref() {
            Expr::Value(value) => value_to_any_value(&value.value, true),
            other => Err(EngineError::new(format!(
                "unsupported INSERT expression: {other}"
            ))),
        },
        Expr::BinaryOp {
            left,
            op: BinaryOperator::PGBitwiseShiftLeft,
            right,
        } => match (expr_to_any_value(left)?, expr_to_any_value(right)?) {
            (AnyValue::Int64(l), AnyValue::Int64(r)) if (0..64).contains(&r) => {
                Ok(AnyValue::Int64(l.wrapping_shl(r as u32)))
            },
            _ => Err(EngineError::new(format!(
                "unsupported INSERT expression: {expr}"
            ))),
        },
        other => Err(EngineError::new(format!(
            "unsupported INSERT expression: {other}"
        ))),
    }
}

fn value_to_any_value(value: &Value, negate: bool) -> Result<AnyValue<'static>, EngineError> {
    let av = match value {
        Value::Number(repr, _) => {
            let text = if negate {
                format!("-{repr}")
            } else {
                repr.clone()
            };
            if let Ok(i) = text.parse::<i64>() {
                AnyValue::Int64(i)
            } else {
                let f = text
                    .parse::<f64>()
                    .map_err(|_| EngineError::new(format!("invalid numeric literal: {text}")))?;
                AnyValue::Float64(f)
            }
        },
        Value::SingleQuotedString(s) | Value::DoubleQuotedString(s) => {
            AnyValue::StringOwned(s.as_str().into())
        },
        Value::Boolean(b) => AnyValue::Boolean(*b),
        Value::Null => AnyValue::Null,
        other => {
            return Err(EngineError::new(format!(
                "unsupported literal in shim: {other}"
            )));
        },
    };
    Ok(av)
}
