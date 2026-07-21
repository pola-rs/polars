use polars_core::prelude::*;
use polars_lazy::prelude::*;
use polars_sql::SQLContext;
use sqlparser::ast::{
    ColumnDef, DataType as SqlDataType, Expr, Insert, ObjectType, SetExpr, Statement, TableObject,
    UnaryOperator, Value,
};

use crate::engine::EngineError;

pub fn is_setup_statement(stmt: &Statement) -> bool {
    matches!(
        stmt,
        Statement::CreateTable(_)
            | Statement::Insert(_)
            | Statement::Drop {
                object_type: ObjectType::Table,
                ..
            }
    )
}

pub fn run_setup_statement(
    tables: &mut PlHashMap<String, DataFrame>,
    ctx: &SQLContext,
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
        _ => Err(EngineError::new(format!(
            "unsupported setup statement: {stmt}"
        ))),
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
    ctx: &SQLContext,
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

    let rows = match insert.source.as_deref().map(|q| q.body.as_ref()) {
        Some(SetExpr::Values(values)) => &values.rows,
        _ => {
            return Err(EngineError::new("INSERT source must be a VALUES clause"));
        },
    };

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
        let s = Series::from_any_values_and_dtype(field_name.clone(), &values, dtype, false)
            .map_err(EngineError::from)?;
        series.push(s.into_column());
    }

    let new_rows = DataFrame::new_infer_height(series).map_err(EngineError::from)?;
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
