use polars::error::PolarsResult;
use polars::prelude::*;
use polars_lazy::utils::expressions_to_schema;
use sqlparser::ast::{
    Expr as SqlExpr, Select, SelectItem, SetExpr, Statement, TableFactor, Value as SQLValue,
};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;

use crate::sql_expr::parse_sql_expr;

#[derive(Default, Clone)]
pub struct SQLContext {
    table_map: PlHashMap<String, LazyFrame>,
}

impl SQLContext {
    pub fn new() -> Self {
        Self {
            table_map: PlHashMap::new(),
        }
    }

    pub fn register(&mut self, name: &str, lf: LazyFrame) {
        self.table_map.insert(name.to_owned(), lf);
    }

    fn execute_select(&self, select_stmt: &Select) -> PolarsResult<LazyFrame> {
        // Determine involved dataframe
        // Implicit join require some more work in query parsers, Explicit join are preferred for now.
        let tbl = select_stmt
            .from
            .get(0)
            .ok_or_else(|| PolarsError::ComputeError("No table name provided in query".into()))?;
        let mut alias_map = PlHashMap::new();
        let tbl_name = match &tbl.relation {
            TableFactor::Table { name, alias, .. } => {
                let tbl_name = name.0.get(0).unwrap().value.as_str();
                if self.table_map.contains_key(tbl_name) {
                    if let Some(alias) = alias {
                        alias_map.insert(alias.name.value.clone(), tbl_name.to_owned());
                    };
                    tbl_name
                } else {
                    return Err(PolarsError::ComputeError(
                        format!("Table name {tbl_name} was not found").into(),
                    ));
                }
            }
            // Support bare table, optional with alias for now
            _ => return Err(PolarsError::ComputeError("Not implemented".into())),
        };
        let lf = self.table_map.get(tbl_name).ok_or_else(|| {
            PolarsError::ComputeError(
                format!("Table '{}' was not registered in the SQLContext", tbl_name).into(),
            )
        })?;

        let mut contains_wildcard = false;

        // Filter Expression
        let lf = match select_stmt.selection.as_ref() {
            Some(expr) => {
                let filter_expression = parse_sql_expr(expr)?;
                lf.clone().filter(filter_expression)
            }
            None => lf.clone(),
        };
        // Column Projections
        let projections: Vec<_> = select_stmt
            .projection
            .iter()
            .map(|select_item| {
                Ok(match select_item {
                    SelectItem::UnnamedExpr(expr) => parse_sql_expr(expr)?,
                    SelectItem::ExprWithAlias { expr, alias } => {
                        let expr = parse_sql_expr(expr)?;
                        expr.alias(&alias.value)
                    }
                    SelectItem::QualifiedWildcard(_) | SelectItem::Wildcard => {
                        contains_wildcard = true;
                        col("*")
                    }
                })
            })
            .collect::<PolarsResult<_>>()?;

        // Check for group by
        // After projection since there might be number.
        let groupby_keys: Vec<Expr> = select_stmt
            .group_by
            .iter()
            .map(|e| match e {
                  SqlExpr::Value(SQLValue::Number(idx, _)) => {
                    let idx = match idx.parse::<usize>() {
                        Ok(0)| Err(_) => Err(
                        PolarsError::ComputeError(
                            format!("Group By Error: Only positive number or expression are supported, got {idx}").into()
                        )),
                        Ok(idx) => Ok(idx)
                    }?;
                    Ok(projections[idx].clone())
                  }
                  SqlExpr::Value(_) => Err(
                      PolarsError::ComputeError("Group By Error: Only positive number or expression are supported".into())
                  ),
                  _ => parse_sql_expr(e)
                }
            )
            .collect::<PolarsResult<_>>()?;

        if groupby_keys.is_empty() {
            Ok(lf.select(projections))
        } else {
            self.process_groupby(lf, contains_wildcard, &groupby_keys, &projections)
        }
    }

    pub fn execute(&self, query: &str) -> PolarsResult<LazyFrame> {
        let ast = Parser::parse_sql(&GenericDialect::default(), query)
            .map_err(|e| PolarsError::ComputeError(format!("{:?}", e).into()))?;
        if ast.len() != 1 {
            Err(PolarsError::ComputeError(
                "One and only one statement at a time please".into(),
            ))
        } else {
            let ast = ast.get(0).unwrap();
            Ok(match ast {
                Statement::Query(query) => {
                    let rs = match &query.body.as_ref() {
                        SetExpr::Select(select_stmt) => self.execute_select(select_stmt)?,
                        _ => {
                            return Err(PolarsError::ComputeError(
                                "INSERT, UPDATE is not supported for polars".into(),
                            ))
                        }
                    };
                    match &query.limit {
                        Some(SqlExpr::Value(SQLValue::Number(nrow, _))) => {
                            let nrow = nrow.parse().map_err(|err| {
                                PolarsError::ComputeError(
                                    format!("Conversion Error: {:?}", err).into(),
                                )
                            })?;
                            rs.limit(nrow)
                        }
                        None => rs,
                        _ => {
                            return Err(PolarsError::ComputeError(
                                "Only support number argument to LIMIT clause".into(),
                            ))
                        }
                    }
                }
                _ => {
                    return Err(PolarsError::ComputeError(
                        format!("Statement type {:?} is not supported", ast).into(),
                    ))
                }
            })
        }
    }
    fn process_groupby(
        &self,
        lf: LazyFrame,
        contains_wildcard: bool,
        groupby_keys: &[Expr],
        projections: &[Expr],
    ) -> PolarsResult<LazyFrame> {
        // check groupby and projection due to difference between SQL and polars
        // Return error on wild card, shouldn't process this
        if contains_wildcard {
            return Err(PolarsError::ComputeError(
                "Group By Error: Can't processed wildcard in groupby".into(),
            ));
        }
        let schema_before = lf.schema().unwrap();

        let groupby_keys_schema =
            expressions_to_schema(groupby_keys, &schema_before, Context::Default).unwrap();

        // remove the groupby keys as polars adds those implicitly
        let mut aggregation_projection = Vec::with_capacity(projections.len());
        for e in projections {
            let field = e.to_field(&schema_before, Context::Default)?;
            if groupby_keys_schema.get(&field.name).is_none() {
                aggregation_projection.push(e.clone())
            }
        }

        let aggregated = lf.groupby(groupby_keys).agg(&aggregation_projection);

        let projection_schema =
            expressions_to_schema(projections, &schema_before, Context::Default).unwrap();

        // a final projection to get the proper order
        let final_projection = projection_schema
            .iter_names()
            .zip(projections)
            .map(|(name, projection_expr)| {
                if groupby_keys_schema.get(name).is_some() {
                    projection_expr.clone()
                } else {
                    col(name)
                }
            })
            .collect::<Vec<_>>();

        Ok(aggregated.select(final_projection))
    }
}
