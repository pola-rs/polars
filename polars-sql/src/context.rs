use std::collections::HashMap;

use crate::sql_expr::parse_sql_expr;
use polars::error::PolarsError;
use polars::prelude::{col, DataFrame, IntoLazy, LazyFrame};
use sqlparser::ast::{
    Expr as SqlExpr, Select, SelectItem, SetExpr, Statement, TableFactor, Value as SQLValue,
};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;

#[derive(Default)]
pub struct SQLContext {
    table_map: HashMap<String, LazyFrame>,
    dialect: GenericDialect,
}

impl SQLContext {
    pub fn new() -> Self {
        Self {
            table_map: HashMap::new(),
            dialect: GenericDialect::default(),
        }
    }

    pub fn register(&mut self, name: &str, df: &DataFrame) {
        self.table_map.insert(name.to_owned(), df.clone().lazy());
    }

    fn execute_select(&self, select_stmt: &Select) -> Result<LazyFrame, PolarsError> {
        // Determine involved dataframe
        // Implicit join require some more work in query parsers, Explicit join are preferred for now.
        let tbl = select_stmt.from.get(0).unwrap();
        let mut alias_map = HashMap::new();
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
            // sqlparser::ast::TableFactor::Derived { lateral, subquery, alias } => todo!(),
            // sqlparser::ast::TableFactor::TableFunction { expr, alias } => todo!(),
            // sqlparser::ast::TableFactor::NestedJoin(_) => todo!(),
            // Support bare table, optional with alias for now
            _ => return Err(PolarsError::ComputeError("Not implemented".into())),
        };
        let df = &self.table_map[tbl_name];
        let mut raw_projection_before_alias: HashMap<String, usize> = HashMap::new();
        let mut contain_wildcard = false;
        // Column Projections
        let projection = select_stmt
            .projection
            .iter()
            .enumerate()
            .map(|(i, select_item)| {
                Ok(match select_item {
                    SelectItem::UnnamedExpr(expr) => {
                        let expr = parse_sql_expr(expr)?;
                        raw_projection_before_alias.insert(format!("{:?}", expr), i);
                        expr
                    }
                    SelectItem::ExprWithAlias { expr, alias } => {
                        let expr = parse_sql_expr(expr)?;
                        raw_projection_before_alias.insert(format!("{:?}", expr), i);
                        expr.alias(&alias.value)
                    }
                    SelectItem::QualifiedWildcard(_) | SelectItem::Wildcard => {
                        contain_wildcard = true;
                        col("*")
                    }
                })
            })
            .collect::<Result<Vec<_>, PolarsError>>()?;
        // Check for group by
        // After projection since there might be number.
        let group_by = select_stmt
            .group_by
            .iter()
            .map(
                |e|match e {
                  SqlExpr::Value(SQLValue::Number(idx, _)) => {
                    let idx = match idx.parse::<usize>() {
                        Ok(0)| Err(_) => Err(
                        PolarsError::ComputeError(
                            format!("Group By Error: Only positive number or expression are supported, got {idx}").into()
                        )),
                        Ok(idx) => Ok(idx)
                    }?;
                    Ok(projection[idx].clone())
                  }
                  SqlExpr::Value(_) => Err(
                      PolarsError::ComputeError("Group By Error: Only positive number or expression are supported".into())
                  ),
                  _ => parse_sql_expr(e)
                }
            )
            .collect::<Result<Vec<_>, PolarsError>>()?;

        // println!("before plan: {:?}", df.schema());
        let df = if group_by.is_empty() {
            df.clone().select(projection)
        } else {
            // check groupby and projection due to difference between SQL and polars
            // Return error on wild card, shouldn't process this
            if contain_wildcard {
                return Err(PolarsError::ComputeError(
                    "Group By Error: Can't processed wildcard in groupby".into(),
                ));
            }
            // Default polars group by will have group by columns at the front
            // need some container to contain position of group by columns and its position
            // at the final agg projection, check the schema for the existant of group by column
            // and its projections columns, keeping the original index
            let (exclude_expr, groupby_pos): (Vec<_>, Vec<_>) = group_by
                .iter()
                .map(|expr| raw_projection_before_alias.get(&format!("{:?}", expr)))
                .enumerate()
                .filter(|(_, proj_p)| proj_p.is_some())
                .map(|(gb_p, proj_p)| (*proj_p.unwrap(), (*proj_p.unwrap(), gb_p)))
                .unzip();
            let (agg_projection, agg_proj_pos): (Vec<_>, Vec<_>) = projection
                .iter()
                .enumerate()
                .filter(|(i, _)| !exclude_expr.contains(i))
                .enumerate()
                .map(|(agg_pj, (proj_p, expr))| (expr.clone(), (proj_p, agg_pj + group_by.len())))
                .unzip();
            // println!("Group By: {:?}, {:?}, {:?}", projection, group_by, agg_projection);
            let agg_df = df.clone().groupby(group_by).agg(agg_projection);
            let mut final_proj_pos = groupby_pos.into_iter()
                .chain(agg_proj_pos.into_iter())
                .collect::<Vec<_>>()
                // .map(|(proj_p, schema_p)|)
            ;

            final_proj_pos.sort_by(|(proj_pa, _), (proj_pb, _)| proj_pa.cmp(proj_pb));
            // println!("after plan: {:?}", agg_df.schema(), );
            let final_proj = final_proj_pos
                .into_iter()
                .map(|(_, shm_p)| col(agg_df.schema().get_index(shm_p).unwrap().0))
                .collect::<Vec<_>>();
            agg_df.select(final_proj)
        };
        // println!("after plan: {:?}", df.schema());
        Ok(df)
    }

    pub fn execute(&self, query: &str) -> Result<LazyFrame, PolarsError> {
        let ast = Parser::parse_sql(&self.dialect, query)
            .map_err(|e| PolarsError::ComputeError(format!("{:?}", e).into()))?;
        if ast.len() != 1 {
            Err(PolarsError::ComputeError(
                "One and only one statement at a time please".into(),
            ))
        } else {
            let ast = ast.get(0).unwrap();
            Ok(match ast {
                Statement::Query(query) => {
                    let rs = match &query.body {
                        SetExpr::Select(select_stmt) => self.execute_select(&*select_stmt)?,
                        // SetExpr::Query(_) => todo!(), // Subqueries
                        // SetExpr::SetOperation {
                        //     op,
                        //     all,
                        //     left,
                        //     right,
                        // } => todo!(), // Union, Except, Intersect
                        // SetExpr::Values(_) => todo!(), // Should not be implemented, return an errors here
                        // SetExpr::Insert(_) => todo!(),
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
}
