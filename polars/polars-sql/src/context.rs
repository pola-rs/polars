use std::cell::RefCell;

use polars_core::prelude::*;
use polars_lazy::prelude::*;
use polars_plan::prelude::*;
use polars_plan::utils::expressions_to_schema;
use sqlparser::ast::{
    Expr as SqlExpr, JoinOperator, OrderByExpr, Select, SelectItem, SetExpr, Statement,
    TableFactor, TableWithJoins, Value as SQLValue,
};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;

use crate::sql_expr::{parse_sql_expr, process_join_constraint};

thread_local! {pub(crate) static TABLES: RefCell<Vec<String>> = RefCell::new(vec![])}

#[derive(Default, Clone)]
pub struct SQLContext {
    pub table_map: PlHashMap<String, LazyFrame>,
}

impl SQLContext {
    pub fn try_new() -> PolarsResult<Self> {
        TABLES.with(|cell| {
            if !cell.borrow().is_empty() {
                Err(PolarsError::ComputeError(
                    "only one sql-context per thread allowed".into(),
                ))
            } else {
                Ok(())
            }
        })?;

        Ok(Self {
            table_map: PlHashMap::new(),
        })
    }

    pub fn register(&mut self, name: &str, lf: LazyFrame) {
        TABLES.with(|cell| cell.borrow_mut().push(name.to_owned()));
        self.table_map.insert(name.to_owned(), lf);
    }

    fn get_relation_name<'a>(&self, relation: &'a TableFactor) -> PolarsResult<&'a str> {
        let tbl_name = match relation {
            TableFactor::Table { name, alias, .. } => {
                let tbl_name = name.0.get(0).unwrap().value.as_str();

                if self.table_map.contains_key(tbl_name) {
                    if let Some(_) = alias {
                        return Err(PolarsError::ComputeError(
                            format!("Table aliases are not supported.").into(),
                        ));
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
        Ok(tbl_name)
    }

    fn get_table(&self, name: &str) -> PolarsResult<LazyFrame> {
        self.table_map.get(name).cloned().ok_or_else(|| {
            PolarsError::ComputeError(
                format!("Table '{}' was not registered in the SQLContext", name).into(),
            )
        })
    }

    fn execute_select(&self, select_stmt: &Select) -> PolarsResult<LazyFrame> {
        // Determine involved dataframe
        // Implicit join require some more work in query parsers, Explicit join are preferred for now.
        let sql_tbl: &TableWithJoins = select_stmt
            .from
            .get(0)
            .ok_or_else(|| PolarsError::ComputeError("No table name provided in query".into()))?;

        let tbl_name = self.get_relation_name(&sql_tbl.relation)?;
        let mut lf = self.get_table(tbl_name)?;

        if !sql_tbl.joins.is_empty() {
            for tbl in &sql_tbl.joins {
                let join_tbl_name = self.get_relation_name(&tbl.relation)?;
                let join_tbl = self.get_table(join_tbl_name)?;
                match &tbl.join_operator {
                    JoinOperator::Inner(constraint) => {
                        let (left_on, right_on) =
                            process_join_constraint(&constraint, tbl_name, join_tbl_name)?;
                        lf = lf.inner_join(join_tbl, left_on, right_on)
                    }
                    JoinOperator::LeftOuter(constraint) => {
                        let (left_on, right_on) =
                            process_join_constraint(&constraint, tbl_name, join_tbl_name)?;
                        lf = lf.left_join(join_tbl, left_on, right_on)
                    }
                    JoinOperator::FullOuter(constraint) => {
                        let (left_on, right_on) =
                            process_join_constraint(&constraint, tbl_name, join_tbl_name)?;
                        lf = lf.outer_join(join_tbl, left_on, right_on)
                    }
                    JoinOperator::CrossJoin => lf = lf.cross_join(join_tbl),
                    join_type => {
                        return Err(PolarsError::ComputeError(
                            format!(
                                "Join type: '{:?}' not yet supported by polars-sql",
                                join_type
                            )
                            .into(),
                        ))
                    }
                }
            }
        }

        let mut contains_wildcard = false;

        // Filter Expression
        let lf = match select_stmt.selection.as_ref() {
            Some(expr) => {
                let filter_expression = parse_sql_expr(expr)?;
                lf.filter(filter_expression)
            }
            None => lf,
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

    // Executes the given statement against the SQLContext
    pub fn execute_statement(&self, stmt: &Statement) -> PolarsResult<LazyFrame> {
        let ast = stmt;
        Ok(match ast {
            Statement::Query(query) => {
                let mut lf = match &query.body.as_ref() {
                    SetExpr::Select(select_stmt) => self.execute_select(select_stmt)?,
                    _ => {
                        return Err(PolarsError::ComputeError(
                            "INSERT, UPDATE is not supported for polars".into(),
                        ))
                    }
                };
                if !query.order_by.is_empty() {
                    lf = self.process_order_by(lf, &query.order_by)?;
                }
                match &query.limit {
                    Some(SqlExpr::Value(SQLValue::Number(nrow, _))) => {
                        let nrow = nrow.parse().map_err(|err| {
                            PolarsError::ComputeError(format!("Conversion Error: {:?}", err).into())
                        })?;
                        lf.limit(nrow)
                    }
                    None => lf,
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

    // Executes the given SQL query against the SQLContext.
    pub fn execute(&self, query: &str) -> PolarsResult<LazyFrame> {
        let ast = Parser::parse_sql(&GenericDialect::default(), query)
            .map_err(|e| PolarsError::ComputeError(format!("{:?}", e).into()))?;
        if ast.len() != 1 {
            return Err(PolarsError::ComputeError(
                "One and only one statement at a time please".into(),
            ));
        }

        let ast = ast.get(0).unwrap();
        return self.execute_statement(ast);
    }

    fn process_order_by(&self, lf: LazyFrame, ob: &[OrderByExpr]) -> PolarsResult<LazyFrame> {
        let mut by = Vec::with_capacity(ob.len());
        let mut reverse = Vec::with_capacity(ob.len());

        for ob in ob {
            by.push(parse_sql_expr(&ob.expr)?);
            if let Some(false) = ob.asc {
                reverse.push(true)
            } else {
                reverse.push(false)
            }

            if ob.nulls_first.is_some() {
                return Err(PolarsError::ComputeError(
                    "nulls first/last is not yet supported".into(),
                ));
            }
        }

        Ok(lf.sort_by_exprs(&by, reverse, false))
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
        let schema_before = lf.schema()?;

        let groupby_keys_schema =
            expressions_to_schema(groupby_keys, &schema_before, Context::Default)?;

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
            expressions_to_schema(projections, &schema_before, Context::Default)?;

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

impl Drop for SQLContext {
    fn drop(&mut self) {
        // drop old tables
        TABLES.with(|cell| {
            cell.borrow_mut().clear();
        });
    }
}
