use polars_core::export::arrow::io::csv;
use polars_lazy::prelude::LazyCsvReader;
use sqlparser::ast;
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;

use polars_lazy::dsl;
use polars_core::prelude::*;


fn main(){
    dbg!("Hello World!");
}

fn extract_projections(items: &[ast::SelectItem]) -> Vec<dsl::Expr> {
    use ast::SelectItem::*;
    items.iter().map(|si| {
        match si {
            Wildcard => {
                dsl::col("*")
            }
            _ => {
                todo!()
            }
        }
    }) .collect()
}


fn extract_files(relation: &ast::TableFactor) -> Vec<&str> {
    use ast::TableFactor::*;
    match relation {
        Table { name, ..} => {
            let idents = &name.0;
            assert_eq!(idents.len(), 1, "only 1 table allowed");
            idents.iter().map(|ident| &*ident.value).collect()
        },
        _ => todo!()
    }
}


fn run_expressions(projections: &[dsl::Expr], table: &str) -> Result<()> {
    
    let splitted = table.split(".").collect::<Vec<_>>();
    if splitted.len() == 1 {
        Err(PolarsError::ComputeError("expected a file with an extension".into()))
    } else {
        match splitted[splitted.len() - 1] {
            "csv" => {
                let df = LazyCsvReader::new(table.into()).finish()?.select(projections).collect()?;
                println!("ran query in {} ms\n {:?}", 1, df);
            }
            "parquet" => {
                todo!()
            }
            "ipc" | "arrow" => {
                todo!()
            }
            ext => panic!("extension {} not yet supported", ext)
        }

        Ok(())
    }
}


fn parse_query(query: &str) {

    let dialect = GenericDialect {}; // or AnsiDialect
    let mut parsed = Parser::parse_sql(&dialect, query).unwrap();
    // still have to decide what to do with with multiple statements.
    assert_eq!(parsed.len(), 1);

    let stmt = parsed.pop().unwrap();

    match stmt {
        ast::Statement::Query(query) => {
            match query.body {
                ast::SetExpr::Select(select) => {
                    let projections = extract_projections(&select.projection);
                
                    let from_tables = select.from;
                    assert_eq!(from_tables.len(), 1, "multiple sources not yet supported");

                    let relation = &from_tables[0].relation;
                    let tables = extract_files(relation);
                    assert_eq!(tables.len(), 1, "multiple sources not yet supported");

                    run_expressions(&projections, &tables[0]).unwrap();

                }
                _ => {
                    panic!("not yet supported")
                }
            }
        }
        _ => {
            panic!("not yet supported")
        }
    }

}


#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_first_query() {
        let query = "select * from \"../../examples/aggregate_multiple_files_in_chunks/datasets/foods1.csv\"";
        parse_query(query);
    }

}