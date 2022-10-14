use std::ffi::OsStr;
use std::io::{self, Read, BufRead, Write};
use std::path::Path;
use std::str;

use polars_core::prelude::*;

#[cfg(feature = "csv")]
use polars_lazy::frame::LazyCsvReader;
use polars_lazy::frame::LazyFrame;
#[cfg(feature = "parquet")]
use polars_lazy::frame::ScanArgsParquet;
use polars_sql::SQLContext;
use sqlparser::ast::{
    Expr as SqlExpr, JoinOperator, OrderByExpr, Select, SelectItem, SetExpr, Statement, Query,
    TableFactor, TableWithJoins, Value as SQLValue,
};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;

// Command: /dd or dataframes
fn print_dataframes(dataframes: &Vec<(String, String)>) {
    println!(
        "{} dataframes registered{}",
        dataframes.len(),
        if dataframes.is_empty() { "." } else { ":" }
    );
    for (name, file) in dataframes.iter() {
        println!("{}:\t {}", name, file);
    }
}

// Command: /? or help
fn print_help() {
    for (name, short, desc) in vec![
        ("dataframes", "dd", "Show registered frames."),
        ("help", "?", "Display this help."),
        (
            "register",
            "rd",
            "Register new dataframe: \\rd <name> <source>",
        ),
        ("quit", "q", "Exit"),
    ]
    .iter()
    {
        println!("{:20}\\{:10} {}", name, short, desc);
    }
}

fn setup_dataframe_from_table(context: &mut SQLContext, relation: &TableFactor) -> PolarsResult<()> {
    let tbl_name = match relation {
        TableFactor::Table { name, alias, .. } => {
            let source = name.0.get(0).unwrap().value.as_str();
            let name = match alias {
                Some(alias) => alias.name.value.to_string(),
                None => source.to_string(),
            };
            let df = match get_extension_from_filename(&source) {
                #[cfg(feature = "csv")]
                Some("csv") => LazyCsvReader::new(source).finish(),
                #[cfg(feature = "parquet")]
                Some("parquet") => LazyFrame::scan_parquet(source, ScanArgsParquet::default()),
                None | Some(_) => {
                    return Err(PolarsError::ComputeError("Unsupported file extension".into()))
                }
            };
        
            match df {
                Ok(frame) => {
                    context.register(&name, frame);
                    // dataframes.push((name.to_owned(), source.to_owned()));
                    // println!("Added dataframe \"{}\" from file {}", name, source)
                }
                Err(e) => println!("{}", e),
            }
        },
        _ => return Err(PolarsError::ComputeError("Unsupported.".into())),
    };

    Ok(())
}

fn setup_dataframes(context: &mut SQLContext, stmt: &Select) -> PolarsResult<()> {
    let sql_tbl: &TableWithJoins = stmt
            .from
            .get(0)
            .ok_or_else(|| PolarsError::ComputeError("No table name provided in query".into()))?;
    
    setup_dataframe_from_table(context, &sql_tbl.relation)?;

    if !sql_tbl.joins.is_empty() {
        for tbl in &sql_tbl.joins {
            setup_dataframe_from_table(context, &tbl.relation)?;
        }
    }

    Ok(())
}

fn execute_query(context: &mut SQLContext, query: &str) -> PolarsResult<DataFrame> {
    let ast = match Parser::parse_sql(&GenericDialect::default(), query) {
        Ok(ast) => ast,
        Err(e) => return Err(PolarsError::ComputeError(format!("Error parsing SQL: {:?}", e).into()))
    };
    if ast.len() != 1 {
        return Err(PolarsError::ComputeError("One and only one statement at a time please".into()));
    }

    let ast = ast.get(0).unwrap();
    match ast {
        Statement::Query(query) => {
            match &query.body.as_ref() {
                SetExpr::Select(select_stmt) => {
                    setup_dataframes(context, &select_stmt);
                },
                // Statement is validated in context::execute_statement
                // so we leave it to them to return an error type for unsupported expressions
                _ => (),
            }
        },
        // Statement is validated in context::execute_statement
        // so we leave it to them to return an error type for unsupported statements
        _ => (),
    }

    // Execute SQL command
    return context.execute_statement(&ast)?.limit(100).collect();
}

fn register_dataframe(
    context: &mut SQLContext,
    dataframes: &mut Vec<(String, String)>,
    command: Vec<&str>,
) {
    if command.len() < 3 {
        println!("Usage: \\rd <name> <file>");
        return;
    }
    let name = command[1];
    let source = command[2];

    let df = match get_extension_from_filename(source) {
        #[cfg(feature = "csv")]
        Some("csv") => LazyCsvReader::new(source).finish(),
        #[cfg(feature = "parquet")]
        Some("parquet") => LazyFrame::scan_parquet(source, ScanArgsParquet::default()),
        None | Some(_) => {
            println!("Unknown extension.");
            return;
        }
    };

    match df {
        Ok(frame) => {
            context.register(name, frame);
            dataframes.push((name.to_owned(), source.to_owned()));
            println!("Added dataframe \"{}\" from file {}", name, source)
        }
        Err(e) => println!("{}", e),
    }
}

pub fn run_tty() -> std::io::Result<()> {
    let mut stdout = io::stdout();
    let mut context = SQLContext::try_new().unwrap();
    let mut dataframes = Vec::new();
    let mut input = String::new();

    println!("Welcome to Polars CLI. Commands end with ; or \\n");
    println!("Type help or \\? for help.");

    loop {
        print!("=> ");
        stdout.flush().unwrap();
        input.clear();

        if let Err(e) = io::stdin().lock().read_line(&mut input) {
            println!("Error reading from stdin: {}", e);
            continue;
        }

        println!("Input: {}", input);
        let command: Vec<&str> = input.trim().split(" ").collect();
        if command[0].is_empty() {
            continue;
        }

        match command[0] {
            "\\dd" | "dataframes" => print_dataframes(&dataframes),
            "\\rd" | "register" => register_dataframe(&mut context, &mut dataframes, command),
            "\\?" | "help" | "?" | "\\h" => print_help(),
            "\\q" | "quit" | "exit" => {
                println!("Bye");
                return Ok(());
            }
            _ => {
                match execute_query(&mut context, input.trim()) {
                    Ok(lf) => println!("{}", lf),
                    Err(e) => println!("{}", e),
                }
            }
        }

        println!();
    }
}

pub fn run() -> io::Result<()> {
    if atty::is(atty::Stream::Stdin) {
        return run_tty();
    } 

    let mut context = SQLContext::try_new().unwrap();
    for line in std::io::stdin().lines() {
        match execute_query(&mut context, line.unwrap().trim()) {
            Ok(lf) => println!("{}", lf),
            Err(e) => println!("{}", e),
        }
    }
    
    Ok(())
}

fn get_extension_from_filename(filename: &str) -> Option<&str> {
    Path::new(filename).extension().and_then(OsStr::to_str)
}
