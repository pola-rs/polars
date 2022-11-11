use std::ffi::OsStr;
use std::io::{self, BufRead, Read, Write};
use std::path::Path;
use std::time::{Duration, Instant};

use polars_core::prelude::*;
#[cfg(feature = "csv")]
use polars_lazy::frame::LazyCsvReader;
use polars_lazy::frame::LazyFrame;
#[cfg(feature = "ipc")]
use polars_lazy::frame::ScanArgsIpc;
#[cfg(feature = "parquet")]
use polars_lazy::frame::ScanArgsParquet;
use polars_sql::SQLContext;
use rustyline::completion::FilenameCompleter;
use rustyline::error::ReadlineError;
use rustyline::{Editor, Result};
use sqlparser::ast::{Select, SetExpr, Statement, TableFactor, TableWithJoins};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;

const SUPPORTED_FILE_EXTENSIONS: &[&str] = &[
    #[cfg(feature = "csv")]
    "csv",
    #[cfg(feature = "parquet")]
    "parquet",
    #[cfg(feature = "ipc")]
    "ipc",
];

// Command: /dd | dataframes
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
// Command: /rd | register
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
    let df = create_dataframe_from_filename(source);

    match df {
        Ok(frame) => {
            context.register(name, frame);
            dataframes.push((name.to_owned(), source.to_owned()));
            println!("Added dataframe \"{}\" from file {}", name, source)
        }
        Err(e) => eprintln!("{}", e),
    }
}

// Command: /? | help
fn print_help() {
    println!("List of all client commands:");
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

fn create_dataframe_from_filename(filename: &str) -> PolarsResult<LazyFrame> {
    return match get_extension_from_filename(filename) {
        #[cfg(feature = "csv")]
        Some("csv") => LazyCsvReader::new(filename).finish(),
        #[cfg(feature = "parquet")]
        Some("parquet") => LazyFrame::scan_parquet(filename, ScanArgsParquet::default()),
        #[cfg(feature = "ipc")]
        Some("ipc") => LazyFrame::scan_ipc(filename, ScanArgsIpc::default()),
        None => Err(PolarsError::ComputeError(
            format!("Unknown dataframe \"{}\". Either specify a dataframe name registered with \\rd or an absolute path to a file.", filename).into(),
        )),
        Some(ext) => Err(PolarsError::ComputeError(
            format!("Unsupported file extension: \"{}\". Supported file extensions are {} and {}.", ext, SUPPORTED_FILE_EXTENSIONS[0..SUPPORTED_FILE_EXTENSIONS.len() - 1].join(", "), SUPPORTED_FILE_EXTENSIONS.last().unwrap()).into(),
        )),
    };
}

fn create_dataframe_from_tablename(
    context: &mut SQLContext,
    relation: &TableFactor,
) -> PolarsResult<()> {
    match relation {
        TableFactor::Table { name, alias, .. } => {
            let source = name.0.get(0).unwrap().value.as_str();
            let name = match alias {
                Some(alias) => alias.name.value.to_string(),
                None => source.to_string(),
            };

            // Return early if table was already registered.
            if context.table_map.contains_key(&name) {
                return Ok(());
            }

            let frame = create_dataframe_from_filename(source)?;
            context.register(&name, frame);
        }
        // We leave the error for unsupported table types up to SQlContext::execute
        _ => (),
    };

    Ok(())
}

fn create_dataframes_from_statement(context: &mut SQLContext, stmt: &Select) -> PolarsResult<()> {
    let sql_tbl: &TableWithJoins = stmt
        .from
        .get(0)
        .ok_or_else(|| PolarsError::ComputeError("No table name provided in query".into()))?;

    create_dataframe_from_tablename(context, &sql_tbl.relation)?;

    if !sql_tbl.joins.is_empty() {
        for tbl in &sql_tbl.joins {
            create_dataframe_from_tablename(context, &tbl.relation)?;
        }
    }

    Ok(())
}

fn execute_query(context: &mut SQLContext, query: &str) -> PolarsResult<DataFrame> {
    let ast = match Parser::parse_sql(&GenericDialect::default(), query) {
        Ok(ast) => ast,
        Err(e) => {
            return Err(PolarsError::ComputeError(
                format!("Error parsing SQL: {:?}", e).into(),
            ))
        }
    };
    if ast.len() != 1 {
        return Err(PolarsError::ComputeError(
            "One and only one statement at a time please".into(),
        ));
    }

    let ast = ast.get(0).unwrap();
    match ast {
        Statement::Query(query) => {
            match &query.body.as_ref() {
                SetExpr::Select(select_stmt) => {
                    create_dataframes_from_statement(context, &select_stmt)?;
                }
                // Statement is validated in context::execute_statement
                // so we leave it to them to return an error type for unsupported expressions
                _ => (),
            }
        }
        // Statement is validated in context::execute_statement
        // so we leave it to them to return an error type for unsupported statements
        _ => (),
    }

    // Execute SQL command
    return context.execute_statement(ast)?.collect();
}

fn get_extension_from_filename(filename: &str) -> Option<&str> {
    Path::new(filename).extension().and_then(OsStr::to_str)
}

pub fn run_tty() -> std::io::Result<()> {
    let mut stdout = io::stdout();
    let mut context = SQLContext::try_new().unwrap();
    let mut dataframes = Vec::new();
    let mut rl = Editor::<()>::new().unwrap();

    println!("Welcome to Polars CLI. Commands end with ; or \\n");
    println!("Type help or \\? for help.");

    loop {
        let input = match rl.readline(">> ") {
            Ok(line) => {
                rl.add_history_entry(&line);
                line.trim().to_owned()
            }
            Err(ReadlineError::Interrupted) => "exit".to_string(),
            Err(e) => {
                eprintln!("Error: {:?}", e);
                "".to_string()
            }
        };

        let command: Vec<&str> = input.trim().split(" ").collect();
        if command[0].is_empty() {
            continue;
        }

        // Otherwise, execute command
        match command[0] {
            "\\dd" | "dataframes" => print_dataframes(&dataframes),
            "\\rd" | "register" => register_dataframe(&mut context, &mut dataframes, command),
            "\\?" | "help" | "?" | "\\h" => print_help(),
            "\\q" | "quit" | "exit" => {
                println!("Bye");
                return Ok(());
            }
            _ => {
                if command[0].starts_with("\\") {
                    print!("Unknown command: {}\n\n", command[0]);
                    print_help();
                    continue;
                }

                let start = Instant::now();
                match execute_query(&mut context, input.trim()) {
                    Ok(lf) => {
                        println!("{}", lf);
                        println!(
                            "{} rows in set ({:.3} sec)",
                            lf.shape().0,
                            start.elapsed().as_secs_f32()
                        )
                    }
                    Err(e) => eprintln!("{}", e),
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
    let mut input: Vec<u8> = Vec::with_capacity(1024);
    let mut stdin = std::io::stdin();

    loop {
        input.clear();
        stdin.lock().read_until(b';', &mut input);

        let query = std::str::from_utf8(&input).unwrap_or("").trim();
        if query.is_empty() {
            break;
        }

        match execute_query(&mut context, query) {
            Ok(lf) => println!("{}", lf),
            Err(e) => eprintln!("{}", e),
        }
    }

    Ok(())
}
