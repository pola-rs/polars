use std::ffi::OsStr;
use std::io::{self, BufRead, Write};
use std::path::Path;

#[cfg(feature = "csv")]
use polars_lazy::frame::LazyCsvReader;
use polars_lazy::frame::LazyFrame;
#[cfg(feature = "parquet")]
use polars_lazy::frame::ScanArgsParquet;
use polars_sql::SQLContext;

use sqlparser::ast::{
    Expr as SqlExpr, JoinOperator, OrderByExpr, Select, SelectItem, SetExpr, Statement,
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

fn execute_query(context: &SQLContext, query: &str) {
    let ast = Parser::parse_sql(&GenericDialect::default(), query).unwrap();
    if ast.len() != 1 {
        println!("One and only one statement at a time please");
        return;
    }
    let ast = ast.get(0).unwrap();
    // TODO: Attempt to unwrap into SetExpr::Select
    // TODO: Register dataframes in SQLContext on the fly from FROM clause

    // Execute SQL command
    let out = match context.execute_statement(&ast) {
        Ok(q) => q.limit(100).collect(),
        Err(e) => Err(e),
    };

    match out {
        Ok(df) => println!("{}", df),
        Err(e) => println!("{}", e),
    }
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

pub fn run() -> io::Result<()> {
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
            _ => execute_query(&context, input.trim()),
        }

        println!();
    }
}

fn get_extension_from_filename(filename: &str) -> Option<&str> {
    Path::new(filename).extension().and_then(OsStr::to_str)
}
