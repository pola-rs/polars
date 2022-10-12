use std::io::{self, Write, BufRead};
use polars_lazy::frame::LazyCsvReader;
use polars_sql::SQLContext;


// Command: /dd or dataframes
fn print_dataframes(dataframes: &Vec<(String, String)>) {
    println!("{} dataframes registered{}", dataframes.len(), if dataframes.is_empty() { "." } else { ":" });
    for (name, file) in dataframes.iter() {
        println!("{}:\t {}", name, file);
    }
}

// Command: /? or help
fn print_help() {
    for (name, short, desc) in vec![
        ("dataframes", "dd", "Show registered frames."),
        ("help", "?", "Display this help."),
        ("register", "rd", "Register new dataframe: \\rd <name> <source>"),
        ("quit", "q", "Exit")
    ].iter() {
        println!("{:20}\\{:10} {}", name, short, desc);
    }
}

fn execute_query(context: &SQLContext, query: &str) {
    // Execute SQL command
    let out = match context.execute(&query) {
        Ok(q)=> q.limit(100).collect(),
        Err(e) => Err(e),
    };
    
    match out {
        Ok(df) => println!("{}", df),
        Err(e) => println!("{}", e),
    }
}

fn register_dataframe(context: &mut SQLContext, dataframes: &mut Vec<(String, String)>, command: Vec<&str>) {
    if command.len() < 3 {
        println!("Usage: \\rd <name> <file>");
        return;
    } 
    let name = command[1];
    let source = command[2];

    match LazyCsvReader::new(source).finish() {
        Ok(frame) => {
            context.register(name, frame);
            dataframes.push((name.to_owned(), source.to_owned()));
            println!("Added dataframe \"{}\" from file {}", name, source)
        },
        Err(e) => println!("{}", e)
    }              
}

fn main() -> io::Result<()> {
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
        
        let command : Vec<&str> = input.trim().split(" ").collect();
        if command[0].is_empty() {
            continue;
        }

        match command[0] {
            "\\dd" | "dataframes" => print_dataframes(&dataframes),
            "\\rd" | "register" => register_dataframe(&mut context, &mut dataframes, command),
            "\\?" | "help" => print_help(),
            "\\q" | "quit" => {
                println!("Bye");
                return Ok(())
            },
            _ => execute_query(&context, input.trim()),
        }
        
        println!();
    }
}
