# CLI

Polars cloud comes with a command line interface (CLI) out of the box. This allows you to interact
with polars cloud resources from the terminal.

```bash
pc --help
```

```
Command line interface for Polars Cloud

Usage: pc [OPTIONS] [COMMAND]

Commands:
  authenticate     Authenticate with Polars Cloud
  login            Login through the browser
  setup            Set up organization and workspace
  organization     Manage Polars Cloud organizations
  workspace        Manage Polars Cloud workspaces
  compute          Manage Polars Cloud compute clusters
  service-account  Manage Polars Cloud service accounts
  help             Print this message or the help of the given subcommand(s)

Options:
  -v, --verbose                  Output debug logging messages.
  -t, --token <TOKEN>            Authentication token to override other auth methods
  -p, --token-path <TOKEN_PATH>  Path to authentication token file
  -h, --help                     Print help
  -V, --version                  Print version
```

If you're just starting out with Polars Cloud then `pc setup` will guide you through setting up your
environment to be able to quickly run queries.
