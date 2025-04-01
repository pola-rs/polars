# CLI

Polars cloud comes with a command line interface (CLI) out of the box. This allows you to interact
with polars cloud resources from the terminal.

```bash
pc --help
```

```
usage: pc [-h] [-v] [-V] {login,workspace,compute} ...

positional arguments:
{login,workspace,compute}
login               Authenticate with Polars Cloud by logging in through the browser
workspace           Manage Polars Cloud workspaces.
compute             Manage Polars Cloud compute clusters.

options:
-h, --help            show this help message and exit
-v, --verbose         Output debug logging messages.
-V, --version         Display the version of the Polars Cloud client.
```

### Authentication

You can authenticate with Polars Cloud from the CLI using

```bash
pc login
```

This refreshes your access token and saves it to disk.

### Workspaces

Create and setup a new workspace

```bash
pc workspace setup
```

List all workspaces

```bash
pc workspace list
```

```
NAME           	ID                                    STATUS
test-workspace 	0194ac0e-5122-7a90-af5e-b1f60b1989f4  Active
polars-ci-2025…	0194287a-e0a5-7642-8058-0f79a39f5b98  Uninitialized
```

### Compute
