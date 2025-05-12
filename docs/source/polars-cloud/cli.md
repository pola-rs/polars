# CLI

Polars cloud comes with a command line interface (CLI) out of the box. This allows you to interact
with polars cloud resources from the terminal.

```bash
pc --help

usage: pc [-h] [-v] [-V] {authenticate,setup,login,organization,workspace,compute} ...

positional arguments:
  {authenticate,setup,login,organization,workspace,compute}
    authenticate        Authenticate with Polars Cloud by loading stored credentials or otherwise logging in through the browser.
    setup               Set up an organization and workspace to quickly run queries. Ideal to get started with Polars Cloud.
    login               Authenticate with Polars Cloud by logging in through the browser.
    organization        Manage Polars Cloud organizations.
    workspace           Manage Polars Cloud workspaces.
    compute             Manage Polars Cloud compute clusters.

optional arguments:
  -h, --help            show this help message and exit.
  -v, --verbose         Output debug logging messages.
  -V, --version         Display the version of the Polars Cloud client.
```

If you're just starting out with Polars Cloud then `pc setup` will guide you through setting up your
environment to be able to quickly run queries.
