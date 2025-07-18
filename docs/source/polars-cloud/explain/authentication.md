# Logging in

Polars cloud allows authentication through short-lived authentication tokens. There are two main
ways to authenticate:

1. `authenticate` Loads cached tokens, otherwise opens the browser to log in and caches them
2. `login` Opens the browser to log in, caches tokens but does not reuse them

Generally you would use `authenticate` unless you want to switch between accounts.

Both methods will attempt to authenticate with the following priority:

1. `POLARS_CLOUD_ACCESS_TOKEN` environment variable
2. `POLARS_CLOUD_CLIENT_ID` / `POLARS_CLOUD_CLIENT_SECRET` service account environment variables
3. Any cached access or refresh tokens

If all methods fail, the user will be redirected to the browser to log in and get a new access
token. If you are in a non-interactive workflow and want to fail if there is no valid token you can
run `pc.authenticate(interactive=False)` instead.

After successful authentication, Polars Cloud stores the token in your OS config directory:

| OS      | Value                                  | Example                                  |
| ------- | -------------------------------------- | ---------------------------------------- |
| Linux   | `$XDG_CONFIG_HOME` or `$HOME/.config/` | home/alice/.config                       |
| macOS   | `$HOME/Library/Application Support`    | /Users/Alice/Library/Application Support |
| Windows | `{FOLDERID_RoamingAppData}`            | C:\Users\Alice\AppData\Roaming           |

You can override this path by setting the environment variable `POLARS_CLOUD_ACCESS_TOKEN_PATH`.

### Commands

You can authenticate from the terminal:

```bash
pc authenticate
pc login
```

Or in Python:

{{code_block('polars-cloud/authentication','login',['login'])}}

## Service accounts

Both flows described above are for interactive logins where a person is present in the process. For
non-interactive workflows such as orchestration tools there are service accounts. These allow you to
login programmatically. See the page on how to set up and
[use service acccounts](service-accounts.md) for more information.
