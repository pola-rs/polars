# Logging in

Polars Cloud allows authentication through short-lived authentication tokens. Authentication also
supports programmatic workflows.

1. `authenticate` loads cached tokens when available, falling back to browser-based login when
   needed. This is the recommended method for most development workflows since it reuses existing
   sessions.
2. `login` always initiates browser-based authentication. Use this when you need to switch between
   accounts.

You can find more information about the usage of authentication options in the
[API Reference documentation](https://docs.cloud.pola.rs/reference/auth/index.html).

## Credential resolution

Both methods will attempt to authenticate with the following priority:

1. `POLARS_CLOUD_ACCESS_TOKEN` environment variable
2. `POLARS_CLOUD_CLIENT_ID` / `POLARS_CLOUD_CLIENT_SECRET` service account environment variables
3. Any cached access or refresh tokens

When all methods fail, the user will be redirected to the browser to log in and get a new access
token. If you are in a non-interactive workflow and want to fail if there is no valid token you can
run `pc.authenticate(interactive=False)` instead.

After successful authentication, Polars Cloud stores the token in your OS config directory:

| OS      | Value                                  | Example                                  |
| ------- | -------------------------------------- | ---------------------------------------- |
| Linux   | `$XDG_CONFIG_HOME` or `$HOME/.config/` | home/alice/.config                       |
| macOS   | `$HOME/Library/Application Support`    | /Users/Alice/Library/Application Support |
| Windows | `{FOLDERID_RoamingAppData}`            | C:\Users\Alice\AppData\Roaming           |

You can override this path by setting the environment variable `POLARS_CLOUD_ACCESS_TOKEN_PATH`.

## Service accounts

Service accounts provide programmatic authentication for automated workflows, orchestration tools,
and CI/CD pipelines. These accounts facilitate automated processes (typically in production
environments) without requiring interactive login sessions. See the page on
[service accounts](service-accounts.md) for more information.
