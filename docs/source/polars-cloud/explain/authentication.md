# Logging in

Polars cloud allows authentication through short-lived authentication tokens. There are two ways you
can obtain an access token:

- command line interface
- python client

After a successful login Polars Cloud stores the token in `{$HOME}/.config/polars_cloud`. You can
alter this path by setting the environment variable `POLARS_CLOUD_CONFIG_DIR`.

### Command Line Interface (CLI)

Authenticate with CLI using the following command

```bash
pc authenticate
```

### Python client

Authenticate with the Polars Cloud using

{{code_block('polars-cloud/authentication','authenticate',['authenticate'])}}

Both methods will attempt to authenticate with the following priority:

1. `POLARS_CLOUD_ACCESS_TOKEN` environment variable
1. `POLARS_CLOUD_CLIENT_ID` / `POLARS_CLOUD_CLIENT_SECRET` environment variables
1. Any cached access or refresh tokens

If all methods fail the user will be redirected to the browser to login and obtain a new access
token. If you are in a non-interactive workflow and want to fail if there is no valid token you can
run `pc.authenticate(interactive=False)` instead.

## Service accounts

For non-interactive workflows such as orchestration tools there are service accounts. These allow
you to login programmatically.

To create a service account go to the Polars Cloud dashboard under Settings and service accounts.
Here you can create a new service account for your workspace. To authenticate set the
`POLARS_CLOUD_CLIENT_ID` and `POLARS_CLOUD_CLIENT_SECRET` environment variables. Polars Cloud will
automatically pick these up if there are no access tokens present in the path.
