# Logging in

Polars cloud allows authentication through short-lived authentication tokens. There are two ways you
can obtain an access token:

- command line interface
- python client

After a successful `login` Polars Cloud stores the token in `{$HOME}/.polars`. You can alter this
path by setting the environment variable `POLARS_CLOUD_ACCESS_TOKEN_PATH`.

### Command Line Interface (CLI)

Authenticate with CLI using the following command

```bash
pc login
```

### Python client

Authenticate with the Polars Cloud using

{{code_block('polars-cloud/authentication','login',['login'])}}

Both methods redirect you to the browser where you can provide your login credentials and continue
the sign in process.

## Service accounts

Both flows described above are for interactive logins where a person is present in the process. For
non-interactive workflows such as orchestration tools there are service accounts. These allow you to
login programmatically.

To create a service account go to the Polars Cloud dashboard under Settings and service accounts.
Here you can create a new service account for your workspace. To authenticate set the
`POLARS_CLOUD_CLIENT_ID` and `POLARS_CLOUD_CLIENT_SECRET` environment variables. Polars Cloud will
automatically pick these up if there are no access tokens present in the path.
