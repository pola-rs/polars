# License

Polars on-premises requires a license key to run. If you haven't contacted Polars yet,
[sign up here to apply](https://w0lzyfh2w8o.typeform.com/to/zuoDgoMv). A license key looks like
this:

```json
{ "params": { "expiry": "2026-01-31T23:59:59Z", "name": "Company" }, "signature": "..." }
```

At the top level of the configuration file, you need to specify the path to the license file using
the `license` key. For example:

```toml
cluster_id = "polars-cluster-dev"
instance_id = "scheduler"
license = "/etc/polars/license.json"
# ...
```

The cluster verifies the license key periodically, and will shutdown once the license expires.

## EULA license

Polars on-premises is licensed under the End User License Agreement (EULA) which can be found in the
Polars on-premises binary. The EULA must be accepted by setting the `POLARS_EULA_ACCEPTED`
environment variable to `1`. If the environment variable is not set, the executable will print the
EULA and exit. You can also manually print the EULA using the following command:

```bash
polars-on-premises --print-eula
```
