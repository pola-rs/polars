# License

Polars on-premises requires a license key to run. If you haven't contacted Polars yet,
[sign up here to apply](https://w0lzyfh2w8o.typeform.com/to/zuoDgoMv). A license key looks like
this:

```json
{ "params": { "expiry": "2026-01-31T23:59:59Z", "name": "Company" }, "signature": "..." }
```

Create a secret for the received offline license key.

```shell
kubectl create secret generic polars-offline-license --from-file=license.json=license.json
```

In the Helm values, you need to specify the secret name and key. For example:

```yaml
license:
    secretName: polars-offline-license
    secretProperty: license.json
```

The cluster verifies the license key periodically, and will shutdown once the license expires.

## EULA license

Polars on-premises is licensed under the End User License Agreement (EULA) which can be found in the
Polars on-premises binary. The EULA must be accepted by setting the `POLARS_EULA_ACCEPTED`
environment variable to `1`. If the environment variable is not set, the executable will print the
EULA and exit.
