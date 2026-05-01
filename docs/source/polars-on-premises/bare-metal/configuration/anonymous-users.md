# Anonymous users

By default, queries can be anonymously submitted to the scheduler. In the dashboard, queries
submitted anonymously are shown as `Anonymous User` in the query list. To attach a username to a
query, you can use the `POLARS_CLOUD_USER_NAME` environment variable or the
`pc.Config.set_user_name()` method.

```python
import polars_cloud as pc
pc.Config.set_user_name("username@example.com")
```

To deny all queries that don't have any username attached, you can set the `deny_anonymous_users`
configuration option to `true` in the scheduler section of the configuration file.

```toml
[scheduler]
enabled = true
deny_anonymous_users = true
```

!!! note "Difference between Anonymous Users and Anonymous Results"

    Note that Anonymous Users and Anonymous Results are different. Anonymous Users refer to queries that are submitted without a username, while Anonymous Results refer to queries without an explicit output sink.
