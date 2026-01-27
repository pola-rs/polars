# Workspace configuration

On the workspace settings page you can set cluster defaults, create labels, add service accounts and
change other workspace related settings.

## Workspace configuration

Configure workspace identity through the name and description fields. Use descriptive naming
conventions that align with your project structure or data domains for easier workspace discovery
and management.

## Default Compute Configuration

Set default computational resources to standardize job execution without requiring explicit compute
context configuration for each operation. Two configuration modes are available:

**Resource-based** configuration allows precise specification of vCPUs, RAM, storage, and cluster
size. Use this approach when you require control over resource allocation or have specific
performance/cost optimization requirements. When the specific configuration is not available on AWS,
the cheapest instance with at least the specifications is selected and used.

**Instance-based** configuration leverages cloud provider instance types. This approach simplifies
configuration by using pre-optimized instance configurations and is typically more cost-effective
for standard workloads.

Default compute configurations eliminate the need to explicitly define a compute context. More
information on configuration can be found in the section on
[setting the compute context](../context/compute-context.md).

## Query and compute labels

Labels help organize and categorize queries and compute within your workspace. Labels can only be
managed through the dashboard interface.

```python
ctx= plc.ComputeContext(
    workspace="your-workspace",
    labels=["docs", "user-guide"],
)
```

## Service Account Management

Service accounts allow scripts to authenticate without human intervention. Each service account
generates an authentication token with workspace-scoped permissions.

Use clear and descriptive names that match your project structure or data domain to improve clarity.
Refer to the [Use service accounts](../explain/service-accounts.md) section for implementation
patterns and security best practices.

## Disable workspace

Workspace admins can disable workspaces removing all data, terminating active processes, and
revoking all access tokens. This action cannot be reversed.

Ensure data export and backup completion before disabling workspaces containing critical datasets or
analysis artifacts.
