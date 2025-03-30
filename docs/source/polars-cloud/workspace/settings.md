# Configuring workspace settings

The Workspace Settings page provides a centralized interface for managing your workspace
configuration.

## General

The General section contains basic information about your workspace:

- **Workspace Name**: Displays the current name of your workspace. You can modify this by clicking
  the "Edit" button.
- **Description**: Provides space for a detailed description of your workspace's purpose. The
  description can be modified by clicking the "Edit" button.

## Default Compute Configuration

This section allows you to define the default computational resources allocated to jobs and
processes within your workspace.

- **Set Cluster Defaults**: Clicking this button lets you configure either a Resource based default
  or Instance based default.
  - With Resource based custom amount of vCPUs, RAM, Storage and cluster size can be defined.
  - Instance based allows to select a default instance type from the cloud service provider.

Setting default compute configurations eliminates the need to explicitly define a compute context.
More information on configuration can be found in the section on
[setting up compute context](../run/compute-context.md).

## Query and compute labels

Labels help organize and categorize queries and compute within your workspace. The labels can only
be set from the dashboard.

```python
ctx= plc.ComputeContext(
    workspace="PolarsCloudDemo",
    labels=["marketing", "cltv"],
)
```

- **Create New Label**: Create custom labels that can be applied to various resources for better
  organization and insight in usage.

## Service Accounts

The Service Accounts section manages programmatic access to your workspace.

- **Token Description**: Displays the purpose of each service account token.
- **Created At**: Shows when a service account was created.
- **Action**: Provides options to delete service accounts.
- **Create New Service Account**: Allows you to generate new service accounts for API access and
  automation.

You can find more information about using Service Accounts in your workflows in the "Use service
accounts" section.

## Disable workspace

Users have the ability to disable the entire workspace. This should be used with caution. Disabling
the workspace will stop processes, remove access and permanently delete all data and content within
the workspace.
