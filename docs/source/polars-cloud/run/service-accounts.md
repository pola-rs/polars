# Using service accounts

Service accounts function as programmatic identities that enable secure machine-to-machine
communication for remote data processing workflows. These accounts facilitate automated processes
(typically in production environments) without requiring interactive login sessions.

## Create a service account

In the workspace settings page, navigate to the Service Accounts section. Here, you can create and
manage service accounts associated with your workspace.

To create a new Service Account:

1. Click the Create New Service Account button
2. Provide a name and description for the account
3. Copy and securely store the Client ID and Client Secret

<!-- dprint-ignore-start -->

!!! info "Client ID and Secret visible once"
    If you lose the Client ID or Client Secret, you will need to generate a new service account.

<!-- dprint-ignore-end -->

## Set up your environment to use a Service Account

To authenticate using a Service Account, you must set environment variables. The following variables
should be defined using the credentials provided when the service account was created:

```bash
export POLARS_CLOUD_CLIENT_ID ="CLIENT_ID_HERE"
export POLARS_CLOUD_CLIENT_SECRET="CLIENT_SECRET_HERE"
```

## Execute query with Service Account

Once the environment variables are set, you do not need to log in to Polars Cloud manually. Your
process will automatically connect to the workspace.

## Revoking access or deleting a Service Account

When a service account is no longer needed, it is recommended to revoke its access. Keep in mind
that deleting a service account is irreversible, and any processes or applications relying on this
account will lose access immediately.

1. Navigate to the Workspace Settings page.
2. Go to the Service Accounts section.
3. Locate the service account to delete.
4. Click the three-dot menu next to the service account and select Delete.
5. Confirm the deletion of the Service Account.
