# Connect to your cloud

Polars Cloud requires connection to your AWS environment to execute queries. After account
registration, you'll set up an [organization](organization/organizations.md) that manages connected
workspaces and team access.

![Set up your organization after registering your account to invite your team and create workspaces](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/polars_cloud/connect-cloud/organization-setup.png)

## Workspace setup

When you first access the Polars Cloud dashboard with a new account, you'll see a notification
indicating your environment requires AWS connection. While you can explore the interface in this
state, executing queries requires completing the setup process.

![An overview of the Polars Cloud dashboard showing a button to connect your cloud environment](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/polars_cloud/connect-cloud/dashboard.png)

The setup process begins with naming your workspace. The default "Personal Workspace" works for
individual use, but consider using team or project names for collaborative environments. This
workspace name becomes part of your [compute context](context/compute-context.md) configuration for
remote query execution.

<!-- dprint-ignore-start -->

!!! tip "Naming your workspace"

    If you’re unsure, you can use a temporary name. You can change the workspace name later under the workspace settings.

<!-- dprint-ignore-end -->

![Connect your cloud screen where you can input a workspace name](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/polars_cloud/connect-cloud/workspace-naming.png)

## AWS infrastructure deployment

The deployment process uses a CloudFormation template to provision the necessary IAM roles and
permissions in your AWS environment. This creates the setup that allows Polars Cloud to execute
queries.

![CloudFormation stack image as step of the setupflow](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/polars_cloud/connect-cloud/cloudformation.png)

For detailed information about the AWS resources and architecture, see
[the AWS Infrastructure page](providers/aws/infra.md).

<!-- dprint-ignore-start -->

!!! info "CloudFormation permissions"

    Users without CloudFormation deployment permissions can easily share the deployment URL with their operations team or AWS administrators.

<!-- dprint-ignore-end -->

CloudFormation stack deployment typically completes within 5 minutes. You can monitor progress
through either the AWS console or the Polars Cloud setup interface for real-time status updates.

![Progress screen in the set up flow](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/polars_cloud/connect-cloud/progress-page.png)

Upon successful deployment, the setup confirms completion and provides access to the full Polars
Cloud dashboard for immediate query execution.

![Final screen of the set up flow indication successful deployment](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/polars_cloud/connect-cloud/successful-setup.png)

You can now run your Polars queries remotely in the cloud. Go to the
[getting started section](quickstart.md) to your first query in minutes,
[learn more how to run queries remotely](context/compute-context.md) or manage your workspace to
invite your team.

<!-- dprint-ignore-start -->

!!! info "One-time workspace setup"

    Cloud environment connection is required once per workspace. Team members invited to connected workspaces can immediately execute remote queries without additional setup.

<!-- dprint-ignore-end -->

## Start your next project

With your workspace connected to AWS, you can execute Polars queries remotely. Consider these next
actions:

- Run your first remote query with the [getting started guide](quickstart.md)
- Learn about [compute context configuration](context/compute-context.md) for performance
  optimization
- Invite [team members](workspace/team.md) to your connected workspace to collaborate on your next
  project.
