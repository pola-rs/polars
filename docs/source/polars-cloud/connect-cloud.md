# Connect cloud environment

To use Polars Cloud, you must connect your workspaces to a cloud environment.

If you log in to the Polars Cloud dashboard for the first time with an account that isn’t connected
to a cloud environment, you will see a blue bar at the top of the screen. You can explore Polars
Cloud in this state, but you won’t be able to execute any queries yet.

![An overview of the Polars Cloud dashboard showing a button to connect your cloud environment](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/polars_cloud/connect-cloud/dashboard.png)

When you click the blue bar you will be redirected to the start of the set up flow.

## 1. Set workspace name

In the first step of the setup flow, you’ll name your workspace. You can keep the default name
"Personal Workspace" or use the name of your team or department. This workspace name will be used by
the compute context to run your queries remotely.

<!-- dprint-ignore-start -->

!!! tip "Naming your workspace"
    If you’re unsure, you can use a temporary name. You can change the workspace name later under the workspace settings.

<!-- dprint-ignore-end -->

![Connect your cloud screen where you can input a workspace name](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/polars_cloud/connect-cloud/workspace-naming.png)

## 2. Deploy to AWS

After naming your workspace, click Deploy to Amazon. This opens a screen in AWS with a
CloudFormation template. This template installs the necessary roles in your AWS environment.

![CloudFormation stack image as step of the setupflow](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/polars_cloud/connect-cloud/cloudformation.png)

If you want to learn more about what Polars Cloud installs in your environment, you can read more on
[the AWS Infrastructure page](providers/aws/infra.md).

<!-- dprint-ignore-start -->
!!! info "No permissions to deploy the stack in AWS"
    If you don't have the required permissions to deploy CloudFormation stacks in your AWS environment, you can copy the URL and share it with your operations team or someone with the permissions. With the URL they can deploy the stack for you.
<!-- dprint-ignore-end -->

## 3. Deploying the environment

After you click Create stack, the CloudFormation stack will be deployed in your environment. This
process usually takes around 5 minutes. You can monitor the progress in your AWS environment or in
the Polars setup flow.

![Progress screen in the set up flow](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/polars_cloud/connect-cloud/progress-page.png)

When the CloudFormation stack deployment completes, you’ll see a confirmation message.

![Final screen of the set up flow indication successful deployment](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/polars_cloud/connect-cloud/successful-setup.png)

If you click "Start exploring", you will be redirected to the Polars Cloud dashboard.

You can now run your Polars query remotely in the cloud. Go to the
[getting started section](quickstart.md) to your first query in minutes,
[learn more how to run queries remote](run/compute-context.md) or manage your workspace to invite
your team.

<!-- dprint-ignore-start -->

!!! info "Only connect a workspace once"
    You only have to connect your workspace once. If you invite your team to a workspace that is connected to a cloud environment they can immediately run queries remotely.

<!-- dprint-ignore-end -->
