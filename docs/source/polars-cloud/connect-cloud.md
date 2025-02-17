# Connect cloud environment

To use Polars Cloud, you have to connect all your workspaces a cloud environment.

If you login to the Polars Cloud dashboard for the first time, you will notice a blue bar at the top of the screen. When you created a new account that is not connected to a cloud environment, you can explore Polars Cloud but you cannot execute any queries yet.

![An overview of the Polars Cloud dashboard showing a button to connect your cloud environment](../assets/connect-cloud/dashboard.png)

When you click the blue bar you will be redirected to the start of the set up flow. In this first step you can name your workspace.

## 1. Set workspace name

In the first step of the setup flow you give a name to your workspace. You can keep the name "Personal Workspace" or use the name of your team/department. This workspace name will be required by the compute context to execute a query remote.

!!! tip "Naming your workspace"
    If you are not sure you can use a temporary name, You can change the name of workspace under the workspace settings at any moment.

![Connect your cloud screen where you can input a workspace name](../assets/connect-cloud/workspace-naming.png)

## 2. Deploy to AWS

When you have entered a name, you can click "Deploy to Amazon". This will open a screen in AWS with a CloudFormation template that is required to install the required roles in your AWS environment.

![CloudFormation stack image as step of the setupflow](../assets/connect-cloud/cloudformation.png)

If you want to learn more about what Polars Cloud installs in your environment, you can read more on [the AWS Infrastructure page](../providers/aws/infra).

!!! info "No permissions to deploy the stack in AWS"
    If you don't have the required persmissions to deploy CloudFormation stacks in your AWS environment, you can copy the URL and share it with your operations team or someone with the permissions. With the URL they can deploy the stack for you.

## 3. Deploying the environment

After you have "Create stack", the CloudFormation stack will be deployed in your environment. This will take around 5 minutes. You can follow the progress in your AWS environment or in the Polars set up flow.

![Progress screen in the set up flow](../assets/connect-cloud/progress-page.png)

When the CloudFormation stack is deployed you will see a confirmation message.

![Final screen of the set up flow indication successful deployment](../assets/connect-cloud/successful-setup.png)

If you click "Start exploring", you will be redirected to the Polars Cloud dashboard.

You can now run your Polars query remotely in the cloud. Go to the [getting started section](../quickstart) to your first query in minutes, [learn more how to run queries remote](../run/compute-context) or manage your workspace to invite your team.

!!! info "Only connect a workspace once"
    You only have to connect your workspace once. If you invite your team to a workspace that is connected to a cloud environment they can immediately run queries remotely.
