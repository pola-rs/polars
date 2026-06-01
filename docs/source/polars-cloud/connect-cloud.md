# Connect to your cloud

Polars Cloud requires connection to your cloud environment to execute queries. After account
registration, a guided onboarding flow walks you through creating an
[organization](organization/organizations.md), selecting a deployment stack, and connecting your
first workspace.

![Create your organization step during the registration onboarding flow](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/polars_cloud/connect-cloud/onboarding-create-org.png)

## Registration onboarding

After creating your account, a three-step flow completes the initial setup:

1. **Create your account**: Already done.
2. **Create your organization**: name the organization that manages workspaces and team access.
3. **Select your stack**: choose a deployment type for your first workspace.

![Select your stack screen showing AWS and Kubernetes deployment options](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/polars_cloud/connect-cloud/onboarding-select-stack.png)

Two deployment types are available:

- **AWS**: workloads run on AWS-managed services, deployed via a CloudFormation template.
- **Kubernetes**: workloads run on a Kubernetes cluster (EKS, GKE, AKS, or on-premises),
  deployed via Helm.

For Kubernetes setup, see the
[EKS](../polars-on-premises/kubernetes/cloud-providers/amazon-elastic-kubernetes-service.md),
[GKE](../polars-on-premises/kubernetes/cloud-providers/google-kubernetes-engine.md), or
[AKS](../polars-on-premises/kubernetes/cloud-providers/azure-kubernetes-service.md) guides. The
rest of this page covers the AWS path.

<!-- dprint-ignore-start -->

!!! tip "Naming your workspace"

    If you're unsure, you can use a temporary name. You can change the workspace name later under the workspace settings.

<!-- dprint-ignore-end -->

Clicking **Create default workspace** completes registration and opens the dashboard with a guided
setup checklist.

## Workspace setup

When you first access the dashboard after registration, a five-step setup checklist guides you
through connecting your workspace.

![Dashboard showing the five-step workspace setup checklist](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/polars_cloud/connect-cloud/dashboard-setup-steps.png)

The checklist steps are:

1. Create workspace
2. Link your AWS account
3. Deploy CloudFormation stack
4. Invite members
5. Run your first query

## AWS infrastructure deployment

The **Link your AWS account** step provides three options depending on your AWS access:

- **Deploy to AWS**: opens the CloudFormation quick-create URL directly in your browser.
- **Deploy to AWS in an existing VPC**: uses your existing VPC rather than creating a new one.
- **Copy the setup link**: share the deployment URL with your operations team or AWS administrator.

<!-- dprint-ignore-start -->

!!! info "CloudFormation permissions"

    Users without CloudFormation deployment permissions can easily share the deployment URL with their operations team or AWS administrators.

<!-- dprint-ignore-end -->

The CloudFormation quick-create page is pre-filled with the template URL and a stack name derived
from your workspace name. You don't have to change anything. 

![CloudFormation quick-create page pre-filled with the Polars Cloud template](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/polars_cloud/connect-cloud/cloudformation-quick-create.png)

Before clicking **Create stack**, scroll to the bottom of the page and check the acknowledgment
that the template creates IAM resources.

![AWS CloudFormation IAM capabilities acknowledgment before creating the stack](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/polars_cloud/connect-cloud/cloudformation-iam-capabilities.png)



For detailed information about the AWS resources and architecture, see
[the AWS Infrastructure page](providers/aws/infra.md).

Back in the Polars Cloud dashboard, the **Deploy CloudFormation stack** step tracks progress through
five stages: Network → Security → Policies → Deploying → Connected.

![CloudFormation deployment progress tracker showing stages from Network to Connected](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/polars_cloud/connect-cloud/cloudformation-progress.png)

Stack deployment typically completes within 5 minutes.

## Invite members

After the stack is deployed, you can invite team members to your workspace by email.

![Invite members step with email input field](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/polars_cloud/connect-cloud/invite-members.png)

<!-- dprint-ignore-start -->

!!! info "One-time workspace setup"

    Cloud environment connection is required once per workspace. Team members invited to connected workspaces can immediately execute remote queries without additional setup.

<!-- dprint-ignore-end -->

This step is optional and you can click **skip** to continue. Members can be added later from the
[Team page](workspace/team.md).

## Run your first query

The final setup step provides a sample query to verify your workspace is connected and ready.

![Run your first query step with sample uv code snippet](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/polars_cloud/connect-cloud/run-first-query.png)

The snippet uses `uv` for a self-contained run. If you don't have it yet, you can follow the provided link on the dashboard to install it. On first execution, a browser window opens to
authenticate. After connecting your session to Polars Cloud, credentials are cached locally for the duration of your session.

Once the query completes successfully, the setup confirms your workspace is fully configured.

![You're all set! The workspace is fully configured and ready to run queries](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/polars_cloud/connect-cloud/setup-complete.png)

## Start your next project

With your workspace connected to AWS, you can execute Polars queries remotely. Consider these next
actions:

- Learn how to [run queries remotely](run/remote-query.md) and get the most out of Polars Cloud
- [Profile your queries](run/query-profile.md) to understand and optimize performance
- Learn about [compute context configuration](context/compute-context.md) for performance
  optimization
- Invite [team members](workspace/team.md) to your connected workspace to collaborate on your next
  project.
