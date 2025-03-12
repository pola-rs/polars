# Infrastructure

Polars Cloud manages the hardware for you by spinning up and down raw EC2 instances. In order to do
this it needs permissions in your own cloud environment. None of the resources below have costs
associated with them. While no compute clusters are running Polars Cloud will not create any AWS
costs. The recommended way of doing this is running `pc workspace setup`.

## Recommended setup

When you deploy Polars Cloud the following infrastructure is setup.

<figure markdown="span">
![AWS infrastructure](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/polars_cloud/aws-infra.png)
</figure>

1. A `VPC` and `subnet` in which Polars EC2 workers can run.
1. Two `security groups`. One for batch mode, which does not have any public ports, and one for
   interactive mode, which allows direct communication between your local environment and the
   cluster.
1. `PolarsWorker` IAM role. Polars EC2 workers run under this IAM role.
1. `UserInitiated` & `Unattended` IAM role. The `UserInitiated` role has the permissions to start
   Polars EC2 workers in your environment. The `Unattended` role can terminate unused compute
   clusters that you might have forgot about.

## Security

By design Polars Cloud never has access to the data inside your cloud environment. The data never
leaves your environment.

### IAM permissions

The list below show an overview of the required permissions for each of the roles.

??? User Initiated

    - ec2:CreateTags
    - ec2:RunInstances
    - ec2:DescribeInstances
    - ec2:DescribeInstanceTypeOfferings
    - ec2:DescribeInstanceTypes
    - ec2:TerminateInstances
    - ec2:CreateFleet
    - ec2:CreateLaunchTemplate
    - ec2:CreateLaunchTemplateVersion
    - ec2:DescribeLaunchTemplates

??? Unattended

    - ec2:DescribeInstances
    - ec2:TerminateInstances
    - ec2:DescribeFleets
    - ec2:DeleteLaunchTemplate
    - ec2:DeleteLaunchTemplateVersions
    - ec2:DeleteFleets
    - sts:GetCallerIdentity
    - sts:TagSession
    - cloudwatch:GetMetricData
    - logs:GetLogEvents
    - logs:FilterLogEvents
    - logs:DescribeLogStreams

??? Worker

    - logs:CreateLogGroup 
    - logs:PutRetentionPolicy 
    - cloudwatch:PutMetricData

## Custom setup

Depending on your enterprise needs or existing infrastructure, you may not require certain
components (e.g. VPC, subnet) of the default setup of Polars Cloud. Or you have additional security
requirements in place. Together with our team of engineers we can integrate Polars Cloud with your
existing infrastructure. Please contact us directly.
