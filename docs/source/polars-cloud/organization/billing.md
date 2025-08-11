# Payment and billing

All payments and billing for Polars Cloud are handled via the AWS Marketplace. This integration
provides a secure, streamlined way to manage your subscription, payments, and billing using AWS's
trusted marketplace infrastructure.

!!! tip "30-day Free Trial"

    New organizations get full access to explore Polars Cloud for 30 days. The trial activates automatically when connecting your first workspace to AWS. After the trial you can easily subscribe via the AWS Marketplace.

## Pricing

Polars Cloud uses usage-based pricing charged per vCPU hour of compute usage. Billing begins when
instances are booted and ready to execute queries, and ends when users stop the instances or when we
detect instances have been terminated externally. Instance startup time and shutdown time after
stopping are not charged.

## Subscribe to Polars Cloud

Subscribing to Polars Cloud requires an AWS account with billing permissions. You can access the
marketplace listing either through the Billing page in your Polars Cloud organization or by
searching "Polars Cloud" directly in AWS Marketplace.

The subscription process connects your Polars Cloud organization to AWS billing, after which your
organization status updates to reflect the marketplace connection. This typically completes within
minutes, though allow up to an hour for status propagation.

### Verify Your Organization Connection

After subscribing via Marketplace, you'll be redirected to the Polars Cloud organization dashboard.
Under `Billing`, the status will update to show your organization is connected to AWS Marketplace.
If the status doesn't update within 15 minutes, contact our support team at
[support@polars.tech](mailto:support@polars.tech).

### Request and Accept Private Offers

Customers with large-scale analytics workloads or enterprise customers requiring custom pricing,
annual commitments, or specific terms can contact our team for a private offer. Reach out to the
team at [support@polars.tech](mailto:support@polars.tech).

### Cost and Usage Monitoring

All up to date data about Polars Cloud usage and costs can be found in the
[AWS Cost Explorer](https://aws.amazon.com/aws-cost-management/aws-cost-explorer/). You can find
more information about usage per workspace in the Polars Cloud dashboard.

## Manage Your Subscription

### Cancelling Your Subscription

Cancel through the AWS Marketplace subscription management interface.

**Important**: Unsubscribing will instantly stop all active queries in your organization. Ensure no
critical workflows are running before cancelling.

### Reactivate Subscription

After cancelling your subscription, you can still access Polars Cloud but no workflows can be
started. You can easily reactivate the subscription following the steps on the `Billing` page in
Polars Cloud or via the AWS Marketplace.
