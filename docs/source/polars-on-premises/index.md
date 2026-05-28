# Polars On-Prem

Polars On-Prem lets you run Polars clusters inside your own infrastructure, whether that's a managed
Kubernetes service or bare metal. It is designed for teams that;

- have data needs beyond a single machine;
- require full control over their infrastructure;
- want elaborative query profiling;
- want lineage support.

If you want immediately start with running Polars On-Prem on kubernetes, follow the
[getting-started](./getting-started) guide.

<video autoplay muted loop playsinline style="width:100%;border-radius:8px;margin:1rem 0">
  <source src="https://pola.rs/query-profiler-video.mp4" type="video/mp4">
</video>

## Supported platforms

Polars On-Prem supports two deployment paths:

**Kubernetes:** runs on any Kubernetes distribution, including:

- Google Kubernetes Engine (GKE)
- Azure Kubernetes Service (AKS)
- Amazon Elastic Kubernetes Service (EKS)
- Any other CNCF-conformant Kubernetes distribution

**Bare metal:** for servers without Kubernetes. Polars On-Prem bootstraps the cluster directly on
your machines, no Kubernetes installation required.

## Plans

Polars On-Prem offers two plans depending on how much connectivity you want between your clusters
and the Polars Cloud portal.

|                           | Self-Serve                            | Enterprise                                       |
| ------------------------- | ------------------------------------- | ------------------------------------------------ |
| License                   | Download from Polars Cloud portal     | Contact us                                       |
| Cluster connectivity      | Reports query history to cloud portal | Fully air-gapped, no inbound or outbound traffic |
| Query profiling & history | Viewable in Polars Cloud portal       | Managed within your infrastructure               |
| Pricing                   | Generous free tier, pay as you go     | Contact us                                       |

**Self-Serve** is the right choice for teams who want to run clusters on their own infrastructure
but still benefit from centralized visibility. Clusters connect back to the Polars Cloud portal to
sync query history and profiling data, giving your team a shared view of usage without any data
leaving your environment.

**Enterprise** is for teams with strict network isolation requirements. The control plane runs
entirely inside your infrastructure with no inbound or outbound connection to the Polars Cloud
portal.

### Self-Serve

Self-Serve is the fastest way to get started. Get credentials from the Polars Cloud portal, deploy
into your own infrastructure, and manage clusters and query profiling from there. Generous free
tier, pay-as-you-go beyond that, no credit card required.

With Self-Serve, query history is automatically synced to the control plane, so you get a persistent
historical view of query performance that survives cluster restarts and teardowns.

### Enterprise

Enterprise is for teams operating in air-gapped or regulated environments. The entire control plane
runs inside your infrastructure, no outbound traffic required.

[Request an Enterprise license](https://w0lzyfh2w8o.typeform.com/to/zuoDgoMv)
