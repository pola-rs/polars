#### Temporary data

Polars itself uses some temporary storage location in the streaming engine and in some cases when
downloading remote files. For most queries this is a relatively small volume and is not
performance-sensitive. By default, the persistent volume for this is disabled, and an `emptyDir`
volume is used instead. However, to prevent the host from running out of disk space during large
queries, it is recommended to enable a persistent volume for this purpose. The feature below will
add a [Generic Ephemeral Volume](https://kubernetes.io/docs/concepts/storage/ephemeral-volumes/) to
each of the pods.

```yaml
temporaryData:
  ephemeralVolumeClaim:
    enabled: true
    storageClassName: "hostpath" # As configured in your k8s cluster
    size: 125Gi
```
