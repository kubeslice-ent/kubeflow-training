# T4 → H100/A10 Production Upgrade Guide

Step-by-step guide to upgrade from the T4 demo setup to production-grade GPU training.

## Overview

| Change | What to Modify | Files |
|---|---|---|
| GPU Hardware | Resource requests + config | `pytorchjob.yaml`, config YAML |
| Model Size | Config file swap | `configs/h100_prod.yaml` |
| Data Source | S3 instead of HF Hub | `pytorchjob.yaml` init container args |
| Storage | Shared PVC (NFS/EFS) | `pvc.yaml`, `pytorchjob.yaml` |
| Network | InfiniBand/RDMA | NCCL env vars in `pytorchjob.yaml` |
| Monitoring | Add WandB | Config + secrets |

## Step 1: Switch Config Profile

```bash
# In k8s/pytorchjob.yaml, change the config arg:
# FROM:
#   - "--config"
#   - "configs/t4_demo.yaml"
# TO:
  - "--config"
  - "configs/h100_prod.yaml"
```

Key differences in `h100_prod.yaml`:
- `model_name: meta-llama/Llama-3.1-8B-Instruct` (larger model)
- `bf16: true` (H100/A10 support bfloat16)
- `lora_r: 64` (higher rank for better quality)
- `per_device_train_batch_size: 8` (more VRAM available)
- `max_seq_length: 2048` (longer sequences)
- `fsdp: "full_shard auto_wrap"` (explicit FSDP)

## Step 2: Update GPU Resources

In `k8s/pytorchjob.yaml`, update resource requests per your H100/A10 nodes:

```yaml
resources:
  requests:
    memory: "64Gi"         # H100 nodes have more RAM
    cpu: "8"
    nvidia.com/gpu: "4"    # If 4 GPUs per node
  limits:
    memory: "128Gi"
    cpu: "16"
    nvidia.com/gpu: "4"
```

Scale workers:
```yaml
Worker:
  replicas: 3    # 1 master + 3 workers = 4 nodes
```

## Step 3: Switch to S3 Data Source

Update init container args in `pytorchjob.yaml`:

```yaml
args:
  - "--source"
  - "s3"
  - "--s3-bucket"
  - "your-training-bucket"
  - "--s3-prefix"
  - "datasets/your-dataset/"
  - "--output"
  - "/mnt/data/dataset"
  - "--skip-if-exists"
```

Update `h100_prod.yaml`:
```yaml
dataset_source: "s3"
dataset_path: "/mnt/data/dataset"
```

## Step 4: Switch to Shared PVC

In `k8s/pvc.yaml`, uncomment the shared PVC section and set your storage class:

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: training-data-shared
spec:
  accessModes: [ReadWriteMany]
  storageClassName: nfs  # or efs-sc, fsx-lustre
  resources:
    requests:
      storage: 500Gi
```

In `k8s/pytorchjob.yaml`, update volume references:
```yaml
volumes:
  - name: data-volume
    persistentVolumeClaim:
      claimName: training-data-shared
  - name: checkpoint-volume
    persistentVolumeClaim:
      claimName: training-checkpoints-shared
```

## Step 5: Enable InfiniBand/RDMA

Update NCCL environment variables in `pytorchjob.yaml`:

```yaml
env:
  - name: NCCL_IB_DISABLE
    value: "0"           # Enable InfiniBand
  - name: NCCL_IB_HCA
    value: "mlx5_0"      # Your IB HCA device
  - name: NCCL_NET_GDR_LEVEL
    value: "5"           # GPUDirect RDMA
  # Remove NCCL_SOCKET_IFNAME (not needed with IB)
```

## Step 6: Enable WandB Monitoring

In your config YAML:
```yaml
report_to:
  - tensorboard
  - wandb
wandb_project: "production-llm-training"
```

Ensure WandB API key is in the secret:
```bash
kubectl create secret generic training-secrets \
  --namespace=kubeflow \
  --from-literal=wandb-api-key=YOUR_KEY \
  ...
```

## Step 7: Enable Elastic Training (Optional)

Uncomment in `pytorchjob.yaml`:
```yaml
spec:
  elasticPolicy:
    rdzvBackend: c10d
    minReplicas: 1
    maxReplicas: 8
```

## Scaling Reference

| Setup | GPUs | Model | Est. Training Time (Alpaca 52K) |
|---|---|---|---|
| 4× T4 (demo) | 4 | Llama 3.2 3B | ~2 hours |
| 4× A10 | 4 | Llama 3.1 8B | ~1.5 hours |
| 8× H100 | 8 | Llama 3.1 8B | ~20 minutes |
| 16× H100 | 16 | Llama 3.1 70B | ~3 hours |
