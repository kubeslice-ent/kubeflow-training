# 🚀 Distributed LLM Fine-Tuning on Kubeflow

Production-ready distributed LLM fine-tuning using **Kubeflow Training Operator**, **PyTorch FSDP**, and **QLoRA**. Designed to demo on T4 GPUs and scale to H100/A10 with **zero code changes**.

## Architecture

```
┌─ Kubeflow Training Operator ─────────────────────────────────┐
│                                                               │
│  PyTorchJob: torchrun --nproc_per_node=2 --nnodes=2          │
│                                                               │
│  ┌──────────────────┐  NCCL  ┌──────────────────┐           │
│  │ Node 1 (2× GPU)  │◄─────►│ Node 2 (2× GPU)  │           │
│  │ Rank 0, Rank 1   │       │ Rank 2, Rank 3   │           │
│  └────────┬─────────┘       └────────┬─────────┘           │
│           │                          │                       │
│  ┌────────▼──────────────────────────▼─────────┐            │
│  │ HuggingFace SFTTrainer + PEFT (QLoRA)       │            │
│  │ + PyTorch FSDP (sharded model/optimizer)    │            │
│  └─────────────────────────────────────────────┘            │
│                                                               │
│  Data: S3 / HF Hub / MinIO ──► Init Container ──► PVC       │
│  Monitoring: TensorBoard / WandB                             │
└───────────────────────────────────────────────────────────────┘
```

## Features

| Feature | Description |
|---|---|
| **Distributed Training** | PyTorch FSDP across multiple nodes/GPUs via `torchrun` |
| **QLoRA** | 4-bit quantized LoRA for memory-efficient fine-tuning |
| **Configurable Data** | S3, HuggingFace Hub, MinIO, or local datasets |
| **Hardware Profiles** | Swap between T4/H100/A10 with config file changes |
| **Model Versioning** | Timestamped checkpoints with training metadata |
| **Production Ready** | Secrets management, PVCs, NCCL tuning, fault tolerance |
| **Monitoring** | TensorBoard + optional WandB integration |

## Quick Start

### Prerequisites

- Kubernetes cluster with GPU nodes
- [Kubeflow Training Operator](https://www.kubeflow.org/docs/components/training/) v1.7+
- NVIDIA [GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/) / Device Plugin
- `kubectl` configured to access your cluster
- Docker (for building images)

### 1. Build & Push Docker Image

```bash
# Build
docker build -t YOUR_REGISTRY/kubeflow-llm-training:latest .

# Push to your private registry
docker push YOUR_REGISTRY/kubeflow-llm-training:latest
```

### 2. Create Secrets

```bash
kubectl create secret generic training-secrets \
  --namespace=kubeflow \
  --from-literal=hf-token=hf_YOUR_TOKEN \
  --from-literal=aws-access-key=YOUR_KEY \
  --from-literal=aws-secret-key=YOUR_SECRET \
  --from-literal=wandb-api-key=YOUR_WANDB_KEY \
  --from-literal=s3-endpoint-url=https://s3.amazonaws.com
```

### 3. Update Image References

Replace `YOUR_REGISTRY/kubeflow-llm-training:latest` in `k8s/pytorchjob.yaml` with your actual registry path.

### 4. Deploy

```bash
# Create PVCs
kubectl apply -f k8s/pvc.yaml

# Create ConfigMap
kubectl apply -f k8s/configmap.yaml

# Submit training job
kubectl apply -f k8s/pytorchjob.yaml

# Deploy TensorBoard
kubectl apply -f k8s/monitoring/tensorboard.yaml
```

### 5. Monitor

```bash
# Check job status
kubectl get pytorchjobs -n kubeflow

# Watch training logs (master)
kubectl logs -f llm-finetune-master-0 -n kubeflow -c pytorch

# Watch worker logs
kubectl logs -f llm-finetune-worker-0 -n kubeflow -c pytorch

# Access TensorBoard
kubectl port-forward svc/tensorboard 6006:6006 -n kubeflow
# Open: http://localhost:6006
```

### 6. Verify Training

Look for these in the logs:
```
DISTRIBUTED TRAINING INFO
  RANK:       0
  LOCAL_RANK: 0
  WORLD_SIZE: 4          ◄── All 4 GPUs detected
  CUDA devices: 2        ◄── 2 GPUs per node

Model loaded. Trainable: 13,631,488 / 3,213,631,488 (0.42%)  ◄── QLoRA params

Starting training...
{'loss': 1.8234, 'learning_rate': 0.0002, 'epoch': 0.01}     ◄── Loss decreasing
{'loss': 1.5621, 'learning_rate': 0.00019, 'epoch': 0.02}
```

## Helm-Based Installation

The Helm chart in `helm/kubeflow-llm-training/` provides a single-command install with a configurable `values.yaml`.

### Prerequisites

- [Helm](https://helm.sh/docs/intro/install/) v3.10+
- Same cluster requirements as above (Kubeflow Training Operator, NVIDIA GPU Operator)

### Install (T4 Demo — defaults)

```bash
helm install training helm/kubeflow-llm-training/ \
  --namespace kubeflow --create-namespace \
  --set image.repository=myregistry.io/kubeflow-llm-training \
  --set secrets.hfToken=hf_YOUR_TOKEN
```

This deploys all resources (ConfigMap, Secrets, PVCs, PyTorchJob, TensorBoard) with T4 demo defaults: 2 nodes × 2 GPUs, per-node storage, 4-bit QLoRA.

### Install (Production — H100 / shared storage)

```bash
helm install training helm/kubeflow-llm-training/ \
  --namespace kubeflow --create-namespace \
  -f helm/kubeflow-llm-training/values-production.yaml \
  --set image.repository=myregistry.io/kubeflow-llm-training \
  --set secrets.hfToken=hf_YOUR_TOKEN \
  --set secrets.awsAccessKey=YOUR_KEY \
  --set secrets.awsSecretKey=YOUR_SECRET
```

Production overrides include: shared NFS storage, 8 GPUs/node, 3 worker replicas, elastic policy, bf16 precision, InfiniBand NCCL.

### Use an Existing Secret

If you manage secrets externally (e.g., Vault, Sealed Secrets), skip secret creation:

```bash
helm install training helm/kubeflow-llm-training/ \
  --namespace kubeflow \
  --set secrets.existingSecret=my-training-secrets \
  --set image.repository=myregistry.io/kubeflow-llm-training
```

### Common Overrides

```bash
# Change model and dataset
helm install training helm/kubeflow-llm-training/ \
  --namespace kubeflow \
  --set training.modelName=mistralai/Mistral-7B-v0.1 \
  --set training.datasetName=your-org/your-dataset \
  --set training.configFile=configs/h100_prod.yaml

# Scale workers
helm install training helm/kubeflow-llm-training/ \
  --namespace kubeflow \
  --set pytorchjob.worker.replicas=3 \
  --set pytorchjob.worker.gpusPerNode=4

# Disable TensorBoard
helm install training helm/kubeflow-llm-training/ \
  --namespace kubeflow \
  --set monitoring.tensorboard.enabled=false

# Use shared storage with a custom storage class
helm install training helm/kubeflow-llm-training/ \
  --namespace kubeflow \
  --set storage.mode=shared \
  --set storage.storageClass=efs-sc \
  --set storage.shared.dataSize=500Gi
```

### Upgrade & Uninstall

```bash
# Upgrade with new values
helm upgrade training helm/kubeflow-llm-training/ \
  --namespace kubeflow \
  --set training.numEpochs=3

# Dry-run to preview changes
helm template training helm/kubeflow-llm-training/ \
  --namespace kubeflow

# Uninstall
helm uninstall training --namespace kubeflow
```

### Key values.yaml Sections

| Section | What It Controls |
|---|---|
| `image.*` | Container registry, tag, pull policy |
| `training.*` | Model, dataset, LoRA, hyperparameters, config file |
| `nccl.*` | NCCL debug level, InfiniBand, network interface |
| `secrets.*` | HF token, AWS creds, WandB key, or `existingSecret` |
| `storage.*` | Mode (`per-node`/`shared`), sizes, storage class |
| `pytorchjob.*` | GPU count, replicas, resources, tolerations, elastic policy |
| `monitoring.tensorboard.*` | Enable/disable, image, service type/port |

See [`helm/kubeflow-llm-training/values.yaml`](helm/kubeflow-llm-training/values.yaml) for the full reference.

## Project Structure

```
├── train.py                    # Core training script
├── configs/
│   ├── t4_demo.yaml            # T4 GPU profile (4-bit QLoRA, fp16)
│   ├── h100_prod.yaml          # H100/A10 profile (bf16, higher rank LoRA)
│   └── fsdp_config.yaml        # Accelerate FSDP config (local testing)
├── data/
│   ├── download_dataset.py     # Dataset downloader (S3/HF Hub/local)
│   └── prepare_dataset.py      # Format converter (Alpaca/ChatML/custom)
├── Dockerfile                  # Multi-stage, CUDA 12.4 + HF stack
├── requirements.txt
├── k8s/                        # Raw manifests (kubectl apply)
│   ├── pytorchjob.yaml         # 2 nodes × 2 GPUs training job
│   ├── pvc.yaml                # Storage (per-node / shared modes)
│   ├── secrets.yaml            # Credentials template
│   ├── configmap.yaml          # Training parameters
│   └── monitoring/
│       └── tensorboard.yaml    # TensorBoard deployment
├── helm/kubeflow-llm-training/ # Helm chart (helm install)
│   ├── Chart.yaml              # Chart metadata
│   ├── values.yaml             # Default values (T4 demo)
│   ├── values-production.yaml  # Production overrides (H100)
│   └── templates/              # Templatized K8s manifests
├── DEMO_SCRIPT.md              # Customer demo walkthrough
└── docs/
    ├── PRODUCTION_UPGRADE.md   # T4 → H100 migration guide
    └── CUSTOM_DATASET.md       # Using your own data
```

## Configuration

### Config Priority (highest → lowest)

1. **CLI arguments** (`--max_steps 5`)
2. **Environment variables** (`TRAIN_BATCH_SIZE=4`)
3. **YAML config file** (`--config configs/t4_demo.yaml`)
4. **Default values** (in `train.py`)

### Hardware Profiles

| Parameter | T4 Demo | H100/A10 Production |
|---|---|---|
| Model | Llama 3.2 3B | Llama 3.1 8B–70B |
| Quantization | 4-bit QLoRA | Optional / none |
| LoRA Rank | 16 | 64 |
| Precision | fp16 | bf16 |
| Batch Size | 2 | 8–16 |
| Seq Length | 512 | 2048 |
| FSDP | Auto | full_shard auto_wrap |

### Storage Modes

| Mode | When to Use | How |
|---|---|---|
| **Per-node PVC** | T4 demo, no shared storage | Default in `pvc.yaml` / `values.yaml` |
| **Shared PVC** | Production with NFS/EFS | Uncomment in `pvc.yaml` or `helm install --set storage.mode=shared` |

### Network Configuration

| Setting | T4 Demo (TCP) | Production (InfiniBand) |
|---|---|---|
| `NCCL_IB_DISABLE` | `1` | `0` |
| `NCCL_SOCKET_IFNAME` | `eth0` | (not needed) |
| `NCCL_IB_HCA` | (not set) | `mlx5_0` |
| `NCCL_NET_GDR_LEVEL` | `0` | `5` |

## Upgrading to Production

See [docs/PRODUCTION_UPGRADE.md](docs/PRODUCTION_UPGRADE.md) for detailed instructions on:
- Switching from T4 → H100/A10 GPUs
- Moving from per-node PVC → shared NFS/EFS
- Enabling InfiniBand/RDMA for NCCL
- Using S3 datasets instead of HuggingFace Hub
- Scaling to more nodes

## Using Custom Datasets

See [docs/CUSTOM_DATASET.md](docs/CUSTOM_DATASET.md) for:
- Preparing domain-specific datasets
- Supported formats (Alpaca, ShareGPT, ChatML, Llama 3, custom)
- S3 upload instructions
- Dataset versioning

## Troubleshooting

| Issue | Solution |
|---|---|
| OOM on T4 | Reduce `per_device_train_batch_size` to 1 or `max_seq_length` to 256 |
| NCCL timeout | Check `NCCL_SOCKET_IFNAME` matches your pod network interface |
| Slow init container | Dataset is large; use `--skip-if-exists` to skip on restart |
| Loss not decreasing | Verify dataset format; check `dataloader` is not empty |
| Pods pending | Check GPU availability: `kubectl describe nodes \| grep nvidia` |

## License

MIT
