# 🎯 Customer Demo Script — Distributed LLM Fine-Tuning

**Duration:** 20–30 minutes  
**Audience:** Technical decision-makers, ML engineers, DevOps  
**Goal:** Show production-ready distributed LLM fine-tuning on Kubernetes with Kubeflow, scalable from T4 → H100 via Avesha EGS

---

## Pre-Demo Checklist

- [ ] Docker image built and pushed to registry
- [ ] Secrets created (`kubectl get secret training-secrets -n kubeflow`)
- [ ] PVCs created (`kubectl get pvc -n kubeflow`)
- [ ] TensorBoard deployed (`kubectl get pods -n kubeflow -l app=tensorboard`)
- [ ] Port forward ready: `kubectl port-forward svc/tensorboard 6006:6006 -n kubeflow`
- [ ] Terminal windows open: one for `kubectl`, one for logs

---

## Demo Flow

### Act 1: The Problem (2 min)

> **Talking Points:**
> - "Fine-tuning LLMs requires distributed GPU infrastructure — this is hard to manage at scale"
> - "You need: multi-node GPU orchestration, data pipelines, fault tolerance, monitoring"
> - "Kubeflow + Kubernetes solves this — and with Avesha EGS, training spans across clusters"

### Act 2: Architecture Walkthrough (3 min)

Show the architecture diagram from README.md.

> **Key points to highlight:**
> 1. **PyTorchJob** — Kubeflow manages the distributed training lifecycle
> 2. **FSDP** — model/optimizer states sharded across GPUs for memory efficiency
> 3. **QLoRA** — 4-bit quantization lets us train on cost-effective T4 GPUs
> 4. **Init containers** — automatic dataset download from S3 or HuggingFace
> 5. **Zero-change upgrade** — same code works on H100s, just swap the config

### Act 3: Configuration (3 min)

Show the two config files side by side:

```bash
# Show T4 demo config
cat configs/t4_demo.yaml

# Show H100 production config
cat configs/h100_prod.yaml
```

> **Highlight:**
> - "The training script is identical — only the config changes"
> - "Batch size, precision, model size, LoRA rank — all configurable"
> - "On H100s: bf16 precision, larger models (8B-70B), higher LoRA rank"

### Act 4: Submit Training Job (5 min)

```bash
# Show the PyTorchJob manifest
cat k8s/pytorchjob.yaml

# Submit the job
kubectl apply -f k8s/pytorchjob.yaml

# Watch job creation
kubectl get pytorchjobs -n kubeflow -w
```

> **Highlight:**
> - "One `kubectl apply` — Kubeflow handles the rest"
> - "Master + Worker pods created automatically"
> - "Each node gets 2 GPUs, `torchrun` launches processes per GPU"

### Act 5: Monitor Training (5 min)

```bash
# Watch the master pod logs
kubectl logs -f llm-finetune-master-0 -n kubeflow -c pytorch
```

**Key log lines to point out:**

```
DISTRIBUTED TRAINING INFO
  WORLD_SIZE: 4          ← "All 4 GPUs across 2 nodes are training together"

Model loaded. Trainable: 13,631,488 / 3,213,631,488 (0.42%)
                          ← "Only 0.42% of params are trained — that's QLoRA"

NCCL INFO: Using network Socket
                          ← "NCCL handling GPU-to-GPU communication"

{'loss': 1.82, 'learning_rate': 0.0002, 'epoch': 0.01}
{'loss': 1.56, 'learning_rate': 0.00019, 'epoch': 0.02}
                          ← "Loss is decreasing — model is learning"
```

**Show TensorBoard:**
```bash
# Open TensorBoard (port-forward should be running)
# Navigate to: http://localhost:6006
```

> **Show:** Loss curve, learning rate schedule, GPU utilization

### Act 6: Production Upgrade Story (3 min)

> **Talking Points:**
> - "What you just saw runs on T4 GPUs — $0.35/hr each"
> - "To move to production H100s, you change 3 things:"
>   1. Swap config: `t4_demo.yaml` → `h100_prod.yaml`
>   2. Update GPU count in `pytorchjob.yaml`
>   3. Point dataset to S3 bucket
> - "**Zero code changes in the training script**"
> - "With Avesha EGS, this same job can be distributed across clusters in different regions"

```bash
# Show the production config
diff configs/t4_demo.yaml configs/h100_prod.yaml
```

### Act 7: Avesha EGS Integration (3 min)

> **Talking Points:**
> - "Avesha EGS enables distributed training ACROSS clusters"
> - "GPU resources in Cluster A + Cluster B appear as a single pool"
> - "NCCL traffic flows through EGS mesh — transparent to the training code"
> - "This is how you scale beyond a single cluster's GPU capacity"

### Act 8: Q&A (5 min)

**Common questions & answers:**

| Question | Answer |
|---|---|
| "How do we use our own data?" | "Replace the dataset in the config — S3, HF Hub, or local. See `docs/CUSTOM_DATASET.md`" |
| "What about 70B models?" | "Same setup, more GPUs. FSDP shards the model across nodes" |
| "What about fault tolerance?" | "PyTorchJob has `restartPolicy: OnFailure`. Add elastic policy for production" |
| "Can we use DeepSpeed?" | "Yes — switch from FSDP to DeepSpeed ZeRO in the config. Same infrastructure" |
| "How do we version models?" | "Built-in — each run saves timestamped checkpoints with full metadata" |

---

## Cleanup

```bash
# Delete the training job
kubectl delete pytorchjob llm-finetune -n kubeflow

# Optional: delete PVCs (preserves checkpoints if you skip this)
# kubectl delete -f k8s/pvc.yaml
```

---

## Demo Variations

### Quick Demo (10 min)
Skip Acts 2 and 7. Use `--dry_run` flag for fast completion:
```bash
# In pytorchjob.yaml, add to args:
# - "--dry_run"
```

### Deep Technical Demo (45 min)
Add:
- Live code walkthrough of `train.py`
- Show FSDP sharding in action (GPU memory per rank)
- Run inference on the fine-tuned model
- Show dataset preparation workflow
