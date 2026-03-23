# ============================================================================
# Multi-stage Dockerfile for Distributed LLM Fine-Tuning
# Compatible: NVIDIA T4 (Turing), A10 (Ampere), H100 (Hopper)
# ============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Base with CUDA + PyTorch
# ---------------------------------------------------------------------------
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel AS base

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Stage 2: Install Python dependencies
# ---------------------------------------------------------------------------
FROM base AS deps

WORKDIR /app

# Install Python dependencies (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Verify critical imports
RUN python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')" && \
    python -c "import transformers; print(f'Transformers: {transformers.__version__}')" && \
    python -c "import peft; print(f'PEFT: {peft.__version__}')" && \
    python -c "import trl; print(f'TRL: {trl.__version__}')" && \
    python -c "import bitsandbytes; print(f'BitsAndBytes: {bitsandbytes.__version__}')"

# ---------------------------------------------------------------------------
# Stage 3: Production image
# ---------------------------------------------------------------------------
FROM deps AS production

WORKDIR /app

# Copy application code
COPY train.py .
COPY data/ ./data/
COPY configs/ ./configs/

# Create directories for data and checkpoints
RUN mkdir -p /mnt/data /mnt/checkpoints

# NCCL optimizations for multi-node training
# These can be overridden via Pod environment variables
ENV NCCL_DEBUG=INFO
ENV NCCL_IB_DISABLE=1
ENV NCCL_SOCKET_IFNAME=eth0
ENV NCCL_P2P_DISABLE=0
# Bump network timeout for cross-node communication
ENV NCCL_NET_GDR_LEVEL=0
ENV TORCH_NCCL_BLOCKING_WAIT=1
# CRITICAL: Without this, NCCL errors cause silent hangs instead of exceptions
ENV TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# HuggingFace cache dir (will be on PVC in k8s)
ENV HF_HOME=/mnt/data/.cache/huggingface
ENV TRANSFORMERS_CACHE=/mnt/data/.cache/huggingface/transformers

# NOTE: Using CMD (not ENTRYPOINT) because PyTorchJob's `command` field
# overrides ENTRYPOINT. CMD is used for standalone container execution only.
# In K8s, the PyTorchJob spec defines command + args directly.
CMD ["python", "train.py", "--config", "configs/t4_demo.yaml"]
