# Using Custom / Domain-Specific Datasets

Guide for replacing the default Alpaca dataset with your own data.

## Supported Formats

| Format | Best For | Key Fields |
|---|---|---|
| **Alpaca** | Instruction tuning | `instruction`, `input`, `output` |
| **ShareGPT** | Multi-turn conversations | `conversations[{from, value}]` |
| **ChatML** | Chat-style fine-tuning | `system`, `user`, `assistant` |
| **Llama 3** | Llama-native chat | `system`, `user`, `assistant` |
| **Custom** | Any format | Configurable column names |

## Option 1: HuggingFace Hub Dataset

1. Find a dataset on [HuggingFace Hub](https://huggingface.co/datasets)
2. Update config:
   ```yaml
   dataset_source: "huggingface"
   dataset_name: "your-org/your-dataset"
   ```
3. If the dataset requires authentication, ensure `HF_TOKEN` is set in secrets

## Option 2: S3 Dataset

### Prepare Your Data

Create a JSONL file with your data:

```json
{"instruction": "Summarize the following text", "input": "Long text here...", "output": "Summary here..."}
{"instruction": "Translate to French", "input": "Hello world", "output": "Bonjour le monde"}
```

### Upload to S3

```bash
# Upload dataset
aws s3 cp my_dataset.jsonl s3://your-bucket/datasets/my-dataset/

# Or for HF Arrow format (recommended for large datasets)
python data/prepare_dataset.py \
  --input my_dataset.jsonl \
  --output /tmp/prepared_dataset \
  --format alpaca

aws s3 sync /tmp/prepared_dataset s3://your-bucket/datasets/my-dataset/
```

### Configure Training

```yaml
dataset_source: "s3"
dataset_path: "/mnt/data/dataset"
```

Update init container in `pytorchjob.yaml`:
```yaml
args:
  - "--source"
  - "s3"
  - "--s3-bucket"
  - "your-bucket"
  - "--s3-prefix"
  - "datasets/my-dataset/"
  - "--output"
  - "/mnt/data/dataset"
```

## Option 3: Custom Format

If your data doesn't match standard formats:

```bash
# Convert custom format to training format
python data/prepare_dataset.py \
  --input /path/to/your/data.jsonl \
  --output /mnt/data/prepared \
  --format custom \
  --instruction-col "question" \
  --response-col "answer" \
  --input-col "context" \
  --system-message "You are a domain expert."
```

## Dataset Versioning

Use the `--revision` flag for HuggingFace datasets:
```yaml
dataset_name: "your-org/your-dataset"
# In download_dataset.py args:
# --revision "v2.0"
```

For S3, use prefixed paths:
```
s3://bucket/datasets/my-dataset/v1/
s3://bucket/datasets/my-dataset/v2/
```

## Domain-Specific Examples

### Medical
```yaml
dataset_name: "medmcqa"  # or your private medical dataset
```
System message: `"You are a medical AI assistant. Provide accurate, evidence-based medical information."`

### Legal
```yaml
dataset_name: "nguha/legalbench"
```
System message: `"You are a legal AI assistant specialized in contract analysis."`

### Code
```yaml
dataset_name: "bigcode/starcoderdata"
```

### Customer Support
Create from your support ticket data:
```json
{"instruction": "Help the customer with their issue", "input": "My order hasn't arrived", "output": "I'm sorry to hear that..."}
```

## Tips

1. **Data quality > quantity** — 1K high-quality samples beats 100K noisy ones
2. **Consistent formatting** — use the same template throughout
3. **Test with small subset** — use `--max-samples 100` for quick validation
4. **Monitor for overfitting** — watch validation loss if using a split
