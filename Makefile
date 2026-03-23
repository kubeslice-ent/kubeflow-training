# ============================================================================
# Local validation — run before docker build
# ============================================================================

.PHONY: setup lint typecheck test helm-lint validate clean

# Install dev dependencies (one-time)
setup:
	pip install -r requirements-dev.txt

# Static analysis — catches undefined attributes, unused imports, syntax errors
lint:
	ruff check train.py data/ tests/

# Type checking — catches wrong attribute names, bad function signatures
typecheck:
	pyright train.py data/

# Unit/smoke tests — catches API mismatches, config key errors
test:
	pytest tests/ -v

# Helm chart validation
helm-lint:
	helm lint helm/kubeflow-llm-training/
	helm template test helm/kubeflow-llm-training/ --namespace kubeflow > /dev/null

# ============================================================================
# Run ALL checks (use this before docker build)
# ============================================================================
validate: lint typecheck test helm-lint
	@echo ""
	@echo "All checks passed — safe to build and push."

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
