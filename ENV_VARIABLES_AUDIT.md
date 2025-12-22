# Environment Variables Audit & Configuration Guide

## Overview
All hardcoded values have been replaced with environment variables from `.env` file.
This ensures configuration is externalized and can be easily modified without code changes.

---

## Environment Variables Reference

### Database Configuration

| Variable | Purpose | Default | Usage |
|----------|---------|---------|-------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://goinsight:goinsight_dev_pass@goinsight-postgres:5432/goinsight?sslmode=disable` | `src/config.py` → All database operations |

**Files Using**: `src/db.py` (engine creation), `src/config.py` (config class)

**Local Dev vs Docker**:
```bash
# Local development (.env)
DATABASE_URL=postgresql://goinsight:goinsight_dev_pass@localhost:5432/goinsight?sslmode=disable

# Docker deployment (docker-compose.yml reads from .env)
DATABASE_URL=postgresql://goinsight:goinsight_dev_pass@goinsight-postgres:5432/goinsight?sslmode=disable
```

---

### Model Configuration

| Variable | Purpose | Default | Usage |
|----------|---------|---------|-------|
| `MODELS_DIR` | Directory to save/load trained models | `models` | `src/config.py` → Model save/load operations |
| `RANDOM_SEED` | Random seed for reproducibility | `42` | `src/config.py` → Feature engineering, train/test split |

**Files Using**:
- `src/config.py` - Configuration loading
- `src/models/churn_model.py` - Model path construction
- `src/models/product_area_model.py` - Model path construction
- `src/training/train_churn.py` - np.random.seed()
- `src/training/train_product_area.py` - Random seed setup

---

### Training Hyperparameters

| Variable | Purpose | Default | Usage |
|----------|---------|---------|-------|
| `BATCH_SIZE` | Samples per gradient update | `32` | Churn model training |
| `EPOCHS` | Complete passes through training data | `50` | Churn model training |
| `VALIDATION_SPLIT` | Fraction for validation (1-this = test fraction) | `0.2` | Test set size calculation |

**Files Using**:
- `src/config.py` - Loads from environment
- `src/models/churn_model.py` - `self.config.batch_size`, `self.config.epochs`
- `src/training/train_churn.py` - `config.validation_split` to calculate test_size

**Example Flow**:
```python
# In train_churn.py
config = get_config()
if test_size is None:
    test_size = 1 - config.validation_split  # Uses env var

train_test_split(X, y, test_size=test_size, ...)
```

---

### Scoring Configuration

| Variable | Purpose | Default | Usage |
|----------|---------|---------|-------|
| `SCORE_BATCH_SIZE` | Predictions to generate at once | `1000` | Batch scoring operations |

**Files Using**:
- `src/config.py` - Loads from environment
- `src/scoring/score_accounts.py` - `config.score_batch_size`
- `src/scoring/score_product_areas.py` - Batch processing (if implemented)

---

### Logging Configuration

| Variable | Purpose | Default | Usage |
|----------|---------|---------|-------|
| `LOG_LEVEL` | Logging verbosity | `INFO` | Logging setup across all modules |

**Files Using**:
- `src/config.py` - Loads from environment
- All modules with `logging.basicConfig(level=config.log_level)`

---

## Configuration Flow Diagram

```
┌─────────────────────────────────┐
│    .env file (local dev)        │
│  or docker-compose.yml env      │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│    src/config.py                │
│    Config class loads from      │
│    os.getenv() with defaults    │
└──────────────┬──────────────────┘
               │
               ├──────────────────────────────┐
               │                              │
               ▼                              ▼
     ┌──────────────────────┐    ┌──────────────────────┐
     │ src/db.py            │    │ Model/Training/      │
     │ - get_engine()       │    │ Scoring modules      │
     │                      │    │ - get_config()       │
     │ Uses:                │    │                      │
     │ DATABASE_URL         │    │ Uses:                │
     │ MODELS_DIR           │    │ All env vars         │
     └──────────────────────┘    └──────────────────────┘
```

---

## Verification Checklist

### Database Configuration
- [x] `DATABASE_URL` read from env (src/config.py:18)
- [x] Falls back to Docker default if not set
- [x] Used in: `src/db.py` get_engine()
- [x] docker-compose.yml references ${DATABASE_URL}

### Model Configuration
- [x] `MODELS_DIR` read from env (src/config.py:21)
- [x] Default: `models/`
- [x] Used in: src/models/churn_model.py, product_area_model.py

### Training Configuration
- [x] `BATCH_SIZE` read from env (src/config.py:24)
- [x] `EPOCHS` read from env (src/config.py:25)
- [x] `VALIDATION_SPLIT` read from env (src/config.py:26)
- [x] Used in: src/training/train_churn.py via model.train()
- [x] Model receives from config: epochs, batch_size

### Scoring Configuration
- [x] `SCORE_BATCH_SIZE` read from env (src/config.py:29)
- [x] Used in: src/scoring/score_accounts.py

### Logging Configuration
- [x] `LOG_LEVEL` read from env (src/config.py:32)
- [x] Used in: logging setup across modules

### Docker Integration
- [x] docker-compose.yml reads all vars from .env
- [x] Syntax: `${VAR_NAME:-default_value}`
- [x] All training params passed to container

---

## Command Examples

### Local Development (using .env)
```bash
# Set custom values in .env
BATCH_SIZE=16
EPOCHS=100
VALIDATION_SPLIT=0.15
DATABASE_URL=postgresql://user:pass@localhost:5432/db

# Run training (reads from .env)
python cli.py train churn
```

### Docker Deployment
```bash
# Build with env vars from .env
docker compose build

# All env vars automatically passed to container
docker compose run --rm tens-insight python cli.py train churn
```

### Override Specific Variables
```bash
# Override BATCH_SIZE for just this run
BATCH_SIZE=64 docker compose run --rm tens-insight python cli.py train churn

# Or use .env.production for different environment
cp .env.example .env.production
# Edit .env.production with prod values
# docker compose --env-file .env.production ...
```

---

## Files Modified

### docker-compose.yml
```yaml
# BEFORE (hardcoded)
environment:
  - DATABASE_URL=postgresql://goinsight:...@goinsight-postgres:5432/...
  - MODELS_DIR=/app/models
  - LOG_LEVEL=INFO

# AFTER (uses env vars)
environment:
  - DATABASE_URL=${DATABASE_URL:-postgresql://goinsight:...}
  - MODELS_DIR=${MODELS_DIR:-/app/models}
  - LOG_LEVEL=${LOG_LEVEL:-INFO}
  - BATCH_SIZE=${BATCH_SIZE:-32}
  - EPOCHS=${EPOCHS:-50}
  - VALIDATION_SPLIT=${VALIDATION_SPLIT:-0.2}
  - SCORE_BATCH_SIZE=${SCORE_BATCH_SIZE:-1000}
  - RANDOM_SEED=${RANDOM_SEED:-42}
```

### src/config.py
Already properly implemented! All values loaded via `os.getenv()` with defaults.

### src/training/train_churn.py
```python
# Now uses config for test_size calculation
if test_size is None:
    test_size = 1 - config.validation_split
```

### src/scoring/score_accounts.py
```python
# Now uses config for batch_size
if batch_size is None:
    batch_size = config.score_batch_size
```

### src/models/churn_model.py
Already properly implemented! Model training uses `self.config.batch_size` and `self.config.epochs`.

### .env.example
Enhanced with detailed comments explaining each variable:
```
# BATCH_SIZE: Number of samples per gradient update during training
BATCH_SIZE=32
# EPOCHS: Number of complete passes through the training dataset
EPOCHS=50
# VALIDATION_SPLIT: Fraction of training data to use for validation
VALIDATION_SPLIT=0.2
# SCORE_BATCH_SIZE: Number of predictions to generate at once
SCORE_BATCH_SIZE=1000
```

---

## Complete Environment Variable Usage Map

| Variable | Loaded In | Used In | Level |
|----------|-----------|---------|-------|
| DATABASE_URL | src/config.py | src/db.py | Module |
| MODELS_DIR | src/config.py | src/models/*.py | Class |
| RANDOM_SEED | src/config.py | src/training/train_churn.py | Function |
| BATCH_SIZE | src/config.py | src/models/churn_model.py | Method |
| EPOCHS | src/config.py | src/models/churn_model.py | Method |
| VALIDATION_SPLIT | src/config.py | src/training/train_churn.py | Function |
| SCORE_BATCH_SIZE | src/config.py | src/scoring/score_accounts.py | Function |
| LOG_LEVEL | src/config.py | All modules | Logging |

---

## Best Practices Implemented

**Centralized Configuration**: All env vars loaded in one place (src/config.py)
**Sensible Defaults**: All variables have defaults in Config class
**No Hardcoded Values**: Zero hardcoding in application code
**Docker Integration**: All vars passed through docker-compose.yml
**Documentation**: .env.example fully documented
**Type Safety**: All values properly typed and converted (int(), float(), str)
**Flexibility**: Easy to override per-environment
**Security**: Database credentials via environment variable

---

## Future Enhancements

- [ ] Support for .env.production, .env.staging files
- [ ] Configuration validation on startup
- [ ] Environment-specific defaults (dev vs prod)
- [ ] Configuration hot-reloading
- [ ] Secrets management integration (e.g., HashiCorp Vault)
