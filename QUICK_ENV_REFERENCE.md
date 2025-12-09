# Quick Reference: Environment Variables

## All Configuration is Now Externalized ✅

**8 Environment Variables** - No hardcoded values in application code

---

## Quick Reference Card

### Database (Location: .env)
```
DATABASE_URL=postgresql://goinsight:goinsight_dev_pass@localhost:5432/goinsight?sslmode=disable
```
- **Local Dev**: Use `localhost:5432`
- **Docker**: Use `goinsight-postgres:5432`

### Model Storage (Location: .env)
```
MODELS_DIR=models
```
- Directory for saving trained models
- In Docker: `/app/models`

### Training Configuration (Location: .env)
```
BATCH_SIZE=32              # Samples per update (try: 16, 32, 64)
EPOCHS=50                  # Training passes (try: 10, 50, 100)
VALIDATION_SPLIT=0.2       # Validation fraction (test_size = 1 - this)
RANDOM_SEED=42             # Reproducibility (any integer)
```

### Scoring Configuration (Location: .env)
```
SCORE_BATCH_SIZE=1000      # Predictions at once (try: 100, 1000, 10000)
```

### Logging (Location: .env)
```
LOG_LEVEL=INFO             # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

---

## Usage Examples

### Change Batch Size
```bash
# Option 1: Edit .env
BATCH_SIZE=64

# Option 2: Command line override
BATCH_SIZE=64 python cli.py train churn

# Option 3: Docker override
docker compose run --rm -e BATCH_SIZE=64 tens-insight python cli.py train churn
```

### Change Training Epochs
```bash
# In .env
EPOCHS=100

# Or command line
EPOCHS=100 python cli.py train churn
```

### Change Database URL
```bash
# Local development .env
DATABASE_URL=postgresql://user:pass@localhost:5432/db

# Docker deployment .env
DATABASE_URL=postgresql://user:pass@goinsight-postgres:5432/db
```

### Multiple Overrides
```bash
BATCH_SIZE=64 EPOCHS=100 VALIDATION_SPLIT=0.25 python cli.py train churn
```

---

## Configuration Locations

| Component | Configuration Source | File |
|-----------|---------------------|------|
| Env Variables | .env template | .env.example |
| Env Variables | Local/Docker config | .env (git ignored) |
| Config Class | Central loading point | src/config.py |
| Usage Reference | Complete guide | ENV_VARIABLES_AUDIT.md |

---

## Docker Deployment

```bash
# Build reads from .env automatically
docker compose build

# Run reads from .env automatically
docker compose run --rm tens-insight python cli.py train churn

# Or use different .env file
docker compose --env-file .env.production build
docker compose --env-file .env.production run --rm tens-insight python cli.py train churn
```

---

## Verification

```bash
# Check env vars loaded correctly
python -c "from src.config import get_config; c = get_config(); print(f'BATCH_SIZE={c.batch_size}, EPOCHS={c.epochs}')"

# Check docker-compose uses env vars
grep "\${" docker-compose.yml

# Check all vars in .env
cat .env | grep "^[A-Z]"
```

---

## Environment-Specific Setups

### Development (.env)
```
BATCH_SIZE=32
EPOCHS=50
VALIDATION_SPLIT=0.2
DATABASE_URL=postgresql://goinsight:goinsight_dev_pass@localhost:5432/goinsight
LOG_LEVEL=INFO
```

### Production (.env.production)
```
BATCH_SIZE=64
EPOCHS=200
VALIDATION_SPLIT=0.15
DATABASE_URL=postgresql://user:secure_pass@prod-db.example.com:5432/prod_db
LOG_LEVEL=WARNING
```

### Staging (.env.staging)
```
BATCH_SIZE=48
EPOCHS=100
VALIDATION_SPLIT=0.2
DATABASE_URL=postgresql://user:staging_pass@staging-db.example.com:5432/staging_db
LOG_LEVEL=INFO
```

---

## No Hardcoded Values ✅

**Verified:**
- ✅ docker-compose.yml: All values use `${VAR}` syntax
- ✅ src/config.py: All values loaded via `os.getenv()`
- ✅ src/models/churn_model.py: Uses `self.config.*`
- ✅ src/training/train_churn.py: Uses `config.*` values
- ✅ src/scoring/score_accounts.py: Uses `config.*` values
- ✅ .env.example: Fully documented

---

## Common Tasks

### Tune Hyperparameters
```bash
# Edit .env
BATCH_SIZE=16
EPOCHS=100

# Run training
python cli.py train churn
```

### Debug Logging
```bash
# Increase verbosity
LOG_LEVEL=DEBUG python cli.py train churn
```

### Change Database
```bash
# Edit .env
DATABASE_URL=postgresql://new_user:new_pass@new_host:5432/new_db

# Run setup
python setup.py
```

### Scale Predictions
```bash
# Increase batch size for faster scoring
SCORE_BATCH_SIZE=10000 python cli.py score all
```

---

## Defaults (Used if env var not set)

| Variable | Default | Min | Max |
|----------|---------|-----|-----|
| DATABASE_URL | postgresql://...goinsight-postgres... | - | - |
| MODELS_DIR | models | - | - |
| RANDOM_SEED | 42 | 1 | unlimited |
| BATCH_SIZE | 32 | 1 | 1000 |
| EPOCHS | 50 | 1 | 1000 |
| VALIDATION_SPLIT | 0.2 | 0.1 | 0.5 |
| SCORE_BATCH_SIZE | 1000 | 1 | 100000 |
| LOG_LEVEL | INFO | - | - |

---

## Troubleshooting

**Q: Changes to .env not taking effect?**
A: Restart the application or container

**Q: Can't find env var?**
A: Check .env file exists and variable name matches (case-sensitive)

**Q: Docker not reading .env?**
A: Ensure docker-compose.yml uses `${VAR}` syntax (not hardcoded values)

**Q: Want to override just one variable?**
A: Use `VAR=value command` before running (e.g., `BATCH_SIZE=64 python cli.py train`)

---

## Key Points

✅ All configuration externalized to environment variables
✅ No hardcoded values in application code
✅ Works with local dev, Docker, staging, production
✅ Sensible defaults for all variables
✅ Easy to override per-environment
✅ Fully documented and audited

---

For detailed information, see: `ENV_VARIABLES_AUDIT.md`
