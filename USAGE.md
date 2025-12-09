# Tens-Insight Usage Guide

Complete guide for using the Tens-Insight ML pipeline.

## Quick Start

### 1. Install Dependencies

```bash
cd tens-insight
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env if needed (default values work with goinsight)
```

### 3. Run Setup

```bash
python setup.py
```

This will:
- Create the `models/` directory
- Test database connection
- Create prediction tables (`account_risk_scores`, `product_area_impact`)

### 4. Train Models

```bash
# Train churn prediction model
python -m src.training.train_churn

# Train product area impact model
python -m src.training.train_product_area
```

### 5. Run Scoring

```bash
# Score all accounts
python -m src.scoring.score_accounts

# Score product areas
python -m src.scoring.score_product_areas
```

## Docker Usage

### Build and Run with Docker Compose

```bash
# Build the image
docker compose build

# Run setup
docker compose run tens-insight python setup.py

# Train models
docker compose run tens-insight python -m src.training.train_churn
docker compose run tens-insight python -m src.training.train_product_area

# Run scoring
docker compose run tens-insight python -m src.scoring.score_accounts
docker compose run tens-insight python -m src.scoring.score_product_areas
```

### Integration with GoInsight

If you're running this alongside goinsight:

1. Use the same docker network:
   ```yaml
   networks:
     - goinsight_default
   ```

2. Point to goinsight's postgres service:
   ```bash
   DATABASE_URL=postgres://goinsight:goinsight_dev_pass@postgres:5432/goinsight?sslmode=disable
   ```

3. Make sure goinsight is running first so the database exists.

## Automation & Scheduling

### Using Cron

Add entries to your crontab to run scoring periodically:

```bash
# Retrain models weekly (Sunday at 2 AM)
0 2 * * 0 cd /path/to/tens-insight && python -m src.training.train_churn >> logs/train.log 2>&1

# Score accounts daily (every day at 3 AM)
0 3 * * * cd /path/to/tens-insight && python -m src.scoring.score_accounts >> logs/score.log 2>&1

# Score product areas daily (every day at 3:30 AM)
30 3 * * * cd /path/to/tens-insight && python -m src.scoring.score_product_areas >> logs/score.log 2>&1
```

### Using CI/CD

Example GitHub Actions workflow:

```yaml
name: ML Pipeline

on:
  schedule:
    - cron: '0 3 * * *'  # Daily at 3 AM UTC
  workflow_dispatch:      # Manual trigger

jobs:
  score:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Score accounts
        env:
          DATABASE_URL: ${{ secrets.DATABASE_URL }}
        run: python -m src.scoring.score_accounts
      - name: Score product areas
        env:
          DATABASE_URL: ${{ secrets.DATABASE_URL }}
        run: python -m src.scoring.score_product_areas
```

## Advanced Usage

### Custom Lookback Windows

```bash
# Use 30-day lookback for features
python -m src.training.train_churn --lookback-days 30
python -m src.scoring.score_accounts --lookback-days 30
```

### Model Versioning

```bash
# Train a new model version
python -m src.training.train_churn --model-version v2

# Score with specific version
python -m src.scoring.score_accounts --model-version v2
```

### Testing Database Connection

```python
from src.db import execute_query

# Test connection
df = execute_query("SELECT COUNT(*) FROM feedback_enriched")
print(f"Feedback records: {df.iloc[0, 0]}")
```

### Inspecting Predictions

```python
from src.db import execute_query

# View recent account predictions
df = execute_query("""
    SELECT account_id, churn_probability, health_score, risk_category
    FROM account_risk_scores
    ORDER BY predicted_at DESC
    LIMIT 10
""")
print(df)

# View top priority product areas
df = execute_query("""
    SELECT product_area, segment, priority_score
    FROM product_area_impact
    WHERE priority_score >= 50
    ORDER BY priority_score DESC
""")
print(df)
```

## Monitoring & Logging

### View Logs

```bash
# Training logs
tail -f logs/train.log

# Scoring logs
tail -f logs/score.log
```

### Check Prediction Tables

```sql
-- Count predictions by risk category
SELECT risk_category, COUNT(*)
FROM account_risk_scores
GROUP BY risk_category;

-- View highest priority product areas
SELECT product_area, segment, priority_score, feedback_count
FROM product_area_impact
ORDER BY priority_score DESC
LIMIT 10;
```

## Troubleshooting

### Database Connection Issues

```bash
# Test connection
psql "$DATABASE_URL"

# Verify tables exist
psql "$DATABASE_URL" -c "\dt"
```

### Model Not Found

```bash
# List saved models
ls -lh models/

# Retrain if needed
python -m src.training.train_churn
```

### No Data to Score

```bash
# Check feedback table
psql "$DATABASE_URL" -c "SELECT COUNT(*) FROM feedback_enriched"

# If empty, run goinsight seed data
cd ../goinsight
go run cmd/seed/main.go
```

## Best Practices

1. **Always run setup first** after cloning or on a new environment
2. **Train models before scoring** on first run
3. **Monitor prediction quality** by tracking churn/retention metrics
4. **Retrain regularly** (weekly or monthly) as new data arrives
5. **Version your models** when making significant changes
6. **Back up model artifacts** in the `models/` directory
7. **Log everything** to track performance over time

## Next Steps

- Add custom features based on your domain knowledge
- Implement A/B testing for model versions
- Add monitoring dashboards (e.g., Grafana)
- Integrate with alerting systems for high-risk accounts
- Build a model retraining pipeline with validation checks
