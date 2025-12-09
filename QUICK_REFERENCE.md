# Tens-Insight Quick Reference

## Common Commands

### Setup & Status
```bash
python setup.py              # Initial setup
python cli.py status         # Check system status
```

### Training
```bash
python cli.py train churn              # Train churn model
python cli.py train product-area       # Train product area model
python cli.py train all                # Train both models
```

### Scoring
```bash
python cli.py score accounts           # Score all accounts
python cli.py score product-areas      # Score product areas
python cli.py score all                # Score everything
```

### Docker
```bash
docker compose build                           # Build image
docker compose run tens-insight python cli.py status
docker compose run tens-insight python cli.py train all
docker compose run tens-insight python cli.py score all
```

## File Locations

```
Config:        .env (or environment variables)
Models:        models/*.keras, models/*.pkl
Logs:          stdout/stderr (redirect to files if needed)
Database:      Shared with goinsight
```

## Database Tables

### Input (from GoInsight)
- `feedback_enriched`: Customer feedback data

### Output (ML predictions)
- `account_risk_scores`: Account churn predictions
- `product_area_impact`: Product area priority scores

## Key Configuration

```bash
# Database
DATABASE_URL=postgres://goinsight:goinsight_dev_pass@postgres:5432/goinsight?sslmode=disable

# Training
EPOCHS=50
BATCH_SIZE=32
VALIDATION_SPLIT=0.2

# Scoring
SCORE_BATCH_SIZE=1000
```

## Monitoring Queries

### Check Predictions
```sql
-- Latest account predictions
SELECT account_id, churn_probability, health_score, risk_category
FROM account_risk_scores
ORDER BY predicted_at DESC
LIMIT 10;

-- High risk accounts
SELECT account_id, health_score, risk_category
FROM account_risk_scores
WHERE risk_category IN ('high', 'critical')
ORDER BY churn_probability DESC;

-- Top priority product areas
SELECT product_area, segment, priority_score, feedback_count
FROM product_area_impact
ORDER BY priority_score DESC
LIMIT 10;
```

### Check Data Quality
```sql
-- Feedback count by date
SELECT DATE(created_at), COUNT(*)
FROM feedback_enriched
GROUP BY DATE(created_at)
ORDER BY DATE(created_at) DESC
LIMIT 7;

-- Prediction freshness
SELECT 
    MAX(predicted_at) as latest_prediction,
    COUNT(*) as total_predictions
FROM account_risk_scores;
```

## Troubleshooting

### Database Connection Failed
```bash
# Test connection
psql "$DATABASE_URL"

# Check if goinsight is running
docker ps | grep goinsight
```

### No Data to Score
```bash
# Check feedback table
psql "$DATABASE_URL" -c "SELECT COUNT(*) FROM feedback_enriched"

# Run goinsight seed data if empty
cd ../goinsight && go run cmd/seed/main.go
```

### Model Not Found
```bash
# Check models directory
ls -lh models/

# Retrain models
python cli.py train all
```

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Check Python version (need 3.8+)
python --version
```

## Scheduling

### Daily Scoring (Cron)
```bash
# Add to crontab
0 3 * * * cd /path/to/tens-insight && python cli.py score all >> logs/score.log 2>&1
```

### Weekly Training (Cron)
```bash
# Add to crontab
0 2 * * 0 cd /path/to/tens-insight && python cli.py train all >> logs/train.log 2>&1
```

## Model Files

```
models/
├── churn_model_v1.keras          # Keras model
├── churn_model_v1_scaler.pkl     # Feature scaler
├── product_area_model_v1_config.pkl  # Model config
```

## Health Checks

### System Health
```bash
python cli.py status
```

### Database Health
```bash
psql "$DATABASE_URL" -c "SELECT 1"
```

### Model Health
```bash
ls models/*.keras && echo "Models OK" || echo "Models missing"
```

## API Integration

GoInsight reads predictions from these tables:

```go
// Example GoInsight code (hypothetical)

// Get account health
func GetAccountHealth(accountID string) (*Health, error) {
    var health Health
    err := db.QueryRow(`
        SELECT churn_probability, health_score, risk_category
        FROM account_risk_scores
        WHERE account_id = $1
    `, accountID).Scan(&health.ChurnProb, &health.Score, &health.Risk)
    return &health, err
}

// Get top priority product areas
func GetTopPriorities(limit int) ([]Priority, error) {
    rows, err := db.Query(`
        SELECT product_area, segment, priority_score
        FROM product_area_impact
        ORDER BY priority_score DESC
        LIMIT $1
    `, limit)
    // ... parse rows
}
```

## Performance

### Typical Runtimes (on moderate hardware)
- Setup: < 1 second
- Training churn model: 1-5 minutes (depends on data size)
- Training product area model: < 10 seconds
- Scoring accounts: < 30 seconds for 1000 accounts
- Scoring product areas: < 5 seconds

### Optimization Tips
- Use appropriate `SCORE_BATCH_SIZE` for your hardware
- Train on GPU if available (TensorFlow auto-detects)
- Index database prediction tables properly
- Run scoring off-peak hours

## Support

### Documentation
- `README.md` - Overview
- `USAGE.md` - Detailed usage guide  
- `ARCHITECTURE.md` - System design
- `PROJECT_SUMMARY.md` - Complete summary

### Code
- `src/` - Well-commented source code
- `cli.py` - Command-line interface
- `setup.py` - Setup script

### Community
- GitHub issues for bugs
- Discussions for questions
- Pull requests for contributions
