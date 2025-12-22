# Tens-Insight Project Summary

## What We Built

A complete, production-ready TensorFlow ML pipeline for the GoInsight customer feedback analytics platform.

## Project Structure

```
tens-insight/
├── src/
│   ├── __init__.py
│   ├── config.py                    # Configuration management
│   ├── db.py                        # Database connection & utilities
│   ├── features/
│   │   ├── __init__.py
│   │   ├── accounts.py              # Account-level feature engineering
│   │   └── product_areas.py         # Product-area aggregation features
│   ├── models/
│   │   ├── __init__.py
│   │   ├── churn_model.py           # TensorFlow churn prediction model
│   │   └── product_area_model.py    # Rule-based priority scoring
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train_churn.py           # Churn model training script
│   │   └── train_product_area.py    # Product area training script
│   └── scoring/
│       ├── __init__.py
│       ├── score_accounts.py        # Batch account scoring
│       └── score_product_areas.py   # Batch product area scoring
├── models/                          # Saved model artifacts (created by training)
├── .env.example                     # Environment variable template
├── .gitignore                       # Git ignore rules
├── ARCHITECTURE.md                  # Detailed architecture documentation
├── cli.py                           # Unified command-line interface
├── docker-compose.yml               # Docker orchestration
├── Dockerfile                       # Container image definition
├── README.md                        # Project overview
├── requirements.txt                 # Python dependencies
├── setup.py                         # Setup script
└── USAGE.md                         # Comprehensive usage guide
```

## Key Features

### 1. Database Integration
- Connects to GoInsight's Postgres database
- Reads from `feedback_enriched` table
- Creates prediction tables (`account_risk_scores`, `product_area_impact`)
- UPSERT functionality for efficient updates

### 2. Feature Engineering
- Account-level features for churn prediction
- Product-area aggregation by segment
- Derived features (ratios, trends, flags)
- Categorical encoding
- Missing value handling

### 3. ML Models
- **Churn Model**: TensorFlow neural network for account churn prediction
- **Product Area Model**: Rule-based priority scoring
- Model versioning support
- Model persistence (save/load)
- Preprocessing pipelines (StandardScaler)

### 4. Training Pipeline
- Modular training scripts
- Train/test splitting
- Early stopping & learning rate scheduling
- Comprehensive logging
- Metrics reporting (accuracy, AUC, precision, recall)
- Synthetic label generation (for demo purposes)

### 5. Scoring Pipeline
- Batch prediction for all accounts
- Batch scoring for product areas
- Health score calculation (0-100 scale)
- Risk categorization (low/medium/high/critical)
- Database write with UPSERT
- Prediction statistics and summaries

### 6. Operations
- Environment variable configuration
- Command-line interface (cli.py)
- Docker containerization
- Docker Compose orchestration
- Setup script for initialization
- Status checking utility

### 7. Documentation
- README with project overview
- USAGE guide with examples
- ARCHITECTURE document with design decisions
- Code comments throughout
- Example configurations

## Quick Start

```bash
# 1. Setup
cd tens-insight
pip install -r requirements.txt
python setup.py

# 2. Train models
python cli.py train all

# 3. Run scoring
python cli.py score all

# 4. Check status
python cli.py status
```

## Output Tables

### account_risk_scores
```sql
CREATE TABLE account_risk_scores (
    account_id VARCHAR PRIMARY KEY,
    churn_probability FLOAT NOT NULL,
    health_score FLOAT NOT NULL,
    risk_category VARCHAR NOT NULL,
    predicted_at TIMESTAMP NOT NULL,
    model_version VARCHAR NOT NULL
);
```

### product_area_impact
```sql
CREATE TABLE product_area_impact (
    product_area VARCHAR,
    segment VARCHAR,
    priority_score FLOAT NOT NULL,
    feedback_count INTEGER NOT NULL,
    avg_sentiment_score FLOAT NOT NULL,
    negative_count INTEGER NOT NULL,
    critical_count INTEGER NOT NULL,
    predicted_at TIMESTAMP NOT NULL,
    model_version VARCHAR NOT NULL,
    PRIMARY KEY (product_area, segment)
);
```

## Workflow

1. **Training** (weekly/monthly):
   ```bash
   python -m src.training.train_churn
   python -m src.training.train_product_area
   ```

2. **Scoring** (daily):
   ```bash
   python -m src.scoring.score_accounts
   python -m src.scoring.score_product_areas
   ```

3. **GoInsight consumes predictions**:
   - `/api/accounts/{id}/health` reads `account_risk_scores`
   - `/api/ask` may query `product_area_impact` for context

## Design Principles

1. **Modular**: Clear separation of concerns (features, models, training, scoring)
2. **Database-Centric**: All data flows through Postgres
3. **Batch Processing**: Predictions computed offline, cached in DB
4. **Versioned**: Support for multiple model versions
5. **Observable**: Comprehensive logging at every step
6. **Reproducible**: Seeded randomness, documented processes
7. **Simple**: No unnecessary complexity, easy to understand and maintain

## Technologies

- **Python 3.11+**: Core language
- **TensorFlow 2.x**: Deep learning framework
- **Pandas/NumPy**: Data manipulation
- **SQLAlchemy**: Database ORM
- **psycopg2**: PostgreSQL driver
- **scikit-learn**: Preprocessing & metrics
- **Docker**: Containerization

## What's Next?

### Ready for Production
- [x] Database integration
- [x] Model training
- [x] Batch scoring
- [x] Docker deployment
- [x] Documentation

### Future Enhancements
- [ ] A/B testing framework
- [ ] Model monitoring dashboards
- [ ] Feature importance analysis
- [ ] Automated hyperparameter tuning
- [ ] Real-time scoring API
- [ ] Advanced model architectures

## Key Insights

### Why This Architecture?

1. **Decoupling**: ML pipeline runs independently of GoInsight API
2. **Scalability**: Easy to add more models or features
3. **Maintainability**: Clear structure, well-documented
4. **Flexibility**: Can swap out models without changing API
5. **Cost-Effective**: Batch processing is cheaper than real-time

### Assumptions Made

1. **Schema**: Used GoInsight's actual schema from migrations
2. **No accounts table yet**: Used `customer_tier` as proxy for accounts
3. **Synthetic labels**: Created synthetic churn labels for demo (replace with real data)
4. **Rule-based product scoring**: Sufficient for MVP, can upgrade to ML later

### Production Considerations

1. **Run setup first**: Creates prediction tables
2. **Train before scoring**: Can't score without trained models
3. **Monitor data quality**: Watch for schema changes in GoInsight
4. **Version your models**: Easy rollback if new version underperforms
5. **Schedule wisely**: Train weekly, score daily (or as needed)

## 📞 Integration with GoInsight

### Shared Database
```
DATABASE_URL=postgres://goinsight:goinsight_dev_pass@postgres:5432/goinsight?sslmode=disable
```

### Input Tables (from GoInsight)
- `feedback_enriched`: Customer feedback data

### Output Tables (for GoInsight)
- `account_risk_scores`: Churn predictions
- `product_area_impact`: Priority scores

### No Direct API Calls
- Tens-Insight is purely ETL/ML
- No HTTP endpoints
- No dependency on GoInsight API

## Validation Checklist

- [x] Connects to Postgres database
- [x] Reads feedback_enriched table
- [x] Creates prediction tables
- [x] Trains churn model
- [x] Trains product area model
- [x] Generates account predictions
- [x] Generates product area predictions
- [x] Writes predictions to database
- [x] Handles missing data gracefully
- [x] Logs all operations
- [x] Can be run via CLI
- [x] Can be run in Docker
- [x] Has comprehensive documentation

## Learning Resources

### For Understanding the Code
1. Start with `README.md` for overview
2. Read `ARCHITECTURE.md` for design decisions
3. Check `USAGE.md` for practical examples
4. Explore `src/` directory (well-commented)

### For Running the Pipeline
1. Run `python cli.py status` to check setup
2. Use `python cli.py train all` for first-time training
3. Use `python cli.py score all` for batch predictions
4. Check logs for detailed information

### For Extending the Pipeline
1. Add features in `src/features/`
2. Modify models in `src/models/`
3. Update training scripts in `src/training/`
4. Adjust scoring scripts in `src/scoring/`

## Success Metrics

The pipeline is successful if:
- Trains models without errors
- Generates predictions for all accounts
- Writes predictions to database
- GoInsight can query and display predictions
- Predictions are interpretable and actionable
- Can be run on a schedule (cron/CI)

---

**Built for the GoInsight project**
