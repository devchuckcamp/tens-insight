# Tens-Insight: ML Pipeline Architecture

## Overview

Tens-Insight is a TensorFlow-based machine learning pipeline that complements the GoInsight customer feedback analytics platform. It trains predictive models and writes predictions back to the shared Postgres database.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        GoInsight API                         │
│  ┌────────────────┐  ┌──────────────┐  ┌─────────────────┐ │
│  │ /api/accounts/ │  │  /api/ask    │  │  /api/insights  │ │
│  │     health     │  │              │  │                 │ │
│  └────────┬───────┘  └──────┬───────┘  └────────┬────────┘ │
│           │                  │                    │          │
└───────────┼──────────────────┼────────────────────┼──────────┘
            │                  │                    │
            │        READ      │         READ       │
            ▼                  ▼                    ▼
┌─────────────────────────────────────────────────────────────┐
│                    Postgres Database                         │
│                                                              │
│  ┌──────────────────┐  ┌────────────────────────────────┐  │
│  │ feedback_enriched│  │  Predictive Tables (ML Output) │  │
│  │  (from ETL)      │  │  ┌──────────────────────────┐  │  │
│  │                  │  │  │ account_risk_scores      │  │  │
│  │  - sentiment     │  │  │  - churn_probability     │  │  │
│  │  - priority      │  │  │  - health_score          │  │  │
│  │  - product_area  │  │  │  - risk_category         │  │  │
│  │  - customer_tier │  │  │                          │  │  │
│  │  - region        │  │  │ product_area_impact      │  │  │
│  └──────────────────┘  │  │  - priority_score        │  │  │
│                        │  │  - feedback_count        │  │  │
│                        │  │  - sentiment_score       │  │  │
│                        │  └──────────────────────────┘  │  │
│                        └────────────────────────────────┘  │
│                                      ▲                      │
└──────────────────────────────────────┼──────────────────────┘
                                       │
                                   WRITE (UPSERT)
                                       │
┌──────────────────────────────────────┼──────────────────────┐
│                    Tens-Insight ML Pipeline                  │
│                                      │                       │
│  ┌────────────┐    ┌──────────────┐ │ ┌─────────────────┐  │
│  │  Feature   │───▶│  TensorFlow  │─┴─│   Predictions   │  │
│  │ Engineering│    │    Models    │   │   (Batch Score) │  │
│  └────────────┘    └──────────────┘   └─────────────────┘  │
│       │                   │                                 │
│       │                   │                                 │
│  ┌────▼───────┐    ┌──────▼────────┐                       │
│  │ accounts.py│    │ churn_model   │                       │
│  │ product_   │    │ product_area_ │                       │
│  │  areas.py  │    │     model     │                       │
│  └────────────┘    └───────────────┘                       │
│                                                             │
│  Training Scripts:           Scoring Scripts:              │
│  - train_churn.py            - score_accounts.py           │
│  - train_product_area.py     - score_product_areas.py      │
│                                                             │
│  Execution: CLI / Cron / CI/CD                              │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow

### Training Flow

1. **Feature Engineering**: Read `feedback_enriched` table
2. **Aggregate & Transform**: Build account/product-area features
3. **Model Training**: Train TensorFlow models
4. **Model Persistence**: Save models to `models/` directory

### Scoring Flow

1. **Load Model**: Load trained model from disk
2. **Feature Engineering**: Build features for all accounts/product-areas
3. **Batch Prediction**: Generate predictions
4. **Database Write**: UPSERT predictions into prediction tables

### GoInsight Consumption

1. **API Request**: User hits `/api/accounts/{id}/health`
2. **Database Read**: GoInsight queries `account_risk_scores`
3. **Response**: Returns health score and risk category
4. **Context**: May also query `product_area_impact` for insights

## Key Design Decisions

### Why Rule-Based for Product Areas?

The product area model uses weighted aggregation rather than learned weights because:
- **Interpretability**: Business stakeholders can understand the scoring
- **Simplicity**: No training data needed, just feedback patterns
- **Flexibility**: Weights can be easily adjusted based on business priorities
- **Sufficient**: For this use case, complex ML may be overkill

### Why TensorFlow for Churn?

- **Flexibility**: Can easily add more complex features/architectures
- **Industry Standard**: Well-documented, widely adopted
- **Integration**: Easy to deploy in various environments
- **Scalability**: Can handle larger datasets as the business grows

### Database-Centric Architecture

- **Single Source of Truth**: All data flows through Postgres
- **Decoupling**: ML pipeline doesn't need to know about GoInsight's API
- **Batch Processing**: Predictions computed offline, not on-demand
- **Caching**: Predictions are "cached" in DB until next scoring run

## Scaling Considerations

### Current State (MVP)

- ✅ Works for 100s-1000s of accounts
- ✅ Simple deployment (single Python process)
- ✅ Easy to debug and monitor
- ✅ Low operational complexity

### Future Scaling Paths

**For 10K+ accounts:**
- Use batch processing in chunks
- Parallelize feature computation
- Consider Dask/Ray for distributed computing

**For 100K+ accounts:**
- Move to Spark for feature engineering
- Use TensorFlow Extended (TFX) for production ML
- Implement feature stores (Feast, Tecton)
- Add model monitoring (Evidently, WhyLabs)

**For real-time scoring:**
- Deploy models as REST APIs (TensorFlow Serving)
- Use caching layer (Redis)
- Implement streaming pipelines (Kafka + Flink)

## Model Retraining Strategy

### Current: Manual

- Run training scripts on-demand
- Version models manually (v1, v2, etc.)
- Deploy by replacing model files

### Recommended: Scheduled

- **Weekly retraining**: Capture evolving patterns
- **A/B testing**: Compare v1 vs v2 performance
- **Validation gates**: Only deploy if metrics improve
- **Rollback capability**: Keep previous versions

### Future: Automated

- Trigger retraining on data drift detection
- Automated hyperparameter tuning
- Shadow deployment for new models
- Automated rollback on performance degradation

## Monitoring & Observability

### Metrics to Track

**Model Performance:**
- Prediction latency
- Batch scoring duration
- Model accuracy/AUC (if you have ground truth)

**Data Quality:**
- Feature distribution shifts
- Missing value rates
- Schema changes

**Business Impact:**
- Prediction distribution (risk categories)
- High-risk account alerts
- False positive/negative rates

### Logging

All scripts log:
- Input data statistics
- Model predictions summary
- Database write confirmation
- Errors and warnings

### Alerting

Set up alerts for:
- Scoring job failures
- Unusual prediction distributions
- Database connection issues
- Model loading errors

## Integration Points

### With GoInsight

**Read:**
- `feedback_enriched` table (input data)

**Write:**
- `account_risk_scores` (churn predictions)
- `product_area_impact` (priority scores)

**Shared:**
- Same Postgres database
- Same Docker network (if containerized)

### With ETL Pipeline

Tens-Insight assumes `feedback_enriched` is populated by an ETL process that:
- Ingests raw feedback from various sources
- Enriches with metadata (sentiment, priority, etc.)
- Writes to Postgres

### With Alerting Systems

Predictions can trigger alerts:
- High churn risk → notify customer success
- Critical product area → alert product team
- Sudden score changes → investigate

## Development Workflow

### Local Development

1. Connect to dev database
2. Run training scripts with small data samples
3. Test scoring scripts
4. Validate predictions in database

### Testing

- Unit tests for feature engineering
- Integration tests for database operations
- Model validation on hold-out set

### Deployment

- Build Docker image
- Deploy to production environment
- Schedule scoring jobs (cron/Airflow)
- Monitor logs and metrics

## Future Enhancements

### Short Term (1-3 months)

- [ ] Add more sophisticated features (trend analysis, seasonality)
- [ ] Implement proper train/validation/test splits
- [ ] Add model explainability (SHAP values)
- [ ] Create Jupyter notebooks for analysis

### Medium Term (3-6 months)

- [ ] A/B testing framework for models
- [ ] Automated hyperparameter tuning
- [ ] Feature importance analysis
- [ ] Model performance dashboards

### Long Term (6-12 months)

- [ ] Real-time scoring API
- [ ] Advanced architectures (transformers, GNNs)
- [ ] Multi-model ensembles
- [ ] Automated feature discovery
- [ ] MLOps platform integration (MLflow, Kubeflow)

## Resources

- **Documentation**: README.md, USAGE.md
- **Code**: `src/` directory (well-commented)
- **Models**: `models/` directory (persisted artifacts)
- **Config**: `.env.example` (configuration template)
