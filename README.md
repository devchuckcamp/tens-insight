# Tens-Insight: ML Pipeline for GoInsight

TensorFlow-based machine learning pipeline for the GoInsight customer feedback analytics platform.

## Overview

Tens-Insight is a Python/TensorFlow companion project for [GoInsight](https://github.com/devchuckcamp/goinsight). It provides ML models that:

- **Predict account churn risk** and health scores
- **Score product-area impact** by customer segment
- **Write predictions** back to Postgres tables that GoInsight consumes

## Architecture

```
tens-insight/
├── src/
│   ├── config.py              # Configuration & environment variables
│   ├── db.py                  # Database connection & utilities
│   ├── features/              # Feature engineering
│   │   ├── accounts.py        # Account-level features for churn model
│   │   └── product_areas.py   # Product-area aggregation features
│   ├── models/                # TensorFlow models
│   │   ├── churn_model.py     # Account churn prediction model
│   │   └── product_area_model.py  # Product-area priority scoring
│   ├── training/              # Model training scripts
│   │   ├── train_churn.py     # Train churn model
│   │   └── train_product_area.py  # Train product-area model
│   └── scoring/               # Batch scoring scripts
│       ├── score_accounts.py  # Score all accounts
│       └── score_product_areas.py  # Score product areas
├── models/                    # Saved model artifacts
├── requirements.txt           # Python dependencies
└── README.md

```

## Database Schema

Tens-Insight shares the same Postgres database as GoInsight. 

### Input Tables (from GoInsight)
- `feedback_enriched`: Customer feedback data

### Output Tables (ML predictions)
- `account_risk_scores`: Account-level churn risk and health scores
- `product_area_impact`: Product area priority scores by segment

## Prerequisites

- Python 3.8+ 
- Access to the GoInsight Postgres database
- GoInsight migrations must be run first (creates `feedback_enriched` table)

## Setup

### Option 1: Local Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env if needed (defaults work with goinsight)
   ```

3. **Run setup script:**
   ```bash
   python setup.py
   ```
   
   This will:
   - Create the `models/` directory
   - Test database connection
   - Create prediction tables

### Option 2: Docker Setup

1. **Build the image:**
   ```bash
   docker compose build
   ```

2. **Run setup:**
   ```bash
   docker compose run tens-insight python setup.py
   ```

## Usage

### Using Docker (Recommended)

```bash
# Check status
docker-compose run --rm tens-insight python cli.py status

# Train all models
docker-compose run --rm tens-insight python cli.py train all

# Score everything
docker-compose run --rm tens-insight python cli.py score all

# Train specific model
docker-compose run --rm tens-insight python cli.py train churn --lookback-days 90 --version v1

# Score specific target
docker-compose run --rm tens-insight python cli.py score accounts --version v1
```

### Using CLI Directly (inside container)

```bash
# First, enter the container
docker-compose run --rm tens-insight bash

# Then run commands
python cli.py status
python cli.py train all
python cli.py score all
```

## Development

- **Python 3.8+** required
- **TensorFlow 2.x** for model training
- **SQLAlchemy + psycopg2** for database access
- **Pandas + NumPy** for data manipulation

## Integration with GoInsight

GoInsight reads predictions from:
- `/api/accounts/{id}/health` → reads `account_risk_scores`
- `/api/ask` → may query `product_area_impact` for context

This repo runs independently via CLI/cron/CI and writes predictions to the shared database.
