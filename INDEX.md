# Tens-Insight: Complete File Index

## 📁 Project Structure Overview

This document provides a complete index of all files in the tens-insight project.

## 📄 Documentation Files

| File | Description |
|------|-------------|
| `README.md` | Main project overview and quick start guide |
| `USAGE.md` | Comprehensive usage guide with examples |
| `ARCHITECTURE.md` | System architecture and design decisions |
| `PROJECT_SUMMARY.md` | Complete project summary and success metrics |
| `QUICK_REFERENCE.md` | Quick reference for common commands and queries |
| `INTEGRATION_GUIDE.md` | Integration guide for GoInsight + Tens-Insight |
| `INDEX.md` | This file - complete project index |

## 🐍 Source Code Files

### Core Modules

| File | Purpose | Key Classes/Functions |
|------|---------|---------------------|
| `src/__init__.py` | Package initialization | - |
| `src/config.py` | Configuration management | `Config`, `get_config()` |
| `src/db.py` | Database connectivity | `get_engine()`, `execute_query()`, `upsert_dataframe()`, `create_prediction_tables()` |

### Feature Engineering (`src/features/`)

| File | Purpose | Key Functions |
|------|---------|---------------|
| `src/features/__init__.py` | Package exports | All feature functions |
| `src/features/accounts.py` | Account-level features | `build_account_features()`, `prepare_training_data()`, `create_synthetic_labels()` |
| `src/features/product_areas.py` | Product area aggregation | `build_product_area_features()`, `calculate_priority_scores()`, `prepare_for_scoring()` |

### ML Models (`src/models/`)

| File | Purpose | Key Classes |
|------|---------|-------------|
| `src/models/__init__.py` | Package exports | All model classes |
| `src/models/churn_model.py` | TensorFlow churn prediction | `ChurnModel`, `create_churn_model()` |
| `src/models/product_area_model.py` | Rule-based priority scoring | `ProductAreaModel`, `create_product_area_model()` |

### Training Scripts (`src/training/`)

| File | Purpose | Entry Point |
|------|---------|-------------|
| `src/training/__init__.py` | Package exports | - |
| `src/training/train_churn.py` | Train churn model | `train_churn_model()`, `__main__` |
| `src/training/train_product_area.py` | Train product area model | `train_product_area_model()`, `__main__` |

### Scoring Scripts (`src/scoring/`)

| File | Purpose | Entry Point |
|------|---------|-------------|
| `src/scoring/__init__.py` | Package exports | - |
| `src/scoring/score_accounts.py` | Batch account scoring | `score_accounts()`, `__main__` |
| `src/scoring/score_product_areas.py` | Batch product area scoring | `score_product_areas()`, `__main__` |

## 🛠️ Configuration & Setup Files

| File | Purpose |
|------|---------|
| `.env.example` | Environment variable template |
| `.gitignore` | Git ignore rules |
| `requirements.txt` | Python dependencies |
| `setup.py` | Initial setup script |
| `cli.py` | Unified command-line interface |

## 🐳 Docker Files

| File | Purpose |
|------|---------|
| `Dockerfile` | Container image definition |
| `docker-compose.yml` | Docker orchestration configuration |

## 💾 Database Migrations

| File | Purpose |
|------|---------|
| `migrations/003_ml_predictions.sql` | Creates prediction tables (for GoInsight) |

## 📊 Generated/Runtime Files (Not in Repo)

These files are created when you run the pipeline:

| Directory/File | Purpose |
|----------------|---------|
| `models/` | Saved model artifacts |
| `models/churn_model_v1.keras` | Trained churn model |
| `models/churn_model_v1_scaler.pkl` | Feature scaler for churn model |
| `models/product_area_model_v1_config.pkl` | Product area model configuration |
| `.env` | Your actual environment variables (not committed) |

## 🎯 File Purpose by Use Case

### First-Time Setup
1. Read `README.md`
2. Copy `.env.example` to `.env`
3. Run `setup.py`
4. Check `USAGE.md` for next steps

### Running Training
1. `src/training/train_churn.py`
2. `src/training/train_product_area.py`
3. Or use `cli.py train all`

### Running Scoring
1. `src/scoring/score_accounts.py`
2. `src/scoring/score_product_areas.py`
3. Or use `cli.py score all`

### Understanding the System
1. `ARCHITECTURE.md` - system design
2. `PROJECT_SUMMARY.md` - complete overview
3. `INTEGRATION_GUIDE.md` - GoInsight integration

### Quick Reference
1. `QUICK_REFERENCE.md` - common commands
2. `cli.py` - unified interface

### Docker Deployment
1. `Dockerfile` - image definition
2. `docker-compose.yml` - orchestration
3. `INTEGRATION_GUIDE.md` - deployment guide

## 📝 Documentation by Audience

### For Developers
- `ARCHITECTURE.md` - System design and patterns
- `src/` files - Well-commented source code
- `INTEGRATION_GUIDE.md` - API integration examples

### For Data Scientists
- `src/features/` - Feature engineering logic
- `src/models/` - Model architectures
- `src/training/` - Training procedures
- `ARCHITECTURE.md` - Model selection rationale

### For DevOps/SRE
- `docker-compose.yml` - Container orchestration
- `Dockerfile` - Image definition
- `USAGE.md` - Scheduling and automation
- `QUICK_REFERENCE.md` - Monitoring queries

### For Product/Business
- `README.md` - High-level overview
- `PROJECT_SUMMARY.md` - Business value
- `QUICK_REFERENCE.md` - What the system does

## 🔍 Finding Things

### "How do I...?"

| Task | File(s) to Check |
|------|------------------|
| Install dependencies | `requirements.txt`, `README.md` |
| Configure the database | `.env.example`, `src/config.py` |
| Train a model | `src/training/*.py`, `cli.py` |
| Score accounts | `src/scoring/*.py`, `cli.py` |
| Build features | `src/features/*.py` |
| Modify models | `src/models/*.py` |
| Integrate with GoInsight | `INTEGRATION_GUIDE.md` |
| Deploy with Docker | `docker-compose.yml`, `Dockerfile` |
| Schedule jobs | `USAGE.md`, `QUICK_REFERENCE.md` |
| Monitor the system | `QUICK_REFERENCE.md`, `cli.py status` |
| Troubleshoot issues | `USAGE.md`, `INTEGRATION_GUIDE.md` |

### "What does this file do?"

See the tables above organized by directory/purpose.

### "Where is the code for...?"

| Feature | Location |
|---------|----------|
| Account features | `src/features/accounts.py` |
| Product area features | `src/features/product_areas.py` |
| Churn model | `src/models/churn_model.py` |
| Priority scoring | `src/models/product_area_model.py` |
| Database operations | `src/db.py` |
| Configuration | `src/config.py` |
| Training logic | `src/training/*.py` |
| Scoring logic | `src/scoring/*.py` |
| CLI commands | `cli.py` |
| Setup procedures | `setup.py` |

## 📈 Code Statistics

```
Total Python files: 15
Total documentation files: 7
Total configuration files: 5
Total lines of code: ~2,500+
```

### File Size Distribution

| Size Range | Count | Files |
|------------|-------|-------|
| < 100 lines | 6 | `__init__.py` files, config |
| 100-300 lines | 6 | Models, setup, cli |
| 300+ lines | 3 | Feature engineering, db |

### Code Organization

```
Documentation: 40% (comprehensive guides)
Source Code: 50% (modular, well-commented)
Configuration: 10% (Docker, deps, env)
```

## 🎓 Learning Path

For new contributors:

1. **Day 1**: Read `README.md` + `PROJECT_SUMMARY.md`
2. **Day 2**: Read `ARCHITECTURE.md`, explore `src/` code
3. **Day 3**: Run through `USAGE.md` examples
4. **Day 4**: Study `INTEGRATION_GUIDE.md`
5. **Day 5**: Make your first contribution!

## 📦 Dependencies

See `requirements.txt` for complete list. Key dependencies:

- TensorFlow 2.x
- Pandas + NumPy
- SQLAlchemy + psycopg2
- scikit-learn
- joblib

## 🔗 Related Files

### In GoInsight Repo
- `goinsight/migrations/001_init.sql` - Creates `feedback_enriched`
- `goinsight/migrations/002_seed_feedback.sql` - Sample data
- (Optionally) `goinsight/migrations/003_ml_predictions.sql` - Prediction tables

### External
- GoInsight README: Project overview
- GoInsight API docs: Endpoint specifications

## 📞 Support Resources

1. **Documentation**: Start with README.md
2. **Examples**: Check USAGE.md
3. **Architecture**: Read ARCHITECTURE.md
4. **Quick Help**: Use QUICK_REFERENCE.md
5. **Code Comments**: All Python files are well-commented

## 🚀 Quick Links

- Main README: `README.md`
- Usage Guide: `USAGE.md`
- Architecture: `ARCHITECTURE.md`
- CLI Tool: `cli.py`
- Setup Script: `setup.py`

---

**This index was last updated**: December 2025
**Project Version**: 0.1.0
**Status**: Production-ready MVP
