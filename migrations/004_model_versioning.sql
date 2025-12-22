-- Model Versioning Schema
-- Tracks all model versions, training runs, and metrics
-- Does not modify existing account_predictions or product_area_predictions tables

-- Table: model_versions
-- Stores metadata for each model version
CREATE TABLE IF NOT EXISTS model_versions (
    id SERIAL PRIMARY KEY,
    model_type VARCHAR(50) NOT NULL,  -- 'churn' or 'product_area'
    version VARCHAR(50) NOT NULL,      -- e.g., 'v1', 'v2', 'v1.2.3'
    status VARCHAR(20) NOT NULL DEFAULT 'inactive',  -- 'active', 'inactive', 'archived', 'failed'
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(255),           -- Username or scheduler identifier
    description TEXT,
    hyperparameters JSONB,             -- Training hyperparameters used
    model_path VARCHAR(500),           -- Relative path to model file
    scaler_path VARCHAR(500),          -- Relative path to scaler/config file
    UNIQUE(model_type, version)
);

CREATE INDEX IF NOT EXISTS idx_model_versions_type_status ON model_versions(model_type, status);
CREATE INDEX IF NOT EXISTS idx_model_versions_created_at ON model_versions(created_at DESC);

-- Table: training_runs
-- Detailed log of each training execution
CREATE TABLE IF NOT EXISTS training_runs (
    id SERIAL PRIMARY KEY,
    model_version_id INTEGER NOT NULL REFERENCES model_versions(id) ON DELETE CASCADE,
    started_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMPTZ,
    status VARCHAR(20) NOT NULL,       -- 'running', 'completed', 'failed'
    error_message TEXT,
    training_samples INTEGER,          -- Number of samples used for training
    test_samples INTEGER,              -- Number of samples used for testing
    data_freshness_check JSONB,        -- Metadata about data freshness decision
    duration_seconds INTEGER,          -- Total training time
    triggered_by VARCHAR(255),         -- 'scheduler', 'manual', 'api'
    FOREIGN KEY (model_version_id) REFERENCES model_versions(id)
);

CREATE INDEX IF NOT EXISTS idx_training_runs_status ON training_runs(status);
CREATE INDEX IF NOT EXISTS idx_training_runs_created_at ON training_runs(started_at DESC);

-- Table: model_metrics
-- Performance metrics for each model version
CREATE TABLE IF NOT EXISTS model_metrics (
    id SERIAL PRIMARY KEY,
    model_version_id INTEGER NOT NULL,
    training_run_id INTEGER,
    metric_name VARCHAR(100) NOT NULL,  -- e.g., 'accuracy', 'precision', 'recall', 'auc', 'f1'
    metric_value NUMERIC(10, 4),
    dataset_type VARCHAR(20),           -- 'train', 'validation', 'test'
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (model_version_id) REFERENCES model_versions(id) ON DELETE CASCADE,
    FOREIGN KEY (training_run_id) REFERENCES training_runs(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_model_metrics_version ON model_metrics(model_version_id);
CREATE INDEX IF NOT EXISTS idx_model_metrics_name ON model_metrics(metric_name);

-- Table: model_comparisons
-- Tracks performance comparisons between versions
CREATE TABLE IF NOT EXISTS model_comparisons (
    id SERIAL PRIMARY KEY,
    model_type VARCHAR(50) NOT NULL,
    previous_version_id INTEGER REFERENCES model_versions(id) ON DELETE SET NULL,
    new_version_id INTEGER NOT NULL REFERENCES model_versions(id) ON DELETE CASCADE,
    comparison_date TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metric_improvements JSONB,         -- e.g., {"accuracy": 0.02, "auc": -0.001}
    promotion_recommended BOOLEAN,      -- Whether new version should replace current
    promotion_reason TEXT,
    promotion_threshold NUMERIC(5, 3)  -- Minimum improvement required
);

CREATE INDEX IF NOT EXISTS idx_model_comparisons_type ON model_comparisons(model_type);

-- Table: model_deployments
-- Tracks which version is currently active in production
CREATE TABLE IF NOT EXISTS model_deployments (
    id SERIAL PRIMARY KEY,
    model_type VARCHAR(50) NOT NULL UNIQUE,
    active_version_id INTEGER NOT NULL,
    deployed_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deployed_by VARCHAR(255),
    previous_version_id INTEGER REFERENCES model_versions(id) ON DELETE SET NULL,
    rollback_available BOOLEAN DEFAULT TRUE,
    notes TEXT,
    FOREIGN KEY (active_version_id) REFERENCES model_versions(id)
);

CREATE INDEX IF NOT EXISTS idx_model_deployments_type ON model_deployments(model_type);

-- Table: data_freshness_log
-- Tracks data freshness checks for incremental training decisions
CREATE TABLE IF NOT EXISTS data_freshness_log (
    id SERIAL PRIMARY KEY,
    model_type VARCHAR(50) NOT NULL,
    check_timestamp TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_training_date TIMESTAMPTZ,
    latest_feedback_date TIMESTAMPTZ,
    new_feedback_count INTEGER,
    percent_new_data NUMERIC(5, 2),
    decision_to_retrain BOOLEAN,
    reason TEXT
);

CREATE INDEX IF NOT EXISTS idx_data_freshness_model_type ON data_freshness_log(model_type);
CREATE INDEX IF NOT EXISTS idx_data_freshness_timestamp ON data_freshness_log(check_timestamp DESC);

-- Table: prediction_distribution_history
-- Captures prediction distribution per version (for health checks)
CREATE TABLE IF NOT EXISTS prediction_distribution_history (
    id SERIAL PRIMARY KEY,
    model_type VARCHAR(50) NOT NULL,
    model_version_id INTEGER NOT NULL,
    scoring_date TIMESTAMPTZ NOT NULL,
    metric_name VARCHAR(100),          -- e.g., 'mean_prediction', 'std_prediction', 'min', 'max'
    metric_value NUMERIC(10, 4),
    samples_count INTEGER,
    FOREIGN KEY (model_version_id) REFERENCES model_versions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_prediction_dist_version ON prediction_distribution_history(model_version_id);
CREATE INDEX IF NOT EXISTS idx_prediction_dist_date ON prediction_distribution_history(scoring_date DESC);
