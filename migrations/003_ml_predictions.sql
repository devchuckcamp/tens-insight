-- Migration: Add ML prediction tables for tens-insight integration
-- This should be added to goinsight/migrations/ as 003_ml_predictions.sql

-- Create account_risk_scores table
-- Stores ML predictions for account churn risk and health scores
CREATE TABLE IF NOT EXISTS account_risk_scores (
    account_id VARCHAR PRIMARY KEY,
    churn_probability FLOAT NOT NULL CHECK (churn_probability >= 0 AND churn_probability <= 1),
    health_score FLOAT NOT NULL CHECK (health_score >= 0 AND health_score <= 100),
    risk_category VARCHAR NOT NULL CHECK (risk_category IN ('low', 'medium', 'high', 'critical')),
    predicted_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    model_version VARCHAR NOT NULL
);

-- Create indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_account_risk_category ON account_risk_scores(risk_category);
CREATE INDEX IF NOT EXISTS idx_account_predicted_at ON account_risk_scores(predicted_at DESC);
CREATE INDEX IF NOT EXISTS idx_account_health_score ON account_risk_scores(health_score);

-- Create product_area_impact table
-- Stores ML predictions for product area priority scores by segment
CREATE TABLE IF NOT EXISTS product_area_impact (
    product_area VARCHAR NOT NULL,
    segment VARCHAR NOT NULL,
    priority_score FLOAT NOT NULL CHECK (priority_score >= 0 AND priority_score <= 100),
    feedback_count INTEGER NOT NULL CHECK (feedback_count >= 0),
    avg_sentiment_score FLOAT NOT NULL CHECK (avg_sentiment_score >= -1 AND avg_sentiment_score <= 1),
    negative_count INTEGER NOT NULL CHECK (negative_count >= 0),
    critical_count INTEGER NOT NULL CHECK (critical_count >= 0),
    predicted_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    model_version VARCHAR NOT NULL,
    PRIMARY KEY (product_area, segment)
);

-- Create indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_product_area_priority ON product_area_impact(priority_score DESC);
CREATE INDEX IF NOT EXISTS idx_product_area_predicted_at ON product_area_impact(predicted_at DESC);
CREATE INDEX IF NOT EXISTS idx_product_area_segment ON product_area_impact(segment);

-- Add comments for documentation
COMMENT ON TABLE account_risk_scores IS 'ML predictions for account churn risk and health scores (populated by tens-insight)';
COMMENT ON TABLE product_area_impact IS 'ML predictions for product area priority scores by segment (populated by tens-insight)';

COMMENT ON COLUMN account_risk_scores.churn_probability IS 'Predicted probability of account churn (0-1)';
COMMENT ON COLUMN account_risk_scores.health_score IS 'Account health score (0-100, inverse of churn probability)';
COMMENT ON COLUMN account_risk_scores.risk_category IS 'Risk category: low (<25%), medium (25-50%), high (50-75%), critical (>75%)';

COMMENT ON COLUMN product_area_impact.priority_score IS 'Priority score for this product area/segment combination (0-100)';
COMMENT ON COLUMN product_area_impact.avg_sentiment_score IS 'Average sentiment score (-1 to 1, negative to positive)';
