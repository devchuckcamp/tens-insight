# Integration Guide: Tens-Insight + GoInsight

This guide explains how to integrate the Tens-Insight ML pipeline with the GoInsight API.

## Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     GoInsight API                            │
│  - Reads predictions from database                           │
│  - Displays health scores to users                           │
│  - Uses priorities in insights                               │
└────────────────────┬────────────────────────────────────────┘
                     │ READ
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                   Postgres Database                          │
│  - feedback_enriched (input)                                 │
│  - account_risk_scores (ML output)                           │
│  - product_area_impact (ML output)                           │
└────────────────────▲────────────────────────────────────────┘
                     │ WRITE
                     │
┌────────────────────┴────────────────────────────────────────┐
│                  Tens-Insight Pipeline                       │
│  - Trains ML models                                          │
│  - Scores accounts and product areas                         │
│  - Writes predictions to database                            │
└─────────────────────────────────────────────────────────────┘
```

## Prerequisites

1. GoInsight is running and migrations have been executed
2. `feedback_enriched` table exists and has data
3. Tens-Insight setup has been completed

## Setup Steps

### 1. Add ML Prediction Tables to GoInsight

**Option A: Run as a GoInsight migration**

Copy the migration file to goinsight:
```bash
cp tens-insight/migrations/003_ml_predictions.sql goinsight/migrations/
```

Restart goinsight to apply:
```bash
cd goinsight
docker compose restart api
```

**Option B: Run via Tens-Insight setup**

The tens-insight setup script creates the tables automatically:
```bash
cd tens-insight
python setup.py
```

### 2. Verify Tables Exist

```bash
psql "$DATABASE_URL" -c "\dt account_risk_scores"
psql "$DATABASE_URL" -c "\dt product_area_impact"
```

### 3. Run Initial Training

```bash
cd tens-insight
python cli.py train all
```

### 4. Run Initial Scoring

```bash
python cli.py score all
```

### 5. Verify Predictions

```bash
psql "$DATABASE_URL" -c "SELECT COUNT(*) FROM account_risk_scores"
psql "$DATABASE_URL" -c "SELECT COUNT(*) FROM product_area_impact"
```

## GoInsight API Integration

### Example: Account Health Endpoint

Add a new endpoint to `goinsight/internal/http/handlers.go`:

```go
package http

import (
    "database/sql"
    "encoding/json"
    "net/http"
    
    "github.com/go-chi/chi/v5"
)

// AccountHealth represents the ML-predicted health metrics
type AccountHealth struct {
    AccountID        string  `json:"account_id"`
    ChurnProbability float64 `json:"churn_probability"`
    HealthScore      float64 `json:"health_score"`
    RiskCategory     string  `json:"risk_category"`
    PredictedAt      string  `json:"predicted_at"`
    ModelVersion     string  `json:"model_version"`
}

// GetAccountHealth returns ML predictions for an account
func (h *Handler) GetAccountHealth(w http.ResponseWriter, r *http.Request) {
    accountID := chi.URLParam(r, "id")
    
    var health AccountHealth
    err := h.db.QueryRow(`
        SELECT 
            account_id,
            churn_probability,
            health_score,
            risk_category,
            predicted_at,
            model_version
        FROM account_risk_scores
        WHERE account_id = $1
    `, accountID).Scan(
        &health.AccountID,
        &health.ChurnProbability,
        &health.HealthScore,
        &health.RiskCategory,
        &health.PredictedAt,
        &health.ModelVersion,
    )
    
    if err == sql.ErrNoRows {
        http.Error(w, "Account health not found", http.StatusNotFound)
        return
    }
    if err != nil {
        http.Error(w, "Database error", http.StatusInternalServerError)
        return
    }
    
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(health)
}
```

Register the route in `goinsight/internal/http/router.go`:

```go
r.Get("/api/accounts/{id}/health", handler.GetAccountHealth)
```

### Example: Product Area Priorities Endpoint

```go
// ProductAreaPriority represents ML-predicted priorities
type ProductAreaPriority struct {
    ProductArea       string  `json:"product_area"`
    Segment           string  `json:"segment"`
    PriorityScore     float64 `json:"priority_score"`
    FeedbackCount     int     `json:"feedback_count"`
    AvgSentimentScore float64 `json:"avg_sentiment_score"`
    NegativeCount     int     `json:"negative_count"`
    CriticalCount     int     `json:"critical_count"`
}

// GetTopPriorities returns top priority product areas
func (h *Handler) GetTopPriorities(w http.ResponseWriter, r *http.Request) {
    limit := 10 // or parse from query param
    
    rows, err := h.db.Query(`
        SELECT 
            product_area,
            segment,
            priority_score,
            feedback_count,
            avg_sentiment_score,
            negative_count,
            critical_count
        FROM product_area_impact
        ORDER BY priority_score DESC
        LIMIT $1
    `, limit)
    if err != nil {
        http.Error(w, "Database error", http.StatusInternalServerError)
        return
    }
    defer rows.Close()
    
    var priorities []ProductAreaPriority
    for rows.Next() {
        var p ProductAreaPriority
        err := rows.Scan(
            &p.ProductArea,
            &p.Segment,
            &p.PriorityScore,
            &p.FeedbackCount,
            &p.AvgSentimentScore,
            &p.NegativeCount,
            &p.CriticalCount,
        )
        if err != nil {
            continue
        }
        priorities = append(priorities, p)
    }
    
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(map[string]interface{}{
        "priorities": priorities,
    })
}
```

### Example: Enhanced /api/ask with ML Context

Enhance the existing ask endpoint to include ML insights:

```go
func (h *Handler) HandleAsk(w http.ResponseWriter, r *http.Request) {
    // ... existing code for SQL generation and execution ...
    
    // Add ML insights
    mlInsights, err := h.getMLInsights()
    if err != nil {
        // Log but don't fail the request
        log.Printf("Failed to fetch ML insights: %v", err)
    }
    
    response := map[string]interface{}{
        "question":    question,
        "data_preview": dataPreview,
        "summary":     summary,
        "recommendations": recommendations,
        "actions":     actions,
        "ml_insights": mlInsights, // Add ML context
    }
    
    json.NewEncoder(w).Encode(response)
}

func (h *Handler) getMLInsights() (map[string]interface{}, error) {
    // Get high-risk accounts
    var highRiskCount int
    err := h.db.QueryRow(`
        SELECT COUNT(*) 
        FROM account_risk_scores 
        WHERE risk_category IN ('high', 'critical')
    `).Scan(&highRiskCount)
    if err != nil {
        return nil, err
    }
    
    // Get top priority area
    var topArea string
    var topScore float64
    err = h.db.QueryRow(`
        SELECT product_area, priority_score
        FROM product_area_impact
        ORDER BY priority_score DESC
        LIMIT 1
    `).Scan(&topArea, &topScore)
    if err != nil && err != sql.ErrNoRows {
        return nil, err
    }
    
    return map[string]interface{}{
        "high_risk_accounts": highRiskCount,
        "top_priority_area": topArea,
        "top_priority_score": topScore,
    }, nil
}
```

## Testing the Integration

### 1. Check Account Health

```bash
# Using the new endpoint
curl http://localhost:8080/api/accounts/enterprise/health

# Expected response
{
  "account_id": "enterprise",
  "churn_probability": 0.23,
  "health_score": 77.0,
  "risk_category": "low",
  "predicted_at": "2025-12-08T10:30:00Z",
  "model_version": "v1"
}
```

### 2. Check Top Priorities

```bash
curl http://localhost:8080/api/priorities

# Expected response
{
  "priorities": [
    {
      "product_area": "billing",
      "segment": "enterprise",
      "priority_score": 85.3,
      "feedback_count": 12,
      "avg_sentiment_score": -0.67,
      "negative_count": 10,
      "critical_count": 5
    },
    // ... more priorities
  ]
}
```

### 3. Check Enhanced Insights

```bash
curl -X POST http://localhost:8080/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are critical issues?"}'

# Response should include ml_insights field
{
  "question": "...",
  "summary": "...",
  "ml_insights": {
    "high_risk_accounts": 3,
    "top_priority_area": "billing",
    "top_priority_score": 85.3
  }
}
```

## Scheduling & Automation

### Option 1: Separate Services

Run goinsight and tens-insight as separate services:

```yaml
# docker-compose.yml (combined)
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      - POSTGRES_USER=goinsight
      - POSTGRES_PASSWORD=goinsight_dev_pass
      - POSTGRES_DB=goinsight
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - app_network

  goinsight-api:
    build: ./goinsight
    depends_on:
      - postgres
    environment:
      - DATABASE_URL=postgres://goinsight:goinsight_dev_pass@postgres:5432/goinsight?sslmode=disable
    ports:
      - "8080:8080"
    networks:
      - app_network

  tens-insight:
    build: ./tens-insight
    depends_on:
      - postgres
    environment:
      - DATABASE_URL=postgres://goinsight:goinsight_dev_pass@postgres:5432/goinsight?sslmode=disable
    volumes:
      - ./tens-insight/models:/app/models
    networks:
      - app_network
    # Run scoring on startup
    command: python cli.py score all

networks:
  app_network:
    driver: bridge

volumes:
  postgres_data:
```

### Option 2: Cron-based Scheduling

Add a cron service to run scoring periodically:

```yaml
  tens-insight-cron:
    build: ./tens-insight
    depends_on:
      - postgres
    environment:
      - DATABASE_URL=postgres://goinsight:goinsight_dev_pass@postgres:5432/goinsight?sslmode=disable
    volumes:
      - ./tens-insight/models:/app/models
    networks:
      - app_network
    command: >
      sh -c "
      while true; do
        python cli.py score all
        sleep 86400
      done
      "
```

### Option 3: External Cron

Run tens-insight scoring via cron on the host:

```bash
# /etc/cron.d/tens-insight
0 3 * * * docker compose -f /path/to/docker-compose.yml run tens-insight python cli.py score all
```

## Monitoring

### Add Health Check Endpoint in GoInsight

```go
// MLHealthCheck checks if ML predictions are fresh
func (h *Handler) MLHealthCheck(w http.ResponseWriter, r *http.Request) {
    var lastPrediction time.Time
    err := h.db.QueryRow(`
        SELECT MAX(predicted_at) FROM account_risk_scores
    `).Scan(&lastPrediction)
    
    if err != nil {
        http.Error(w, "ML predictions not available", http.StatusServiceUnavailable)
        return
    }
    
    // Check if predictions are stale (> 25 hours old)
    if time.Since(lastPrediction) > 25*time.Hour {
        http.Error(w, "ML predictions are stale", http.StatusServiceUnavailable)
        return
    }
    
    w.WriteHeader(http.StatusOK)
    json.NewEncoder(w).Encode(map[string]interface{}{
        "status": "healthy",
        "last_prediction": lastPrediction,
    })
}
```

### Monitoring Dashboard Queries

```sql
-- Check prediction freshness
SELECT 
    'account_risk_scores' as table_name,
    MAX(predicted_at) as latest_prediction,
    COUNT(*) as total_predictions
FROM account_risk_scores
UNION ALL
SELECT 
    'product_area_impact',
    MAX(predicted_at),
    COUNT(*)
FROM product_area_impact;

-- High-risk accounts
SELECT COUNT(*), risk_category
FROM account_risk_scores
GROUP BY risk_category;

-- Top priorities
SELECT product_area, segment, priority_score
FROM product_area_impact
ORDER BY priority_score DESC
LIMIT 5;
```

## Best Practices

1. **Run tens-insight on a schedule**: Daily scoring, weekly training
2. **Monitor prediction freshness**: Alert if predictions are > 25 hours old
3. **Version your models**: Track which model version produced each prediction
4. **Handle missing predictions gracefully**: Account might not have a score yet
5. **Cache prediction reads**: Consider Redis for frequently accessed predictions
6. **Log ML endpoint usage**: Track which GoInsight endpoints use ML data
7. **Set up alerts**: High-risk account counts, stale predictions, scoring failures

## Troubleshooting

### Predictions Not Appearing in API

1. Check if scoring has run:
   ```bash
   psql "$DATABASE_URL" -c "SELECT COUNT(*) FROM account_risk_scores"
   ```

2. Check last prediction time:
   ```bash
   psql "$DATABASE_URL" -c "SELECT MAX(predicted_at) FROM account_risk_scores"
   ```

3. Run scoring manually:
   ```bash
   cd tens-insight && python cli.py score all
   ```

### GoInsight Can't Read Predictions

1. Verify tables exist:
   ```bash
   psql "$DATABASE_URL" -c "\dt"
   ```

2. Check table permissions:
   ```bash
   psql "$DATABASE_URL" -c "\dp account_risk_scores"
   ```

3. Test query directly:
   ```bash
   psql "$DATABASE_URL" -c "SELECT * FROM account_risk_scores LIMIT 1"
   ```

## Next Steps

1. ✅ Set up tens-insight pipeline
2. ✅ Add ML prediction tables
3. ✅ Run initial training and scoring
4. ⬜ Add GoInsight API endpoints
5. ⬜ Schedule tens-insight to run daily
6. ⬜ Add monitoring and alerting
7. ⬜ Update GoInsight UI to display ML insights
8. ⬜ Document for end users

## Resources

- GoInsight: https://github.com/devchuckcamp/goinsight
- Tens-Insight: (this repo)
- PostgreSQL Docs: https://www.postgresql.org/docs/
