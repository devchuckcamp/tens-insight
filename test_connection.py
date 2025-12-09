#!/usr/bin/env python
"""Test database connection to GoInsight postgres container.

This script tests if tens-insight can connect to the GoInsight database.
Run this before running setup.py or training scripts.
"""

import os
import sys

# Set up environment
os.environ['DATABASE_URL'] = 'postgresql://goinsight:goinsight_dev_pass@goinsight-postgres:5432/goinsight?sslmode=disable'

print("=" * 60)
print("Testing Database Connection")
print("=" * 60)
print(f"Database URL: {os.environ['DATABASE_URL'].replace('@', '@***')}")
print()

# Test 1: Import modules
print("1. Testing imports...")
try:
    from src.db import execute_query
    print("   ✓ Modules imported successfully")
except ImportError as e:
    print(f"   ✗ Import failed: {e}")
    print()
    print("   Solution: Install dependencies first:")
    print("   pip install -r requirements.txt")
    sys.exit(1)

# Test 2: Database connection
print()
print("2. Testing database connection...")
try:
    df = execute_query('SELECT 1 as test')
    print("   ✓ Database connection successful!")
    print(f"   Result: {df.iloc[0, 0]}")
except Exception as e:
    print(f"   ✗ Database connection failed: {e}")
    print()
    print("   Possible solutions:")
    print("   1. Check if GoInsight postgres container is running:")
    print("      docker ps | grep postgres")
    print()
    print("   2. Verify the password in GoInsight's .env file")
    print("      If POSTGRES_PASSWORD is set, update DATABASE_URL in tens-insight/.env:")
    print("      DATABASE_URL=postgres://goinsight:YOUR_PASSWORD@localhost:5432/goinsight?sslmode=disable")
    print()
    print("   3. Check if port 5432 is exposed:")
    print("      docker port goinsight-postgres")
    sys.exit(1)

# Test 3: Check for feedback_enriched table
print()
print("3. Checking for feedback_enriched table...")
try:
    df = execute_query('SELECT COUNT(*) as count FROM feedback_enriched')
    count = df.iloc[0, 0]
    print(f"   ✓ Found feedback_enriched table with {count} rows")
    if count == 0:
        print("   ⚠ Warning: Table is empty. Run GoInsight seed data first:")
        print("      cd ../goinsight && go run cmd/seed/main.go")
except Exception as e:
    print(f"   ✗ Table not found: {e}")
    print()
    print("   Solution: Run GoInsight migrations first:")
    print("   cd ../goinsight")
    print("   docker compose up --build")

# Test 4: Check for prediction tables (optional)
print()
print("4. Checking for prediction tables...")
try:
    df = execute_query('SELECT COUNT(*) as count FROM account_risk_scores')
    print("   ✓ account_risk_scores table exists")
except Exception as e:
    print("   ⓘ account_risk_scores table not found (expected on first run)")
    print("      Run 'python setup.py' to create prediction tables")

try:
    df = execute_query('SELECT COUNT(*) as count FROM product_area_impact')
    print("   ✓ product_area_impact table exists")
except Exception as e:
    print("   ⓘ product_area_impact table not found (expected on first run)")

print()
print("=" * 60)
print("Connection test complete!")
print("=" * 60)
print()
print("Next steps:")
print("1. Run setup: python setup.py")
print("2. Train models: python cli.py train all")
print("3. Generate predictions: python cli.py score all")
