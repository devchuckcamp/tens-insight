#!/usr/bin/env python3
"""
Run database migrations for the continuous training system.

This script executes all pending SQL migrations to prepare the database.
"""

import logging
import os
import sys
from pathlib import Path

# Add parent directory to path so we can import src
sys.path.insert(0, str(Path(__file__).parent))

from src.db import get_connection

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_migrations():
    """Execute all pending migrations."""
    migrations_dir = Path(__file__).parent / 'migrations'
    
    # Get all migration files in order
    migration_files = sorted([f for f in migrations_dir.glob('*.sql')])
    
    if not migration_files:
        logger.warning("No migration files found")
        return False
    
    logger.info(f"Found {len(migration_files)} migration files")
    
    try:
        with get_connection() as conn:
            for migration_file in migration_files:
                logger.info(f"Running migration: {migration_file.name}")
                
                with open(migration_file, 'r') as f:
                    sql_content = f.read()
                
                # Execute migration
                try:
                    from sqlalchemy import text
                    # Split by semicolon to handle multiple statements
                    statements = [stmt.strip() for stmt in sql_content.split(';') if stmt.strip()]
                    
                    for statement in statements:
                        conn.execute(text(statement))
                    
                    conn.commit()
                    logger.info(f"✓ Successfully executed {migration_file.name}")
                
                except Exception as e:
                    logger.error(f"✗ Error in {migration_file.name}: {e}")
                    conn.rollback()
                    raise
        
        logger.info("✓ All migrations completed successfully")
        return True
    
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return False


if __name__ == '__main__':
    success = run_migrations()
    sys.exit(0 if success else 1)
