"""Database connection and utilities for tens-insight.

Provides SQLAlchemy engine and helper functions for interacting with
the shared Postgres database.
"""

import logging
from typing import Any, Dict, List, Optional
from contextlib import contextmanager

import pandas as pd
from sqlalchemy import create_engine, text, Table, Column, Integer, String, Float, TIMESTAMP, MetaData
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from .config import get_config

logger = logging.getLogger(__name__)


def get_engine() -> Engine:
    """Create and return a SQLAlchemy engine."""
    config = get_config()
    engine = create_engine(
        config.database_url,
        pool_pre_ping=True,  # Verify connections before using
        pool_size=5,
        max_overflow=10
    )
    return engine


@contextmanager
def get_connection():
    """Context manager for database connections."""
    engine = get_engine()
    conn = engine.connect()
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


def execute_query(query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Execute a SQL query and return results as a pandas DataFrame.
    
    Args:
        query: SQL query to execute
        params: Optional query parameters
        
    Returns:
        DataFrame with query results
    """
    engine = get_engine()
    try:
        df = pd.read_sql_query(query, engine, params=params)
        logger.info(f"Query executed successfully, returned {len(df)} rows")
        return df
    except SQLAlchemyError as e:
        logger.error(f"Database query failed: {e}")
        raise


def upsert_dataframe(
    df: pd.DataFrame,
    table_name: str,
    conflict_columns: List[str],
    update_columns: List[str]
) -> int:
    """Upsert a DataFrame into a Postgres table using ON CONFLICT.
    
    Args:
        df: DataFrame to upsert
        table_name: Target table name
        conflict_columns: Columns to check for conflicts (unique constraint)
        update_columns: Columns to update on conflict
        
    Returns:
        Number of rows affected
    """
    if df.empty:
        logger.warning(f"Empty DataFrame, no rows to upsert into {table_name}")
        return 0
    
    with get_connection() as conn:
        # Convert DataFrame to list of dicts
        records = df.to_dict('records')
        
        # Build the INSERT statement
        columns = df.columns.tolist()
        column_names = ', '.join(columns)
        placeholders = ', '.join([f':{col}' for col in columns])
        
        # Build the ON CONFLICT clause
        conflict_cols = ', '.join(conflict_columns)
        update_set = ', '.join([f"{col} = EXCLUDED.{col}" for col in update_columns])
        
        query = f"""
            INSERT INTO {table_name} ({column_names})
            VALUES ({placeholders})
            ON CONFLICT ({conflict_cols})
            DO UPDATE SET {update_set}
        """
        
        try:
            result = conn.execute(text(query), records)
            rows_affected = result.rowcount if hasattr(result, 'rowcount') else len(records)
            logger.info(f"Upserted {rows_affected} rows into {table_name}")
            return rows_affected
        except SQLAlchemyError as e:
            logger.error(f"Upsert failed for table {table_name}: {e}")
            raise


def create_prediction_tables():
    """Create the prediction output tables if they don't exist.
    
    This should be run once during setup to ensure the tables exist.
    """
    engine = get_engine()
    metadata = MetaData()
    
    # Define account_risk_scores table
    account_risk_scores = Table(
        'account_risk_scores',
        metadata,
        Column('account_id', String, primary_key=True),
        Column('churn_probability', Float, nullable=False),
        Column('health_score', Float, nullable=False),
        Column('risk_category', String, nullable=False),
        Column('predicted_at', TIMESTAMP, nullable=False),
        Column('model_version', String, nullable=False),
    )
    
    # Define product_area_impact table
    product_area_impact = Table(
        'product_area_impact',
        metadata,
        Column('product_area', String, primary_key=True),
        Column('segment', String, primary_key=True),
        Column('priority_score', Float, nullable=False),
        Column('feedback_count', Integer, nullable=False),
        Column('avg_sentiment_score', Float, nullable=False),
        Column('negative_count', Integer, nullable=False),
        Column('critical_count', Integer, nullable=False),
        Column('predicted_at', TIMESTAMP, nullable=False),
        Column('model_version', String, nullable=False),
    )
    
    # Create tables
    try:
        metadata.create_all(engine)
        logger.info("Prediction tables created successfully")
    except SQLAlchemyError as e:
        logger.error(f"Failed to create prediction tables: {e}")
        raise


if __name__ == '__main__':
    """Run migrations when module is executed directly."""
    logging.basicConfig(level=logging.INFO)
    logger.info("Creating prediction tables...")
    create_prediction_tables()
    logger.info("Migration complete")
