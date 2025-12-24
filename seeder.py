#!/usr/bin/env python
"""Feedback Data Seeder with AI-generated realistic content.

Generates realistic feedback entries using GROQ AI for natural language summaries.
Can be run standalone or imported for programmatic use.

Usage:
    python seeder.py --count 50 --provider groq
    python seeder.py --count 100 --distribution balanced
"""

import argparse
import logging
import os
import random
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from uuid import uuid4

import requests
from dotenv import load_dotenv

from src.db import get_connection

# Load environment variables
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeedbackSeeder:
    """Generate and insert realistic feedback data."""
    
    # Distribution templates for realistic variety
    PRODUCT_AREAS = ['billing', 'onboarding', 'performance', 'features', 'security', 'ui_ux', 'integrations', 'support']
    SOURCES = ['zendesk', 'nps_survey', 'google_play', 'app_store', 'email', 'chat']
    SENTIMENTS = ['positive', 'neutral', 'negative']
    PRIORITIES = [1, 2, 3, 4, 5]  # 1=low, 5=critical
    REGIONS = ['NA', 'EU', 'APAC', 'LATAM', 'MEA']
    CUSTOMER_TIERS = ['free', 'pro', 'enterprise']
    
    # Topic templates by product area
    TOPICS_BY_AREA = {
        'billing': ['refund processing', 'invoice errors', 'payment methods', 'subscription cancellation', 'pricing concerns', 'billing cycles'],
        'onboarding': ['setup wizard', 'missing documentation', 'tutorial completion', 'data import', 'quick start guide', 'account activation'],
        'performance': ['app crashes', 'slow load times', 'memory usage', 'mobile responsiveness', 'api latency', 'database timeouts'],
        'features': ['export functionality', 'api access', 'dark mode', 'collaboration tools', 'notifications', 'search capabilities'],
        'security': ['data breach concerns', 'authentication', 'two-factor auth', 'access controls', 'encryption', 'compliance'],
        'ui_ux': ['design improvements', 'navigation', 'color scheme', 'mobile layout', 'accessibility', 'keyboard shortcuts'],
        'integrations': ['slack integration', 'jira sync', 'zapier support', 'api rate limits', 'webhook reliability', 'oauth setup'],
        'support': ['response time', 'knowledge base', 'chat support', 'ticket resolution', 'documentation quality', 'training materials']
    }
    
    def __init__(self, llm_provider: str = 'groq', api_key: Optional[str] = None):
        """Initialize seeder with LLM configuration.
        
        Args:
            llm_provider: LLM provider to use ('groq', 'openai', or 'none')
            api_key: API key for the provider (uses env var if not provided)
        """
        self.llm_provider = llm_provider.lower()
        
        if self.llm_provider == 'groq':
            self.api_key = api_key or os.getenv('GROQ_API_KEY')
            self.api_url = 'https://api.groq.com/openai/v1/chat/completions'
            self.model = 'mixtral-8x7b-32768'
        elif self.llm_provider == 'openai':
            self.api_key = api_key or os.getenv('OPENAI_API_KEY')
            self.api_url = 'https://api.openai.com/v1/chat/completions'
            self.model = 'gpt-3.5-turbo'
        else:
            self.api_key = None
            logger.warning("No LLM provider configured, using template-based summaries")
    
    def generate_summary(self, product_area: str, topic: str, sentiment: str, priority: int) -> str:
        """Generate realistic feedback summary using LLM or templates.
        
        Args:
            product_area: Product area for context
            topic: Specific topic
            sentiment: Sentiment (positive/neutral/negative)
            priority: Priority level (1-5)
            
        Returns:
            Generated summary text
        """
        if not self.api_key or self.llm_provider == 'none':
            return self._generate_template_summary(product_area, topic, sentiment, priority)
        
        prompt = f"""Generate a realistic customer feedback summary for a SaaS product.

Product Area: {product_area}
Topic: {topic}
Sentiment: {sentiment}
Priority: {priority}/5 (1=low, 5=critical)

Generate a brief, realistic customer feedback message (1-2 sentences) that sounds like it came from:
- A support ticket if priority is 4-5
- An NPS survey if sentiment is positive
- An app store review if the tone is casual

Make it sound natural and authentic. Do not include any preamble, just the feedback text."""
        
        try:
            response = requests.post(
                self.api_url,
                headers={
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json'
                },
                json={
                    'model': self.model,
                    'messages': [{'role': 'user', 'content': prompt}],
                    'temperature': 0.8,
                    'max_tokens': 150
                },
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content'].strip()
            else:
                logger.warning(f"LLM API error ({response.status_code}), falling back to template")
                return self._generate_template_summary(product_area, topic, sentiment, priority)
                
        except Exception as e:
            logger.warning(f"LLM generation failed: {e}, falling back to template")
            return self._generate_template_summary(product_area, topic, sentiment, priority)
    
    def _generate_template_summary(self, product_area: str, topic: str, sentiment: str, priority: int) -> str:
        """Generate summary from templates when LLM is unavailable."""
        templates = {
            'positive': [
                f"Really appreciate the improvements to {topic}, makes {product_area} much easier",
                f"Love the new {topic} feature, exactly what we needed for {product_area}",
                f"Great job on {topic}, the {product_area} experience is much better now"
            ],
            'neutral': [
                f"The {topic} functionality works but could use some improvements in {product_area}",
                f"Would like to see better {topic} options for {product_area}",
                f"{topic.title()} is okay but not great for our {product_area} needs"
            ],
            'negative': [
                f"Critical issue with {topic} blocking our {product_area} workflow",
                f"Unable to use {topic} feature, making {product_area} very difficult",
                f"Serious problems with {topic} need immediate attention in {product_area}"
            ]
        }
        
        template = random.choice(templates.get(sentiment, templates['neutral']))
        
        if priority >= 4:
            template = f"URGENT: {template}"
        
        return template
    
    def generate_feedback_batch(
        self,
        count: int,
        distribution: str = 'realistic',
        days_back: int = 30
    ) -> List[Dict]:
        """Generate batch of feedback entries.
        
        Args:
            count: Number of feedback entries to generate
            distribution: Distribution strategy ('realistic', 'balanced', 'negative-heavy')
            days_back: Spread feedback over this many days
            
        Returns:
            List of feedback dictionaries ready for insertion
        """
        logger.info(f"Generating {count} feedback entries with {distribution} distribution")
        
        feedback_batch = []
        
        for i in range(count):
            # Apply distribution strategy
            if distribution == 'realistic':
                # Realistic distribution: more negative in support channels, mixed elsewhere
                source = random.choice(self.SOURCES)
                if source in ['zendesk', 'chat']:
                    sentiment = random.choices(
                        self.SENTIMENTS,
                        weights=[0.2, 0.3, 0.5]  # 50% negative
                    )[0]
                    priority = random.choices(self.PRIORITIES, weights=[0.1, 0.2, 0.3, 0.25, 0.15])[0]
                else:
                    sentiment = random.choices(
                        self.SENTIMENTS,
                        weights=[0.4, 0.35, 0.25]  # More positive
                    )[0]
                    priority = random.choices(self.PRIORITIES, weights=[0.2, 0.3, 0.3, 0.15, 0.05])[0]
            elif distribution == 'balanced':
                source = random.choice(self.SOURCES)
                sentiment = random.choice(self.SENTIMENTS)
                priority = random.choice(self.PRIORITIES)
            else:  # negative-heavy
                source = random.choice(['zendesk', 'chat', 'email'])
                sentiment = random.choices(self.SENTIMENTS, weights=[0.1, 0.2, 0.7])[0]
                priority = random.choices(self.PRIORITIES, weights=[0.05, 0.1, 0.2, 0.35, 0.3])[0]
            
            # Select product area and corresponding topic
            product_area = random.choice(self.PRODUCT_AREAS)
            topic = random.choice(self.TOPICS_BY_AREA[product_area])
            
            # Other attributes
            region = random.choice(self.REGIONS)
            customer_tier = random.choices(
                self.CUSTOMER_TIERS,
                weights=[0.5, 0.35, 0.15]  # More free users
            )[0]
            
            # Generate timestamp
            days_ago = random.uniform(0, days_back)
            created_at = datetime.now() - timedelta(days=days_ago)
            
            # Generate summary
            summary = self.generate_summary(product_area, topic, sentiment, priority)
            
            feedback_entry = {
                'id': f'fb-seed-{uuid4().hex[:8]}',
                'created_at': created_at,
                'source': source,
                'product_area': product_area,
                'sentiment': sentiment,
                'priority': priority,
                'topic': topic,
                'region': region,
                'customer_tier': customer_tier,
                'summary': summary
            }
            
            feedback_batch.append(feedback_entry)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Generated {i + 1}/{count} entries...")
        
        logger.info(f"✓ Generated {count} feedback entries")
        return feedback_batch
    
    def insert_feedback(self, feedback_batch: List[Dict]) -> int:
        """Insert feedback entries into database.
        
        Args:
            feedback_batch: List of feedback dictionaries
            
        Returns:
            Number of rows inserted
        """
        if not feedback_batch:
            logger.warning("No feedback to insert")
            return 0
        
        logger.info(f"Inserting {len(feedback_batch)} feedback entries into database...")
        
        # Import here to avoid circular dependency
        import psycopg2
        from src.config import get_config
        
        config = get_config()
        conn = psycopg2.connect(config.database_url)
        
        try:
            # Prepare insert query
            insert_query = """
                INSERT INTO feedback_enriched (
                    id, created_at, source, product_area, sentiment,
                    priority, topic, region, customer_tier, summary
                ) VALUES (
                    %(id)s, %(created_at)s, %(source)s, %(product_area)s, %(sentiment)s,
                    %(priority)s, %(topic)s, %(region)s, %(customer_tier)s, %(summary)s
                )
                ON CONFLICT (id) DO NOTHING
            """
            
            cursor = conn.cursor()
            inserted = 0
            
            for feedback in feedback_batch:
                try:
                    cursor.execute(insert_query, feedback)
                    if cursor.rowcount > 0:
                        inserted += 1
                except Exception as e:
                    logger.error(f"Failed to insert {feedback['id']}: {e}")
                    continue
            
            conn.commit()
            cursor.close()
            
        finally:
            conn.close()
        
        logger.info(f"✓ Inserted {inserted} new feedback entries")
        return inserted
    
    def seed(
        self,
        count: int,
        distribution: str = 'realistic',
        days_back: int = 30
    ) -> Dict:
        """Main seeding function - generate and insert feedback.
        
        Args:
            count: Number of entries to generate
            distribution: Distribution strategy
            days_back: Days to spread data over
            
        Returns:
            Summary statistics
        """
        logger.info("=" * 60)
        logger.info("Feedback Data Seeder")
        logger.info("=" * 60)
        logger.info(f"Target count: {count}")
        logger.info(f"Distribution: {distribution}")
        logger.info(f"Time range: {days_back} days")
        logger.info(f"LLM provider: {self.llm_provider}")
        logger.info("=" * 60)
        
        # Generate feedback
        feedback_batch = self.generate_feedback_batch(count, distribution, days_back)
        
        # Insert into database
        inserted = self.insert_feedback(feedback_batch)
        
        # Calculate statistics
        stats = {
            'generated': len(feedback_batch),
            'inserted': inserted,
            'skipped': len(feedback_batch) - inserted,
            'by_sentiment': {},
            'by_product_area': {},
            'by_priority': {}
        }
        
        for fb in feedback_batch:
            stats['by_sentiment'][fb['sentiment']] = stats['by_sentiment'].get(fb['sentiment'], 0) + 1
            stats['by_product_area'][fb['product_area']] = stats['by_product_area'].get(fb['product_area'], 0) + 1
            stats['by_priority'][fb['priority']] = stats['by_priority'].get(fb['priority'], 0) + 1
        
        logger.info("=" * 60)
        logger.info("Seeding Complete!")
        logger.info("=" * 60)
        logger.info(f"Generated: {stats['generated']} entries")
        logger.info(f"Inserted: {stats['inserted']} entries")
        logger.info(f"Skipped: {stats['skipped']} (duplicates)")
        logger.info("")
        logger.info("Distribution by sentiment:")
        for sentiment, count in sorted(stats['by_sentiment'].items()):
            pct = (count / stats['generated'] * 100)
            logger.info(f"  {sentiment:10s}: {count:4d} ({pct:5.1f}%)")
        logger.info("")
        logger.info("Distribution by product area:")
        for area, count in sorted(stats['by_product_area'].items(), key=lambda x: -x[1]):
            pct = (count / stats['generated'] * 100)
            logger.info(f"  {area:15s}: {count:4d} ({pct:5.1f}%)")
        logger.info("=" * 60)
        
        return stats


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Seed realistic feedback data using AI-generated summaries'
    )
    parser.add_argument(
        '--count', '-c',
        type=int,
        default=50,
        help='Number of feedback entries to generate (default: 50)'
    )
    parser.add_argument(
        '--distribution', '-d',
        choices=['realistic', 'balanced', 'negative-heavy'],
        default='realistic',
        help='Distribution strategy (default: realistic)'
    )
    parser.add_argument(
        '--days-back',
        type=int,
        default=30,
        help='Spread feedback over this many days (default: 30)'
    )
    parser.add_argument(
        '--provider',
        choices=['groq', 'openai', 'none'],
        default='groq',
        help='LLM provider for generating summaries (default: groq)'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        help='API key for LLM provider (uses env var if not provided)'
    )
    
    args = parser.parse_args()
    
    try:
        seeder = FeedbackSeeder(
            llm_provider=args.provider,
            api_key=args.api_key
        )
        
        stats = seeder.seed(
            count=args.count,
            distribution=args.distribution,
            days_back=args.days_back
        )
        
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Seeding failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
