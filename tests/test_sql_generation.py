import os
import pytest
import asyncio
from constants import LLMWARE_LLM_MODEL
from src.main import QueryTranslator, setup_logging
from src.interfaces import LLMWareAPIClient
from src.api_clients import ChromaVectorAPIClient
import logging
import yaml
from pathlib import Path
from unittest.mock import Mock
import pytest_asyncio  # type: ignore
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@pytest_asyncio.fixture 
async def translator():
    """Initialize QueryTranslator with test configuration"""
    # Load config
    config_path = Path("src/config/schema.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    model_name = LLMWARE_LLM_MODEL
    # Initialize clients
    vector_client = ChromaVectorAPIClient(collection_name="test_banking_terms")
    llm_client = LLMWareAPIClient(model_name=model_name)
    
    # Create translator
    translator = QueryTranslator(
        config_path=config_path,
        vector_api_client=vector_client,
        llm_api_client=llm_client
    )
    
    return translator

# Basic SQL Query Tests
@pytest.mark.asyncio
async def test_simple_select(translator):
    """Test simple SELECT queries"""
    translator = translator
    test_cases = [
        (
            "Show me all customer credit scores",
            "SELECT credit_score FROM customer_credit"
        ),
        (
            "List all payment amounts",
            "SELECT payment_amount FROM payment_history"
        ),
        (
            "What are the risk ratings for all customers?",
            "SELECT customer_id, risk_rating FROM customer_credit"
        )
    ]
    
    for query, expected_sql in test_cases:
        sql, _ = await translator.translate_to_sql(query)
        assert sql.strip().lower() == expected_sql.strip().lower()

@pytest.mark.asyncio
async def test_where_clauses(translator):
    """Test WHERE clause variations"""
    translator = translator
    test_cases = [
        (
            "Find customers with credit score above 700",
            "SELECT customer_id, credit_score FROM customer_credit WHERE credit_score > 700"
        ),
        (
            "Show payments less than $1000",
            "SELECT * FROM payment_history WHERE payment_amount < 1000"
        ),
        (
            "List customers with high risk rating",
            "SELECT customer_id, credit_score FROM customer_credit WHERE risk_rating = 'HIGH'"
        )
    ]
    
    for query, expected_sql in test_cases:
        sql, _ = await translator.translate_to_sql(query)
        assert sql.strip().lower() == expected_sql.strip().lower()

@pytest.mark.asyncio
async def test_joins(translator):
    """Test JOIN operations"""
    translator = translator
    test_cases = [
        (
            "Show credit scores and payment history for all customers",
            """
            SELECT cc.customer_id, cc.credit_score, ph.payment_amount, ph.payment_date 
            FROM customer_credit cc 
            JOIN payment_history ph ON cc.customer_id = ph.customer_id
            """
        ),
        (
            "Find all payment records for high risk customers",
            """
            SELECT ph.* 
            FROM payment_history ph 
            JOIN customer_credit cc ON ph.customer_id = cc.customer_id 
            WHERE cc.risk_rating = 'HIGH'
            """
        )
    ]
    
    for query, expected_sql in test_cases:
        sql, _ = await translator.translate_to_sql(query)
        assert sql.strip().lower() == expected_sql.strip().lower()

@pytest.mark.asyncio
async def test_aggregations(translator):
    """Test GROUP BY and aggregate functions"""
    translator = translator
    test_cases = [
        (
            "What is the average credit score by risk rating?",
            """
            SELECT risk_rating, AVG(credit_score) as avg_credit_score 
            FROM customer_credit 
            GROUP BY risk_rating
            """
        ),
        (
            "Count number of payments by status",
            """
            SELECT payment_status, COUNT(*) as payment_count 
            FROM payment_history 
            GROUP BY payment_status
            """
        )
    ]
    
    for query, expected_sql in test_cases:
        sql, _ = await translator.translate_to_sql(query)
        assert sql.strip().lower() == expected_sql.strip().lower()

# Natural Language Understanding Tests
@pytest.mark.asyncio
async def test_synonyms(translator):
    """Test handling of synonyms"""
    translator = translator
    synonyms = [
        (
            "What is the credit worthiness of customers?",
            "What is the credit score of customers?",
            "SELECT customer_id, credit_score FROM customer_credit"
        ),
        (
            "Show me late payments",
            "Show me overdue payments",
            "SELECT * FROM payment_history WHERE payment_status = 'Late'"
        )
    ]
    
    for query1, query2, expected_sql in synonyms:
        sql1, _ = await translator.translate_to_sql(query1)
        sql2, _ = await translator.translate_to_sql(query2)
        assert sql1.strip().lower() == sql2.strip().lower() == expected_sql.strip().lower()

@pytest.mark.asyncio
async def test_negations(translator):
    """Test handling of negations"""
    translator = translator
    test_cases = [
        (
            "Show customers who don't have late payments",
            """
            SELECT DISTINCT cc.* 
            FROM customer_credit cc 
            LEFT JOIN payment_history ph ON cc.customer_id = ph.customer_id 
            WHERE ph.payment_status != 'Late' OR ph.payment_status IS NULL
            """
        ),
        (
            "Find customers except those with high risk",
            "SELECT * FROM customer_credit WHERE risk_rating != 'HIGH'"
        )
    ]
    
    for query, expected_sql in test_cases:
        sql, _ = await translator.translate_to_sql(query)
        assert sql.strip().lower() == expected_sql.strip().lower()

# Edge Cases Tests
@pytest.mark.asyncio
async def test_edge_cases(translator):
    """Test various edge cases"""
    
    translator = translator
    # Empty input
    with pytest.raises(ValueError):
        await translator.translate_to_sql("")
    
    # Non-existent table
    with pytest.raises(Exception):
        await translator.translate_to_sql("Show me the employee salaries")
    
    # Misspelled words should still work
    sql, _ = await translator.translate_to_sql("Show me credit scors")
    assert "SELECT" in sql.upper() and "credit_score" in sql.lower()
    
    # Very long query
    long_query = """
    Can you please show me a detailed analysis of all customers who have a credit score 
    above 700 and have made at least 3 payments on time but also had exactly one late 
    payment in the past year, and sort them by their risk rating in descending order 
    while also calculating their average payment amount?
    """
    sql, _ = await translator.translate_to_sql(long_query)
    assert "SELECT" in sql.upper() and "JOIN" in sql.upper()

@pytest.mark.asyncio
async def test_complex_conditions(translator):
    """Test complex conditional queries"""
    translator = translator
    test_cases = [
        (
            "Find customers with credit score between 600 and 750",
            """
            SELECT * FROM customer_credit 
            WHERE credit_score BETWEEN 600 AND 750
            """
        ),
        (
            "Show payments greater than or equal to $5000",
            """
            SELECT * FROM payment_history 
            WHERE payment_amount >= 5000
            """
        ),
        (
            "List customers with at least 3 late payments",
            """
            SELECT cc.*, COUNT(ph.payment_id) as late_payments 
            FROM customer_credit cc 
            JOIN payment_history ph ON cc.customer_id = ph.customer_id 
            WHERE ph.payment_status = 'Late' 
            GROUP BY cc.customer_id 
            HAVING COUNT(ph.payment_id) >= 3
            """
        )
    ]
    
    for query, expected_sql in test_cases:
        sql, _ = await translator.translate_to_sql(query)
        assert sql.strip().lower() == expected_sql.strip().lower()

if __name__ == "__main__":
    pytest.main(["-v", "test_sql_generation.py"]) 