import pytest
from src.app_2 import SemanticAnalyzer, DomainContext

@pytest.fixture
def analyzer():
    config = {...}  # Test configuration
    return SemanticAnalyzer(config)

def test_extract_base_intent():
    analyzer = analyzer()
    query = "What is the average credit score for high-risk customers?"
    intent = analyzer._extract_base_intent(query)
    assert intent.action_type == "AGGREGATE"
    assert "credit_score" in intent.main_entities 