"""Tests for data models."""

from vlm_agent_gateway.models import Agent, AlertEvent, Tool


def test_agent_creation():
    """Test Agent dataclass creation."""
    agent = Agent(
        model="gpt-4o",
        endpoint="https://api.openai.com/v1/chat/completions",
        api_key="sk-test",
        provider="openai",
    )
    assert agent.model == "gpt-4o"
    assert agent.provider == "openai"
    assert len(agent.agent_id) == 8  # UUID prefix


def test_agent_result_success(mock_successful_result):
    """Test AgentResult success property."""
    assert mock_successful_result.success is True
    assert mock_successful_result.content == "Test response content"


def test_agent_result_failure(mock_failed_result):
    """Test AgentResult failure property."""
    assert mock_failed_result.success is False
    assert mock_failed_result.error == "Connection timeout"


def test_tool_creation():
    """Test Tool dataclass creation."""
    tool = Tool(
        name="test_tool",
        description="A test tool",
        parameters="{}",
        fn=lambda *args, **kwargs: "result",
    )
    assert tool.name == "test_tool"
    assert callable(tool.fn)


def test_alert_event_creation():
    """Test AlertEvent dataclass creation."""
    event = AlertEvent(
        timestamp="2024-01-01T00:00:00Z",
        alert=True,
        summary="Test alert",
        confidence="HIGH",
        recommended_action="Check camera",
        thought="I see something",
        latency_ms=150.0,
    )
    assert event.alert is True
    assert event.confidence == "HIGH"
