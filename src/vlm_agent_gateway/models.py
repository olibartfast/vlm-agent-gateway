"""
Data models (dataclasses) for the VLM Agent Gateway.
"""

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class Agent:
    """Configuration for a single VLM agent."""
    model: str
    endpoint: str
    api_key: str
    provider: str = "openai"
    agent_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


@dataclass
class AgentResult:
    """Result from executing a single agent call."""
    agent_id: str
    model: str
    provider: str
    content: str
    raw_response: dict
    latency_ms: float
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.error is None


@dataclass
class Tool:
    """Definition of a ReAct tool."""
    name: str
    description: str
    parameters: str  # JSON schema hint shown to the agent
    fn: Callable  # fn(agent, image_paths, detail, max_tokens, resize, target_size, **kwargs) -> str


@dataclass
class AlertEvent:
    """Result from a video monitoring cycle."""
    timestamp: str
    alert: bool
    summary: str
    confidence: str
    recommended_action: str
    thought: str
    latency_ms: float
