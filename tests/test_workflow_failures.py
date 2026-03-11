"""Tests for workflow failure handling."""

import pytest

from vlm_agent_gateway.models import Agent, AgentResult
from vlm_agent_gateway.workflows.iterative import run_iterative
from vlm_agent_gateway.workflows.moa import run_moa


def _agent(name: str) -> Agent:
    return Agent(
        model=name,
        endpoint="http://localhost:8000/v1/chat/completions",
        api_key="EMPTY",
        provider="openai",
    )


def test_iterative_raises_when_evaluator_fails(monkeypatch):
    """Evaluator transport failures should stop the workflow explicitly."""
    agent = _agent("generator")
    evaluator = _agent("evaluator")
    results = iter(
        [
            AgentResult(
                agent_id=agent.agent_id,
                model=agent.model,
                provider=agent.provider,
                content="candidate answer",
                raw_response={},
                latency_ms=10.0,
            ),
            AgentResult(
                agent_id=evaluator.agent_id,
                model=evaluator.model,
                provider=evaluator.provider,
                content="",
                raw_response={},
                latency_ms=5.0,
                error="timeout",
            ),
        ]
    )

    monkeypatch.setattr("vlm_agent_gateway.workflows.iterative.run_agent", lambda *args, **kwargs: next(results))

    with pytest.raises(RuntimeError, match="Evaluator failed during iteration 1: timeout"):
        run_iterative(
            agent,
            prompt="Describe the image",
            image_paths=["image.jpg"],
            detail="low",
            max_tokens=100,
            resize=False,
            target_size=(512, 512),
            evaluator_agent=evaluator,
        )


def test_moa_raises_when_aggregator_fails(monkeypatch):
    """Aggregator failures should not be hidden behind a proposer fallback."""
    proposer_a = _agent("proposer-a")
    proposer_b = _agent("proposer-b")
    aggregator = _agent("aggregator")
    results = iter(
        [
            AgentResult(
                agent_id=proposer_a.agent_id,
                model=proposer_a.model,
                provider=proposer_a.provider,
                content="answer a",
                raw_response={},
                latency_ms=10.0,
            ),
            AgentResult(
                agent_id=proposer_b.agent_id,
                model=proposer_b.model,
                provider=proposer_b.provider,
                content="answer b",
                raw_response={},
                latency_ms=12.0,
            ),
            AgentResult(
                agent_id=aggregator.agent_id,
                model=aggregator.model,
                provider=aggregator.provider,
                content="",
                raw_response={},
                latency_ms=8.0,
                error="upstream 502",
            ),
        ]
    )

    monkeypatch.setattr("vlm_agent_gateway.workflows.moa.run_agent", lambda *args, **kwargs: next(results))

    with pytest.raises(RuntimeError, match="Aggregator agent failed: upstream 502"):
        run_moa(
            proposer_agents=[proposer_a, proposer_b],
            aggregator_agent=aggregator,
            prompt="Compare these frames",
            image_paths=["frame1.jpg"],
            detail="low",
            max_tokens=100,
            resize=False,
            target_size=(512, 512),
        )
