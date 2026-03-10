"""
Parallel workflow: All agents process input simultaneously.

Input ──► [Agent-1] ─┐
Input ──► [Agent-2] ─┤──► [Aggregator] ──► final
Input ──► [Agent-3] ─┘
"""

import concurrent.futures

from vlm_agent_gateway.models import Agent
from vlm_agent_gateway.providers import run_agent


def run_parallel(
    agents: list[Agent],
    prompt: str,
    image_paths: list[str],
    detail: str,
    max_tokens: int,
    resize: bool,
    target_size: tuple[int, int],
) -> dict:
    """
    Execute all agents concurrently and select the best response.

    Results are collected once all branches complete.
    The best response is selected by content length.
    """
    if len(agents) < 2:
        raise ValueError("parallel workflow requires at least 2 agents")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(
                run_agent, a, prompt, image_paths, detail, max_tokens, resize, target_size
            ): a
            for a in agents
        }
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    successful = [r for r in results if r.success]
    if not successful:
        raise RuntimeError(f"All parallel agents failed: {[r.error for r in results]}")

    best = max(successful, key=lambda r: len(r.content))

    return {
        "workflow": "parallel",
        "selected_agent_id": best.agent_id,
        "selected_model": best.model,
        "selected_provider": best.provider,
        "content": best.content,
        "agents": [
            {
                "agent_id": r.agent_id,
                "model": r.model,
                "provider": r.provider,
                "latency_ms": round(r.latency_ms, 1),
                "success": r.success,
                "error": r.error,
            }
            for r in results
        ],
    }
