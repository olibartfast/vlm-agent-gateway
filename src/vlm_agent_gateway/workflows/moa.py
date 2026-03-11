"""
Mixture-of-Agents (MoA) workflow: Multiple proposers, one aggregator.

Input ──► [Proposer-1] ─┐
Input ──► [Proposer-2] ─┤──► candidates ──► [Aggregator] ──► final
Input ──► [Proposer-3] ─┘
"""

import concurrent.futures

from vlm_agent_gateway.models import Agent
from vlm_agent_gateway.providers import run_agent


def run_moa(
    proposer_agents: list[Agent],
    aggregator_agent: Agent,
    prompt: str,
    image_paths: list[str],
    detail: str,
    max_tokens: int,
    resize: bool,
    target_size: tuple[int, int],
) -> dict:
    """
    Execute multiple proposers in parallel, then aggregate results.

    Multiple proposer agents process the input in parallel. Their outputs are
    all passed to an aggregator agent that synthesizes a single final answer.
    """
    if len(proposer_agents) < 2:
        raise ValueError("moa workflow requires at least 2 proposer agents")

    # Step 1 — Parallel proposers
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(run_agent, a, prompt, image_paths, detail, max_tokens, resize, target_size)
            for a in proposer_agents
        ]
        proposer_results = [f.result() for f in concurrent.futures.as_completed(futures)]

    successful = [r for r in proposer_results if r.success]
    if not successful:
        raise RuntimeError(f"All proposer agents failed: {[r.error for r in proposer_results]}")

    # Step 2 — Aggregator synthesizes all candidates
    candidates_block = "\n\n".join(
        f"[Candidate {i + 1} — {r.model} / {r.provider}]\n{r.content}" for i, r in enumerate(successful)
    )
    aggregator_prompt = (
        f"You are an impartial synthesizer. Below are {len(successful)} candidate answers "
        f'to the question: "{prompt}"\n\n'
        f"{candidates_block}\n\n"
        "Compare the candidates, extract consensus points, resolve conflicts, "
        "and produce one final, comprehensive best answer."
    )
    agg_result = run_agent(
        aggregator_agent,
        aggregator_prompt,
        image_paths,
        detail,
        max_tokens,
        resize,
        target_size,
    )
    if agg_result.error:
        raise RuntimeError(f"Aggregator agent failed: {agg_result.error}")

    return {
        "workflow": "moa",
        "aggregator_model": aggregator_agent.model,
        "aggregator_provider": aggregator_agent.provider,
        "aggregator_latency_ms": round(agg_result.latency_ms, 1),
        "content": agg_result.content,
        "proposers": [
            {
                "agent_id": r.agent_id,
                "model": r.model,
                "provider": r.provider,
                "latency_ms": round(r.latency_ms, 1),
                "success": r.success,
                "error": r.error,
            }
            for r in proposer_results
        ],
        "raw_aggregator_response": agg_result.raw_response,
    }
