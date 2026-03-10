"""
Sequential workflow: Agents execute one after another with accumulated context.

Input ──► [Agent-1] ──► output-1
                          │
          prompt + output-1 ──► [Agent-2] ──► output-2
                                                │
                   prompt + output-1 + output-2 ──► [Agent-3] ──► final
"""


from vlm_agent_gateway.models import Agent
from vlm_agent_gateway.providers import run_agent


def run_sequential(
    agents: list[Agent],
    prompt: str,
    image_paths: list[str],
    detail: str,
    max_tokens: int,
    resize: bool,
    target_size: tuple[int, int],
) -> dict:
    """
    Execute agents sequentially with accumulated context.

    Each agent receives the original prompt plus all prior agents' outputs,
    allowing each stage to build on the previous analysis.
    """
    if not agents:
        raise ValueError("sequential workflow requires at least 1 agent")

    stages = []
    for i, agent in enumerate(agents):
        if stages:
            context_block = "\n\n".join(
                f"[Stage {s['stage']} — {s['model']}]\n{s['content']}" for s in stages
            )
            current_prompt = (
                f"{prompt}\n\n"
                f"Prior stage outputs:\n{context_block}\n\n"
                "Building on the above, provide your specialized analysis."
            )
        else:
            current_prompt = prompt

        result = run_agent(agent, current_prompt, image_paths, detail, max_tokens, resize, target_size)
        if result.error:
            raise RuntimeError(f"Stage {i + 1} failed: {result.error}")

        stages.append({
            "stage": i + 1,
            "agent_id": result.agent_id,
            "model": result.model,
            "provider": result.provider,
            "latency_ms": round(result.latency_ms, 1),
            "content": result.content,
        })

    return {
        "workflow": "sequential",
        "stages": stages,
        "content": stages[-1]["content"],
        "total_stages": len(stages),
    }
