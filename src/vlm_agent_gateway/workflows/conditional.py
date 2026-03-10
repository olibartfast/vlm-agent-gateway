"""
Conditional workflow: Router classifies input, specialist handles request.

Input ──► [Router Agent] ──► category
                                │
            ┌───────────────────┼──────────────────┐
        [Specialist-A]   [Specialist-B]   [Specialist-C]
         (if cat=A)       (if cat=B)       (if cat=C)
                                │
                             final output
"""


from vlm_agent_gateway.models import Agent
from vlm_agent_gateway.providers import run_agent


def run_conditional(
    router_agent: Agent,
    specialist_agents: list[Agent],
    categories: list[str],
    prompt: str,
    image_paths: list[str],
    detail: str,
    max_tokens: int,
    resize: bool,
    target_size: tuple[int, int],
) -> dict:
    """
    Route input to a specialist based on router classification.

    A lightweight router agent first classifies the input into one of the known
    categories. The matching specialist agent then handles the full request.
    If the router's answer doesn't match any category, the first specialist is
    used as the default fallback.
    """
    if len(specialist_agents) != len(categories):
        raise ValueError("specialist_agents and categories must have the same length")

    # Step 1 — Router classifies input
    router_prompt = (
        f"Analyze the following image and/or prompt and classify it into exactly one of these "
        f"categories: {categories}.\n"
        "Reply with ONLY the category name, nothing else.\n\n"
        f"Prompt: {prompt}"
    )
    router_result = run_agent(
        router_agent, router_prompt, image_paths, detail, max_tokens, resize, target_size
    )
    if router_result.error:
        raise RuntimeError(f"Router agent failed: {router_result.error}")

    route = router_result.content.strip().lower()

    # Step 2 — Find matching specialist (case-insensitive, fallback to first)
    specialist = None
    matched_category = None
    for cat, spec in zip(categories, specialist_agents, strict=True):
        if cat.lower() in route or route in cat.lower():
            specialist = spec
            matched_category = cat
            break

    if specialist is None:
        matched_category = categories[0]
        specialist = specialist_agents[0]

    # Step 3 — Specialist handles the actual request
    spec_result = run_agent(specialist, prompt, image_paths, detail, max_tokens, resize, target_size)
    if spec_result.error:
        raise RuntimeError(f"Specialist agent failed: {spec_result.error}")

    return {
        "workflow": "conditional",
        "router_model": router_agent.model,
        "router_raw_decision": router_result.content.strip(),
        "matched_category": matched_category,
        "specialist_model": specialist.model,
        "specialist_provider": specialist.provider,
        "router_latency_ms": round(router_result.latency_ms, 1),
        "specialist_latency_ms": round(spec_result.latency_ms, 1),
        "content": spec_result.content,
    }
