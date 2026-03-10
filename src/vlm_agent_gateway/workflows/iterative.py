"""
Iterative workflow: Agent loops with evaluation until convergence.

prompt ──► [Agent] ──► output-1
                          │
                    [Evaluator] ── not converged ──► prompt + output-1 ──► [Agent] ──► output-2
                          │                                                        │
                      converged ◄──────────────────────────────────── [Evaluator] ─┘
                          │
                       final output
"""


from vlm_agent_gateway.models import Agent
from vlm_agent_gateway.providers import run_agent


def run_iterative(
    agent: Agent,
    prompt: str,
    image_paths: list[str],
    detail: str,
    max_tokens: int,
    resize: bool,
    target_size: tuple[int, int],
    evaluator_agent: Agent | None = None,
    max_iterations: int = 3,
) -> dict:
    """
    Execute agent in a refinement loop until convergence.

    After each iteration the evaluator (or a simple heuristic) judges whether
    the output has converged. If not, the output is fed back as accumulated
    context so the next iteration can refine it.
    The loop stops when converged or max_iterations is reached.
    """
    iterations = []
    current_prompt = prompt

    for i in range(max_iterations):
        result = run_agent(agent, current_prompt, image_paths, detail, max_tokens, resize, target_size)
        if result.error:
            raise RuntimeError(f"Iteration {i + 1} failed: {result.error}")

        # Evaluate convergence
        if evaluator_agent:
            eval_prompt = (
                f"Rate the following response from 1 to 10 for completeness and accuracy.\n"
                f"Original question: {prompt}\n\n"
                f"Response: {result.content}\n\n"
                "Reply with ONLY a single integer between 1 and 10."
            )
            eval_result = run_agent(
                evaluator_agent, eval_prompt, [], detail, max_tokens, resize, target_size
            )
            try:
                score = int(
                    "".join(filter(str.isdigit, eval_result.content.strip()))[:2] or "0"
                )
            except ValueError:
                score = 0
            converged = score >= 7
        else:
            # Simple heuristic: converged when response is substantive
            converged = len(result.content.strip()) >= 100

        iterations.append({
            "iteration": i + 1,
            "agent_id": result.agent_id,
            "model": result.model,
            "latency_ms": round(result.latency_ms, 1),
            "content": result.content,
            "converged": converged,
        })

        if converged:
            break

        # Feed accumulated outputs back as context for next iteration
        history_block = "\n\n".join(
            f"[Iteration {it['iteration']}]\n{it['content']}" for it in iterations
        )
        current_prompt = (
            f"{prompt}\n\n"
            f"Previous attempts:\n{history_block}\n\n"
            "Please refine and improve your response based on the above attempts, "
            "addressing any gaps or inaccuracies."
        )

    final = iterations[-1]
    return {
        "workflow": "iterative",
        "total_iterations": len(iterations),
        "converged": final["converged"],
        "stop_reason": "converged" if final["converged"] else "max_iterations_reached",
        "content": final["content"],
        "iterations": iterations,
    }
