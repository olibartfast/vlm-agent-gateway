"""
ReAct (Reasoning + Acting) workflow.

prompt ──► [Agent: Thought + Action] ──► tool call
                       ▲                       │
                       └── Observation ◄───────┘
                       (repeated up to max_steps)
                       ──► Final Answer
"""


from vlm_agent_gateway.config import REACT_SYSTEM_PROMPT
from vlm_agent_gateway.models import Agent, Tool
from vlm_agent_gateway.providers import run_agent
from vlm_agent_gateway.tools import BUILTIN_TOOLS, _parse_react_step


def run_react(
    agent: Agent,
    prompt: str,
    image_paths: list[str],
    detail: str,
    max_tokens: int,
    resize: bool,
    target_size: tuple[int, int],
    enabled_tools: list[str] | None = None,
    max_steps: int = 5,
) -> dict:
    """
    Execute ReAct reasoning loop with tool calling.

    The agent interleaves Thought / Action / Observation steps.
    Each Action calls a registered tool. The Observation is appended back
    into the conversation so the agent can reason about it and decide the
    next action. The loop ends when the agent emits "Final Answer:" or the
    step budget is exhausted.
    """
    tools: dict[str, Tool] = (
        {k: v for k, v in BUILTIN_TOOLS.items() if k in enabled_tools}
        if enabled_tools
        else BUILTIN_TOOLS
    )
    if not tools:
        raise ValueError(f"No valid tools enabled. Available: {list(BUILTIN_TOOLS.keys())}")

    tool_descriptions = "\n".join(
        f"  {t.name}: {t.description} | parameters: {t.parameters}" for t in tools.values()
    )
    system_prompt = REACT_SYSTEM_PROMPT.format(tool_descriptions=tool_descriptions)
    conversation = f"{system_prompt}\n\nQuestion: {prompt}\n"

    steps = []
    final_answer = None

    for step_num in range(1, max_steps + 1):
        result = run_agent(agent, conversation, image_paths, detail, max_tokens, resize, target_size)
        if result.error:
            raise RuntimeError(f"ReAct step {step_num} agent call failed: {result.error}")

        model_output = result.content
        thought, action, action_input, final_answer = _parse_react_step(model_output)

        if final_answer is not None:
            steps.append({"step": step_num, "thought": "(final)", "final_answer": final_answer})
            break

        # Execute the tool
        if action not in tools:
            observation = (
                f"Unknown tool '{action}'. Available tools: {list(tools.keys())}. "
                "Please choose a valid tool."
            )
        else:
            try:
                observation = tools[action].fn(
                    agent, image_paths, detail, max_tokens, resize, target_size, **action_input
                )
            except Exception as exc:
                observation = f"Tool '{action}' raised an error: {exc}"

        steps.append({
            "step": step_num,
            "thought": thought,
            "action": action,
            "action_input": action_input,
            "observation": observation,
            "latency_ms": round(result.latency_ms, 1),
        })

        # Append model output + observation to conversation for next step
        conversation += f"\n{model_output.rstrip()}\nObservation: {observation}\n"
    else:
        # max_steps exhausted without Final Answer
        final_answer = steps[-1].get("observation", "") if steps else "(no answer produced)"

    return {
        "workflow": "react",
        "model": agent.model,
        "provider": agent.provider,
        "total_steps": len(steps),
        "stop_reason": "final_answer" if steps and "final_answer" in steps[-1] else "max_steps_reached",
        "content": final_answer,
        "steps": steps,
    }
