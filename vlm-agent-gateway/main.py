"""
VLM Agent Gateway
Multi-provider vision inference with sequential, parallel, conditional,
iterative, Mixture-of-Agents (MoA), and ReAct (Reasoning + Acting) workflow support.
"""

import os
import re
import requests
import base64
import argparse
import json
import uuid
import time
import concurrent.futures
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable
from PIL import Image, ImageOps
import io

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROVIDER_ENV_MAP = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "together": "TOGETHER_API_KEY",
    "azure": "AZURE_OPENAI_API_KEY",
    "groq": "GROQ_API_KEY",
    "mistral": "MISTRAL_API_KEY",
}

DEFAULT_ENDPOINT = "https://api.openai.com/v1/chat/completions"

WORKFLOW_CHOICES = ["sequential", "parallel", "conditional", "iterative", "moa", "react"]

# ReAct prompt template
REACT_SYSTEM_PROMPT = """You are a vision analysis agent. Solve the task step by step using the available tools.

Available tools:
{tool_descriptions}

Use this EXACT format for every step:
Thought: <your reasoning about what to do next>
Action: <tool_name>
Action Input: <JSON object with the tool's parameters>
Observation: <result will be filled in by the system>

Once you have enough information, end with:
Thought: I now have enough information to answer.
Final Answer: <your complete answer>

Rules:
- Only call one tool per step.
- Always wait for the Observation before continuing.
- Never fabricate an Observation — the system provides it.
- Stop as soon as you can write Final Answer."""

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Agent:
    model: str
    endpoint: str
    api_key: str
    provider: str = "openai"
    agent_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


@dataclass
class AgentResult:
    agent_id: str
    model: str
    provider: str
    content: str
    raw_response: dict
    latency_ms: float
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def is_url(path: str) -> bool:
    return path.startswith("http://") or path.startswith("https://")


def resize_with_padding(image, target_size=(512, 512)):
    if isinstance(image, str) and not is_url(image):
        img = Image.open(image)
    elif isinstance(image, bytes):
        img = Image.open(io.BytesIO(image))
    else:
        raise ValueError("Unsupported image input")
    if img.mode != "RGB":
        img = img.convert("RGB")
    img_ratio = img.width / img.height
    target_ratio = target_size[0] / target_size[1]
    if img_ratio > target_ratio:
        new_width = target_size[0]
        new_height = int(target_size[0] / img_ratio)
    else:
        new_height = target_size[1]
        new_width = int(target_size[1] * img_ratio)
    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    img_padded = ImageOps.pad(img, target_size, color=(0, 0, 0), centering=(0.5, 0.5))
    buf = io.BytesIO()
    img_padded.save(buf, format="JPEG")
    return buf.getvalue()


def encode_image(image_path, resize=False, target_size=(512, 512)):
    if resize:
        if is_url(image_path):
            img_bytes = resize_with_padding(requests.get(image_path).content, target_size)
        else:
            img_bytes = resize_with_padding(image_path, target_size)
        return base64.b64encode(img_bytes).decode("utf-8")
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ---------------------------------------------------------------------------
# Request / response primitives
# ---------------------------------------------------------------------------

def create_payload(prompt, image_paths, model, detail, max_tokens, resize=False, target_size=(512, 512)):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        "max_tokens": max_tokens,
    }
    for image_path in image_paths:
        if is_url(image_path) and not resize:
            img_block = {"type": "image_url", "image_url": {"url": image_path, "detail": detail}}
        else:
            b64 = encode_image(image_path, resize, target_size)
            img_block = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": detail}}
        payload["messages"][0]["content"].append(img_block)
    return payload


def send_request(api_key, url, payload):
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        raise RuntimeError(f"API request failed: {response.status_code} {response.text}")
    return response.json()


def normalize_response(response, provider="openai") -> str:
    """Extract text content from any supported provider response shape."""
    try:
        return response["choices"][0]["message"]["content"]  # OpenAI-compatible
    except (KeyError, IndexError):
        pass
    try:
        return response["content"][0]["text"]  # Anthropic
    except (KeyError, IndexError):
        pass
    try:
        return response["candidates"][0]["content"]["parts"][0]["text"]  # Google
    except (KeyError, IndexError):
        pass
    return json.dumps(response)


def run_agent(agent: Agent, prompt: str, image_paths: List[str], detail, max_tokens, resize, target_size) -> AgentResult:
    """Execute a single agent call and return an AgentResult."""
    payload = create_payload(prompt, image_paths, agent.model, detail, max_tokens, resize, target_size)
    t0 = time.time()
    try:
        response = send_request(agent.api_key, agent.endpoint, payload)
        latency_ms = (time.time() - t0) * 1000
        return AgentResult(
            agent_id=agent.agent_id,
            model=agent.model,
            provider=agent.provider,
            content=normalize_response(response, agent.provider),
            raw_response=response,
            latency_ms=latency_ms,
        )
    except Exception as exc:
        return AgentResult(
            agent_id=agent.agent_id,
            model=agent.model,
            provider=agent.provider,
            content="",
            raw_response={},
            latency_ms=(time.time() - t0) * 1000,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Workflow: Sequential
#
# Agents execute one after another in a defined order.
# Each agent receives the original prompt PLUS all prior agents' outputs as
# accumulated context, allowing each stage to build on the previous analysis.
#
# Input ──► [Agent-1] ──► output-1
#                           │
#           prompt + output-1 ──► [Agent-2] ──► output-2
#                                                 │
#                    prompt + output-1 + output-2 ──► [Agent-3] ──► final
# ---------------------------------------------------------------------------

def run_sequential(agents: List[Agent], prompt: str, image_paths, detail, max_tokens, resize, target_size):
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


# ---------------------------------------------------------------------------
# Workflow: Parallel
#
# All agents receive the same input simultaneously. Results are collected once
# all branches complete and the best response is selected.
#
# Input ──► [Agent-1] ─┐
# Input ──► [Agent-2] ─┤──► [Aggregator] ──► final
# Input ──► [Agent-3] ─┘
# ---------------------------------------------------------------------------

def run_parallel(agents: List[Agent], prompt: str, image_paths, detail, max_tokens, resize, target_size):
    if len(agents) < 2:
        raise ValueError("parallel workflow requires at least 2 agents")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(run_agent, a, prompt, image_paths, detail, max_tokens, resize, target_size): a
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


# ---------------------------------------------------------------------------
# Workflow: Conditional
#
# A lightweight router agent first classifies the input into one of the known
# categories. The matching specialist agent then handles the full request.
# If the router's answer doesn't match any category, the first specialist is
# used as the default fallback.
#
# Input ──► [Router Agent] ──► category
#                                 │
#             ┌───────────────────┼──────────────────┐
#         [Specialist-A]   [Specialist-B]   [Specialist-C]
#          (if cat=A)       (if cat=B)       (if cat=C)
#                                 │
#                              final output
# ---------------------------------------------------------------------------

def run_conditional(
    router_agent: Agent,
    specialist_agents: List[Agent],
    categories: List[str],
    prompt: str,
    image_paths,
    detail,
    max_tokens,
    resize,
    target_size,
):
    if len(specialist_agents) != len(categories):
        raise ValueError("specialist_agents and categories must have the same length")

    # Step 1 — Router classifies input
    router_prompt = (
        f"Analyze the following image and/or prompt and classify it into exactly one of these "
        f"categories: {categories}.\n"
        "Reply with ONLY the category name, nothing else.\n\n"
        f"Prompt: {prompt}"
    )
    router_result = run_agent(router_agent, router_prompt, image_paths, detail, max_tokens, resize, target_size)
    if router_result.error:
        raise RuntimeError(f"Router agent failed: {router_result.error}")

    route = router_result.content.strip().lower()

    # Step 2 — Find matching specialist (case-insensitive, fallback to first)
    specialist = None
    matched_category = None
    for cat, spec in zip(categories, specialist_agents):
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


# ---------------------------------------------------------------------------
# Workflow: Iterative
#
# The agent runs in a loop. After each iteration the evaluator (or a simple
# heuristic) judges whether the output has converged. If not, the output is
# fed back as accumulated context so the next iteration can refine it.
# The loop stops when converged or max_iterations is reached.
#
# prompt ──► [Agent] ──► output-1
#                           │
#                     [Evaluator] ── not converged ──► prompt + output-1 ──► [Agent] ──► output-2
#                           │                                                        │
#                       converged ◄──────────────────────────────────── [Evaluator] ─┘
#                           │
#                        final output
# ---------------------------------------------------------------------------

def run_iterative(
    agent: Agent,
    prompt: str,
    image_paths,
    detail,
    max_tokens,
    resize,
    target_size,
    evaluator_agent: Optional[Agent] = None,
    max_iterations: int = 3,
):
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
            eval_result = run_agent(evaluator_agent, eval_prompt, [], detail, max_tokens, resize, target_size)
            try:
                score = int("".join(filter(str.isdigit, eval_result.content.strip()))[:2] or "0")
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


# ---------------------------------------------------------------------------
# Workflow: Mixture-of-Agents (MoA)
#
# Multiple proposer agents process the input in parallel. Their outputs are
# all passed to an aggregator agent that synthesizes a single final answer.
#
# Input ──► [Proposer-1] ─┐
# Input ──► [Proposer-2] ─┤──► candidates ──► [Aggregator] ──► final
# Input ──► [Proposer-3] ─┘
# ---------------------------------------------------------------------------

def run_moa(
    proposer_agents: List[Agent],
    aggregator_agent: Agent,
    prompt: str,
    image_paths,
    detail,
    max_tokens,
    resize,
    target_size,
):
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
        f"[Candidate {i + 1} — {r.model} / {r.provider}]\n{r.content}"
        for i, r in enumerate(successful)
    )
    aggregator_prompt = (
        f"You are an impartial synthesizer. Below are {len(successful)} candidate answers "
        f"to the question: \"{prompt}\"\n\n"
        f"{candidates_block}\n\n"
        "Compare the candidates, extract consensus points, resolve conflicts, "
        "and produce one final, comprehensive best answer."
    )
    agg_result = run_agent(
        aggregator_agent, aggregator_prompt, image_paths, detail, max_tokens, resize, target_size
    )

    return {
        "workflow": "moa",
        "aggregator_model": aggregator_agent.model,
        "aggregator_provider": aggregator_agent.provider,
        "aggregator_latency_ms": round(agg_result.latency_ms, 1),
        "content": agg_result.content if not agg_result.error else successful[0].content,
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


# ---------------------------------------------------------------------------
# Workflow: ReAct (Reasoning + Acting)
#
# The agent interleaves Thought / Action / Observation steps.
# Each Action calls a registered tool. The Observation is appended back
# into the conversation so the agent can reason about it and decide the
# next action. The loop ends when the agent emits "Final Answer:" or the
# step budget is exhausted.
#
# prompt ──► [Agent: Thought + Action] ──► tool call
#                        ▲                       │
#                        └── Observation ◄───────┘
#                        (repeated up to max_steps)
#                        ──► Final Answer
# ---------------------------------------------------------------------------

@dataclass
class Tool:
    name: str
    description: str
    parameters: str          # JSON schema hint shown to the agent
    fn: Callable             # fn(agent, image_paths, detail, max_tokens, resize, target_size, **kwargs) -> str


def _tool_describe(agent, image_paths, detail, max_tokens, resize, target_size, **kwargs) -> str:
    """Ask the VLM to describe the image(s)."""
    prompt = kwargs.get("prompt", "Describe this image in detail.")
    result = run_agent(agent, prompt, image_paths, detail, max_tokens, resize, target_size)
    return result.content if result.success else f"ERROR: {result.error}"


def _tool_detect_objects(agent, image_paths, detail, max_tokens, resize, target_size, **kwargs) -> str:
    """Ask the VLM to list all objects visible in the image(s)."""
    result = run_agent(
        agent,
        "List every distinct object you can see in this image. Return as a JSON array of strings.",
        image_paths, detail, max_tokens, resize, target_size,
    )
    return result.content if result.success else f"ERROR: {result.error}"


def _tool_read_text(agent, image_paths, detail, max_tokens, resize, target_size, **kwargs) -> str:
    """Ask the VLM to extract all visible text (OCR) from the image(s)."""
    result = run_agent(
        agent,
        "Extract and return all text visible in this image, preserving the reading order.",
        image_paths, detail, max_tokens, resize, target_size,
    )
    return result.content if result.success else f"ERROR: {result.error}"


def _tool_analyze_region(agent, image_paths, detail, max_tokens, resize, target_size, **kwargs) -> str:
    """Ask the VLM to focus analysis on a described region of the image."""
    region = kwargs.get("region", "the center of the image")
    question = kwargs.get("question", "What do you see?")
    result = run_agent(
        agent,
        f"Focus only on {region}. {question}",
        image_paths, detail, max_tokens, resize, target_size,
    )
    return result.content if result.success else f"ERROR: {result.error}"


def _tool_count_objects(agent, image_paths, detail, max_tokens, resize, target_size, **kwargs) -> str:
    """Ask the VLM to count occurrences of a specific object."""
    object_name = kwargs.get("object", "objects")
    result = run_agent(
        agent,
        f"Count exactly how many '{object_name}' are visible in this image. "
        "Return only an integer.",
        image_paths, detail, max_tokens, resize, target_size,
    )
    return result.content if result.success else f"ERROR: {result.error}"


BUILTIN_TOOLS: Dict[str, Tool] = {
    "describe": Tool(
        name="describe",
        description="Generate a detailed description of the image(s).",
        parameters='{"prompt": "optional focus instruction (string)"}',
        fn=_tool_describe,
    ),
    "detect_objects": Tool(
        name="detect_objects",
        description="List all distinct objects visible in the image(s).",
        parameters="{}",
        fn=_tool_detect_objects,
    ),
    "read_text": Tool(
        name="read_text",
        description="Extract all visible text from the image(s) (OCR).",
        parameters="{}",
        fn=_tool_read_text,
    ),
    "analyze_region": Tool(
        name="analyze_region",
        description="Focus analysis on a specific region of the image.",
        parameters='{"region": "description of the region, e.g. top-left corner", "question": "what to answer about that region"}',
        fn=_tool_analyze_region,
    ),
    "count_objects": Tool(
        name="count_objects",
        description="Count occurrences of a specific object in the image(s).",
        parameters='{"object": "name of the object to count"}',
        fn=_tool_count_objects,
    ),
}


def _parse_react_step(text: str):
    """Parse one ReAct step from model output.
    Returns (thought, action, action_input_dict, final_answer) where
    final_answer is non-None only when the agent is done.
    """
    final_match = re.search(r"Final Answer\s*:\s*(.+)", text, re.DOTALL | re.IGNORECASE)
    if final_match:
        return None, None, {}, final_match.group(1).strip()

    thought_match = re.search(r"Thought\s*:\s*(.+?)(?=Action\s*:|$)", text, re.DOTALL | re.IGNORECASE)
    action_match = re.search(r"Action\s*:\s*(\w+)", text, re.IGNORECASE)
    input_match = re.search(r"Action Input\s*:\s*(\{.*?\}|\S.*?)(?=\nObservation|\nThought|\nAction|$)", text, re.DOTALL | re.IGNORECASE)

    thought = thought_match.group(1).strip() if thought_match else ""
    action = action_match.group(1).strip() if action_match else ""
    try:
        action_input = json.loads(input_match.group(1).strip()) if input_match else {}
    except (json.JSONDecodeError, AttributeError):
        action_input = {}

    return thought, action, action_input, None


def run_react(
    agent: Agent,
    prompt: str,
    image_paths: List[str],
    detail,
    max_tokens,
    resize,
    target_size,
    enabled_tools: Optional[List[str]] = None,
    max_steps: int = 5,
):
    tools: Dict[str, Tool] = (
        {k: v for k, v in BUILTIN_TOOLS.items() if k in enabled_tools}
        if enabled_tools
        else BUILTIN_TOOLS
    )
    if not tools:
        raise ValueError(f"No valid tools enabled. Available: {list(BUILTIN_TOOLS.keys())}")

    tool_descriptions = "\n".join(
        f"  {t.name}: {t.description} | parameters: {t.parameters}"
        for t in tools.values()
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
        conversation += (
            f"\n{model_output.rstrip()}"
            f"\nObservation: {observation}\n"
        )
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


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def make_agent(model, provider, endpoint) -> Agent:
    env_var = PROVIDER_ENV_MAP.get(provider.lower(), "OPENAI_API_KEY")
    api_key = os.getenv(env_var)
    if api_key is None:
        raise RuntimeError(f"{env_var} not set (required for provider '{provider}')")
    return Agent(model=model, endpoint=endpoint, api_key=api_key, provider=provider)


def build_agents(models, providers, endpoints) -> List[Agent]:
    n = len(models)
    providers = providers or ["openai"] * n
    endpoints = endpoints or [DEFAULT_ENDPOINT] * n
    if len(providers) != n:
        raise ValueError(f"--providers length ({len(providers)}) must match --models length ({n})")
    if len(endpoints) != n:
        raise ValueError(f"--endpoints length ({len(endpoints)}) must match --models length ({n})")
    return [make_agent(m, p, e) for m, p, e in zip(models, providers, endpoints)]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    correlation_id = str(uuid.uuid4())[:12]

    parser = argparse.ArgumentParser(
        prog="vlm-agent-gateway",
        description=(
            "VLM Agent Gateway — multi-provider vision inference with "
            "sequential, parallel, conditional, iterative, and MoA workflows"
        ),
    )

    # ── Image / prompt ──────────────────────────────────────────────────────
    parser.add_argument("--prompt", "-p", type=str, default="What's in this image?")
    parser.add_argument("--images", "-i", type=str, nargs="+", required=True, help="Image paths or URLs")
    parser.add_argument("--detail", "-d", type=str, default="low", choices=["auto", "low", "high"])
    parser.add_argument("--tokens", "-t", type=int, default=300, help="Max tokens per response")
    parser.add_argument("--resize", "-r", action="store_true", help="Resize images with padding")
    parser.add_argument("--size", "-s", type=int, nargs=2, default=[512, 512], metavar=("W", "H"))

    # ── Workflow ────────────────────────────────────────────────────────────
    parser.add_argument(
        "--workflow", "-w", type=str, default="sequential", choices=WORKFLOW_CHOICES,
        help="sequential | parallel | conditional | iterative | moa  (default: sequential)",
    )

    # ── Agent targets ───────────────────────────────────────────────────────
    parser.add_argument("--models", type=str, nargs="+", metavar="MODEL",
                        help="One model per agent (zip-paired with --providers and --endpoints)")
    parser.add_argument("--providers", type=str, nargs="+", metavar="PROVIDER",
                        help=f"Provider per agent. Known: {list(PROVIDER_ENV_MAP.keys())}")
    parser.add_argument("--endpoints", type=str, nargs="+", metavar="URL",
                        help="API endpoint per agent")
    # Single-agent fallback (backward-compatible)
    parser.add_argument("--model", "-m", type=str, default="gpt-4o-mini")
    parser.add_argument("--url", "-u", type=str, default=DEFAULT_ENDPOINT)
    parser.add_argument("--provider", type=str, default="openai")

    # ── Special-role agents ─────────────────────────────────────────────────
    parser.add_argument("--aggregator-model", type=str, default=None,
                        help="moa: aggregator model (default: first model)")
    parser.add_argument("--aggregator-provider", type=str, default=None)
    parser.add_argument("--aggregator-endpoint", type=str, default=None)

    parser.add_argument("--router-model", type=str, default=None,
                        help="conditional: router/classifier model (default: first model)")
    parser.add_argument("--router-provider", type=str, default=None)
    parser.add_argument("--router-endpoint", type=str, default=None)
    parser.add_argument("--categories", type=str, nargs="+", default=["general"],
                        help="conditional: category labels to route between (one per --models entry)")

    parser.add_argument("--evaluator-model", type=str, default=None,
                        help="iterative: evaluator model; omit for heuristic evaluation")
    parser.add_argument("--evaluator-provider", type=str, default=None)
    parser.add_argument("--evaluator-endpoint", type=str, default=None)
    parser.add_argument("--max-iterations", type=int, default=3,
                        help="iterative: maximum refinement iterations (default: 3)")

    # ── ReAct args ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--tools", type=str, nargs="+", default=None,
        metavar="TOOL",
        help=(
            f"react: tools to enable. Available: {list(BUILTIN_TOOLS.keys())}. "
            "Omit to enable all tools."
        ),
    )
    parser.add_argument("--max-steps", type=int, default=5,
                        help="react: maximum Thought/Action/Observation steps (default: 5)")

    args = parser.parse_args()

    try:
        models = args.models if args.models else [args.model]
        providers = args.providers or [args.provider] * len(models)
        endpoints = args.endpoints or [args.url] * len(models)
        agents = build_agents(models, providers, endpoints)

        common = dict(
            prompt=args.prompt,
            image_paths=args.images,
            detail=args.detail,
            max_tokens=args.tokens,
            resize=args.resize,
            target_size=tuple(args.size),
        )

        if args.workflow == "sequential":
            output = run_sequential(agents, **common)

        elif args.workflow == "parallel":
            output = run_parallel(agents, **common)

        elif args.workflow == "conditional":
            router_model = args.router_model or agents[0].model
            router_provider = args.router_provider or agents[0].provider
            router_endpoint = args.router_endpoint or agents[0].endpoint
            router = make_agent(router_model, router_provider, router_endpoint)
            if len(args.categories) != len(agents):
                raise ValueError(
                    f"--categories ({len(args.categories)}) must match --models ({len(agents)})"
                )
            output = run_conditional(router, agents, args.categories, **common)

        elif args.workflow == "iterative":
            evaluator = None
            if args.evaluator_model:
                evaluator = make_agent(
                    args.evaluator_model,
                    args.evaluator_provider or agents[0].provider,
                    args.evaluator_endpoint or agents[0].endpoint,
                )
            output = run_iterative(
                agents[0], **common,
                evaluator_agent=evaluator,
                max_iterations=args.max_iterations,
            )

        elif args.workflow == "moa":
            agg_model = args.aggregator_model or agents[0].model
            agg_provider = args.aggregator_provider or agents[0].provider
            agg_endpoint = args.aggregator_endpoint or agents[0].endpoint
            aggregator = make_agent(agg_model, agg_provider, agg_endpoint)
            output = run_moa(agents, aggregator, **common)

        elif args.workflow == "react":
            output = run_react(
                agents[0], **common,
                enabled_tools=args.tools,
                max_steps=args.max_steps,
            )

        output["correlation_id"] = correlation_id
        print(json.dumps(output, indent=2))

    except Exception as exc:
        print(json.dumps({"error": str(exc), "correlation_id": correlation_id}, indent=2))


if __name__ == "__main__":
    main()
