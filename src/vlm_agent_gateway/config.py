"""
Configuration constants and environment handling.
"""

import os

# ---------------------------------------------------------------------------
# Provider configuration
# ---------------------------------------------------------------------------

PROVIDER_ENV_MAP: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "together": "TOGETHER_API_KEY",
    "azure": "AZURE_OPENAI_API_KEY",
    "groq": "GROQ_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
}

PROVIDER_DEFAULTS: dict[str, str] = {
    "openai": "https://api.openai.com/v1/chat/completions",
    "anthropic": "https://api.anthropic.com/v1/chat/completions",
    "google": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
    "together": "https://api.together.xyz/v1/chat/completions",
    "groq": "https://api.groq.com/openai/v1/chat/completions",
    "mistral": "https://api.mistral.ai/v1/chat/completions",
    "cerebras": "https://api.cerebras.ai/v1/chat/completions",
}


DEFAULT_ENDPOINT = "https://api.openai.com/v1/chat/completions"

WORKFLOW_CHOICES = ["sequential", "parallel", "conditional", "iterative", "moa", "react", "monitor"]

# ---------------------------------------------------------------------------
# ReAct system prompt template
# ---------------------------------------------------------------------------

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
# Video monitoring system prompt
# ---------------------------------------------------------------------------

MONITOR_SYSTEM_PROMPT = """You are a video monitoring agent. You receive a sequence of video frames
and must analyze them for the user's specified condition.

For EACH analysis cycle you MUST output exactly one of these two formats:

FORMAT A — If the condition IS detected:
Thought: <your reasoning about what you see across the frames>
Alert: YES
Summary: <concise description of the detected event, including which frames show it>
Confidence: <HIGH / MEDIUM / LOW>
Recommended Action: <what a human operator should do>

FORMAT B — If the condition is NOT detected:
Thought: <your reasoning about what you see across the frames>
Alert: NO
Summary: <brief description of normal scene activity>

Rules:
- Analyze ALL frames as a temporal sequence (frame 1 is earliest, last frame is latest).
- Look for changes between frames — motion, appearance/disappearance of objects, posture changes.
- Be precise about spatial locations and temporal progression.
- If uncertain, set Confidence to LOW rather than generating a false alert.
- Never fabricate observations; base everything on what you actually see in the frames."""


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def get_api_key(provider: str) -> str:
    """Get API key from environment for a provider."""
    env_var = PROVIDER_ENV_MAP.get(provider.lower(), "OPENAI_API_KEY")
    api_key = os.getenv(env_var)
    if api_key is None:
        raise RuntimeError(f"{env_var} not set (required for provider '{provider}')")
    return api_key


def get_default_endpoint(provider: str) -> str:
    """Get default endpoint for a provider."""
    return PROVIDER_DEFAULTS.get(provider.lower(), DEFAULT_ENDPOINT)


def resolve_endpoint(provider: str, endpoint: str | None = None) -> str:
    """Resolve an endpoint for a provider, requiring explicit URLs when needed."""
    if endpoint:
        return endpoint

    provider_name = provider.lower()
    if provider_name in PROVIDER_DEFAULTS:
        return PROVIDER_DEFAULTS[provider_name]
    if provider_name in PROVIDER_ENV_MAP:
        raise RuntimeError(f"Provider '{provider}' requires an explicit --endpoint value.")
    return DEFAULT_ENDPOINT
