"""
VLM Agent Gateway - Multi-provider vision inference with workflow support.

A framework for building agentic applications with Vision Language Models.
Supports 7 VLM providers (OpenAI, Anthropic, Google, Together, Azure, Groq, Mistral)
and 7 workflow patterns (sequential, parallel, conditional, iterative, MoA, ReAct, monitoring).
"""

from pathlib import Path

# Core models
# Configuration
from vlm_agent_gateway.config import (
    DEFAULT_ENDPOINT,
    PROVIDER_ENV_MAP,
    WORKFLOW_CHOICES,
    get_api_key,
    get_default_endpoint,
    resolve_endpoint,
)

# Image utilities
from vlm_agent_gateway.image import encode_image, is_url, resize_with_padding
from vlm_agent_gateway.models import Agent, AgentResult, AlertEvent, Tool

# Provider functions
from vlm_agent_gateway.providers import (
    build_video_payload,
    create_payload,
    normalize_response,
    run_agent,
    send_request,
)

# Tools
from vlm_agent_gateway.tools import BUILTIN_TOOLS

# Video utilities
from vlm_agent_gateway.video import extract_frames_cv2, frames_to_base64

# Workflows
from vlm_agent_gateway.workflows import (
    WORKFLOW_REGISTRY,
    run_conditional,
    run_continuous_monitoring,
    run_iterative,
    run_moa,
    run_monitoring,
    run_monitoring_cycle,
    run_parallel,
    run_react,
    run_sequential,
)


def _read_version() -> str:
    """Read version from VERSION file."""
    version_file = Path(__file__).parent.parent.parent / "VERSION"
    if version_file.exists():
        return version_file.read_text().strip()
    return "0.0.0"


__version__ = _read_version()

__all__ = [
    # Version
    "__version__",
    # Models
    "Agent",
    "AgentResult",
    "AlertEvent",
    "Tool",
    # Config
    "DEFAULT_ENDPOINT",
    "PROVIDER_ENV_MAP",
    "WORKFLOW_CHOICES",
    "get_api_key",
    "get_default_endpoint",
    "resolve_endpoint",
    # Image
    "encode_image",
    "is_url",
    "resize_with_padding",
    # Video
    "extract_frames_cv2",
    "frames_to_base64",
    # Providers
    "build_video_payload",
    "create_payload",
    "normalize_response",
    "run_agent",
    "send_request",
    # Tools
    "BUILTIN_TOOLS",
    # Workflows
    "WORKFLOW_REGISTRY",
    "run_conditional",
    "run_continuous_monitoring",
    "run_iterative",
    "run_moa",
    "run_monitoring",
    "run_monitoring_cycle",
    "run_parallel",
    "run_react",
    "run_sequential",
]
