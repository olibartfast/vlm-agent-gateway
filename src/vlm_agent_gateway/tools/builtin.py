"""
Built-in ReAct tools for vision analysis.
"""


from vlm_agent_gateway.models import Agent, Tool
from vlm_agent_gateway.providers import run_agent


def _tool_describe(
    agent: Agent,
    image_paths: list[str],
    detail: str,
    max_tokens: int,
    resize: bool,
    target_size: tuple[int, int],
    **kwargs,
) -> str:
    """Ask the VLM to describe the image(s)."""
    prompt = kwargs.get("prompt", "Describe this image in detail.")
    result = run_agent(agent, prompt, image_paths, detail, max_tokens, resize, target_size)
    return result.content if result.success else f"ERROR: {result.error}"


def _tool_detect_objects(
    agent: Agent,
    image_paths: list[str],
    detail: str,
    max_tokens: int,
    resize: bool,
    target_size: tuple[int, int],
    **kwargs,
) -> str:
    """Ask the VLM to list all objects visible in the image(s)."""
    result = run_agent(
        agent,
        "List every distinct object you can see in this image. Return as a JSON array of strings.",
        image_paths,
        detail,
        max_tokens,
        resize,
        target_size,
    )
    return result.content if result.success else f"ERROR: {result.error}"


def _tool_read_text(
    agent: Agent,
    image_paths: list[str],
    detail: str,
    max_tokens: int,
    resize: bool,
    target_size: tuple[int, int],
    **kwargs,
) -> str:
    """Ask the VLM to extract all visible text (OCR) from the image(s)."""
    result = run_agent(
        agent,
        "Extract and return all text visible in this image, preserving the reading order.",
        image_paths,
        detail,
        max_tokens,
        resize,
        target_size,
    )
    return result.content if result.success else f"ERROR: {result.error}"


def _tool_analyze_region(
    agent: Agent,
    image_paths: list[str],
    detail: str,
    max_tokens: int,
    resize: bool,
    target_size: tuple[int, int],
    **kwargs,
) -> str:
    """Ask the VLM to focus analysis on a described region of the image."""
    region = kwargs.get("region", "the center of the image")
    question = kwargs.get("question", "What do you see?")
    result = run_agent(
        agent,
        f"Focus only on {region}. {question}",
        image_paths,
        detail,
        max_tokens,
        resize,
        target_size,
    )
    return result.content if result.success else f"ERROR: {result.error}"


def _tool_count_objects(
    agent: Agent,
    image_paths: list[str],
    detail: str,
    max_tokens: int,
    resize: bool,
    target_size: tuple[int, int],
    **kwargs,
) -> str:
    """Ask the VLM to count occurrences of a specific object."""
    object_name = kwargs.get("object", "objects")
    result = run_agent(
        agent,
        f"Count exactly how many '{object_name}' are visible in this image. "
        "Return only an integer.",
        image_paths,
        detail,
        max_tokens,
        resize,
        target_size,
    )
    return result.content if result.success else f"ERROR: {result.error}"


BUILTIN_TOOLS: dict[str, Tool] = {
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
